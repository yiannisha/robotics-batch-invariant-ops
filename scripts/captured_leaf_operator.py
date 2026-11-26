"""Capture and independently replay supported eager leaf operations.

Local output reproduction alone does not establish singleton/batched stability
or globally first operator attribution. Callers must additionally verify target
input identity, semantic projection, and the preceding execution boundary.
"""
import contextlib
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as functional

from batch_invariant_ops import set_batch_invariant_mode
from batch_invariant_ops import batch_invariant_ops as invariant_implementation
from batch_invariant_ops.batch_invariant_ops import is_batch_invariant_mode_enabled
from scripts.captured_tensor_layout import capture, restore, byte_hash, DTYPES

REDUCTION_FLAGS = ('allow_fp16_reduced_precision_reduction', 'allow_bf16_reduced_precision_reduction',
                   'allow_fp16_accumulation')
NUMERICAL_ENV = ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
                 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLASLT_WORKSPACE_SIZE')


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_operator(module, operand, output, *, provenance):
    check(type(module) in (torch.nn.Linear, torch.nn.Conv2d), 'Unsupported exact leaf module type')
    check(not module.training and torch.is_inference_mode_enabled(), 'Requires eval and inference mode')
    check(isinstance(operand, torch.Tensor) and isinstance(output, torch.Tensor), 'Tensor input/output required')
    check(provenance and isinstance(provenance, dict), 'Source provenance required')
    device_type = operand.device.type
    check(device_type in ('cpu', 'cuda'), 'Unsupported replay device')
    check(module.weight.device == operand.device, 'Mixed operator devices')
    if type(module) is torch.nn.Conv2d:
        check(module.padding_mode == 'zeros', 'Nonzero convolution padding mode is not qualified')
        configuration = dict(stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
    else:
        configuration = {}
    runtime = dict(device_type=device_type, torch_version=torch.__version__,
                   cuda_version=torch.version.cuda, matmul_precision=torch.get_float32_matmul_precision(),
                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_enabled=torch.backends.cudnn.enabled,
                   cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                   autocast_enabled=torch.is_autocast_enabled(device_type),
                   autocast_dtype=str(torch.get_autocast_dtype(device_type)),
                   cuda_reduction_flags={k: getattr(torch.backends.cuda.matmul, k, None) for k in REDUCTION_FLAGS},
                   numerical_environment={k: os.environ.get(k) for k in NUMERICAL_ENV},
                   invariant_mode=is_batch_invariant_mode_enabled())
    if device_type == 'cuda':
        selector = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        check(selector.startswith('GPU-') and ',' not in selector, 'Single UUID-selected GPU required')
        runtime.update(gpu_uuid_selector=selector, gpu_name=torch.cuda.get_device_name(operand.device),
                       gpu_capability=list(torch.cuda.get_device_capability(operand.device)),
                       cudnn_version=torch.backends.cudnn.version())
    return dict(schema_version=1, operator='linear' if type(module) is torch.nn.Linear else 'conv2d',
                configuration=configuration, provenance=provenance, runtime=runtime,
                input=capture(operand), weight=capture(module.weight),
                bias=capture(module.bias) if module.bias is not None else None,
                output=capture(output), helper_sha256=sha(Path(__file__)),
                operator_library_sha256=sha(Path(invariant_implementation.__file__)),
                layout_helper_sha256=sha(Path(__file__).with_name('captured_tensor_layout.py')))


@contextlib.contextmanager
def recorded_runtime(runtime):
    original = dict(matmul_precision=torch.get_float32_matmul_precision(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_enabled=torch.backends.cudnn.enabled,
                    cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
    original['cuda_reduction_flags'] = {k: getattr(torch.backends.cuda.matmul, k, None) for k in REDUCTION_FLAGS}
    def apply(values):
        torch.set_float32_matmul_precision(values['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = values['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = values['cudnn_tf32']
        torch.backends.cudnn.enabled = values['cudnn_enabled']
        torch.backends.cudnn.benchmark = values['cudnn_benchmark']
        torch.backends.cudnn.deterministic = values['cudnn_deterministic']
        for key, value in values['cuda_reduction_flags'].items():
            if value is not None:
                setattr(torch.backends.cuda.matmul, key, value)
        torch.use_deterministic_algorithms(values['deterministic_algorithms'], warn_only=values['deterministic_warn_only'])
    try:
        apply(runtime)
        with set_batch_invariant_mode(runtime['invariant_mode']), torch.inference_mode(), torch.autocast(
                runtime['device_type'], enabled=runtime['autocast_enabled'], dtype=DTYPES[runtime['autocast_dtype']]):
            yield
    finally:
        apply(original)


def replay_operator(record, device, *, require_pointer_alignment=True, maximum_storage_bytes=None):
    check(record['schema_version'] == 1 and record['operator'] in ('linear', 'conv2d'), 'Unknown captured operator schema')
    check(record['helper_sha256'] == sha(Path(__file__))
          and record['operator_library_sha256'] == sha(Path(invariant_implementation.__file__))
          and record['layout_helper_sha256'] == sha(Path(__file__).with_name('captured_tensor_layout.py')),
          'Captured replay implementation changed')
    runtime = record['runtime']
    target = torch.device(device)
    check(target.type == runtime['device_type'] and runtime['torch_version'] == torch.__version__, 'Replay device/runtime differs')
    if target.type == 'cuda':
        check(require_pointer_alignment, 'GPU replay cannot disable operand alignment checks')
        check(runtime['numerical_environment'] == {k: os.environ.get(k) for k in NUMERICAL_ENV},
              'CUDA numerical environment differs')
        check(runtime['cuda_version'] == torch.version.cuda
              and runtime['gpu_uuid_selector'] == os.environ.get('CUDA_VISIBLE_DEVICES')
              and runtime['gpu_name'] == torch.cuda.get_device_name(target)
              and runtime['gpu_capability'] == list(torch.cuda.get_device_capability(target))
              and runtime['cudnn_version'] == torch.backends.cudnn.version(), 'Pinned CUDA execution runtime differs')
    # Verify the retained expected output bytes without relying on its hash alone.
    expected = restore(record['output'], 'cpu', require_pointer_alignment=False,
                       maximum_storage_bytes=maximum_storage_bytes)
    expected_hash = byte_hash(expected)
    results = []
    with recorded_runtime(runtime):
        for _ in range(2):
            operands = {key: restore(record[key], target, require_pointer_alignment=require_pointer_alignment,
                                     maximum_storage_bytes=maximum_storage_bytes) if record[key] is not None else None
                        for key in ('input', 'weight', 'bias')}
            if record['operator'] == 'linear':
                output = functional.linear(operands['input'], operands['weight'], operands['bias'])
            else:
                output = functional.conv2d(operands['input'], operands['weight'], operands['bias'], **record['configuration'])
            check(list(output.shape) == record['output']['shape'] and str(output.dtype) == record['output']['dtype'],
                  'Isolated output schema differs')
            check(list(output.stride()) == record['output']['strides'], 'Isolated output layout differs')
            value = byte_hash(output)
            check(value == expected_hash, 'Isolated operator output does not reproduce captured bytes')
            results.append(value)
    return dict(status='captured_leaf_output_reproduced_twice', operator=record['operator'],
                output_sha256=expected_hash, repeats_exact=results[0] == results[1],
                pointer_alignment_verified=require_pointer_alignment,
                globally_first_operator_attribution_complete=False)

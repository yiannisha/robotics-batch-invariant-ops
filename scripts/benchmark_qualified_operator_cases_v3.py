"""Time source-bound operator cases; use only through the isolation wrapper."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import triton

from batch_invariant_ops import set_batch_invariant_mode
from scripts.operator_performance_operands import operands, operation, tensor_signatures, variant_operands


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def validate_plan(path):
    plan = json.loads(path.read_text())
    if (plan['status'] != 'operator_timing_plan_only_execution_and_independent_acceptance_pending'
            or plan['timing_protocol'] != dict(sessions=5, warmup_calls=20, timed_calls=200)):
        raise ValueError('Wrong operator timing protocol')
    for filename, expected in plan['source_sha256'].items():
        if sha(filename) != expected:
            raise ValueError('Plan source changed: ' + filename)
    builder = Path(__file__).with_name('build_operator_performance_plans_v2.py')
    if sha(builder) != plan['builder_sha256']:
        raise ValueError('Plan builder changed')
    ids = [case['id'] for case in plan['cases']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate operator cases')
    for case in plan['cases']:
        expected = (['whole', 'attention_qk', 'attention_pv']
                    if case['entry']['aten_operation'] == 'aten::scaled_dot_product_attention' else ['whole'])
        if case['variants'] != expected:
            raise ValueError('Missing native-BMM attention component comparison')
    return plan


def record_seed(model, case_id, session):
    return int.from_bytes(hashlib.sha256(f'{model}/{case_id}/{session}/20261109'.encode()).digest()[:8], 'big') % (2 ** 63)


def measure(function):
    # Mode selection, operand creation, tracing and event allocation are outside
    # every measured region. Warm-up includes first-use compilation.
    for _ in range(20):
        function()
    torch.cuda.synchronize()
    reference = function()
    if not torch.isfinite(reference).all():
        raise ValueError('Nonfinite operator result')
    reference_signature = tensor_signatures(reference)
    del reference
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
    # CUDA events lazily allocate device resources on first record. Prime all
    # events before the timed loop so neither event initialization nor its
    # host-side gap contaminates CUDA-event or synchronized wall samples.
    for start, end in zip(starts, ends):
        start.record()
        end.record()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cuda_seconds, wall_seconds = [], []
    result = None
    for start, end in zip(starts, ends):
        begin = time.perf_counter()
        start.record()
        result = function()
        end.record()
        end.synchronize()
        wall_seconds.append(time.perf_counter() - begin)
        cuda_seconds.append(start.elapsed_time(end) / 1000)
    if not torch.isfinite(result).all() or tensor_signatures(result) != reference_signature:
        raise ValueError('Same-input operator output changed during timing')
    if not all(math.isfinite(v) and v > 0 for v in cuda_seconds + wall_seconds):
        raise ValueError('Invalid timing sample')
    return dict(status='timed', cuda_seconds=cuda_seconds, synchronized_wall_seconds=wall_seconds,
                median_cuda_seconds=statistics.median(cuda_seconds),
                p95_cuda_seconds=quantile(cuda_seconds, .95),
                median_synchronized_wall_seconds=statistics.median(wall_seconds),
                p95_synchronized_wall_seconds=quantile(wall_seconds, .95),
                whole_operation_calls_per_second=1 / statistics.median(cuda_seconds),
                peak_incremental_allocated_bytes=torch.cuda.max_memory_allocated() - baseline,
                output_signature=reference_signature, same_case_output_repeat_exact=True,
                kernel_launch_count=None,
                kernel_launch_count_reason='Profiler/tracing disabled; launch count not obtained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid or not args.gpu_uuid.startswith('GPU-'):
        raise ValueError('Explicit isolated GPU UUID required')
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = validate_plan(args.plan)
    plan_hash = sha(args.plan)
    if plan['adapter_execution_configuration']['tf32'] is not False:
        raise ValueError('Unqualified TF32 configuration')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = Path(str(args.output) + '.runtime.json')
    sources = [Path(__file__), Path(__file__).with_name('benchmark_qualified_operator_cases.py'), Path(__file__).with_name('operator_performance_operands.py'),
               Path(__file__).with_name('build_operator_performance_plans_v2.py'),
               Path(__file__).resolve().parents[1] / 'batch_invariant_ops/__init__.py',
               Path(__file__).resolve().parents[1] / 'batch_invariant_ops/batch_invariant_ops.py']
    source_hashes = {str(p.resolve()): sha(p) for p in sources}
    runtime = dict(status='running', model=plan['model'], gpu_uuid=args.gpu_uuid,
                   gpu_name=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                   cuda=torch.version.cuda, plan=str(args.plan.resolve()), plan_sha256=plan_hash,
                   source_sha256=source_hashes, timing_protocol=plan['timing_protocol'],
                   adapter_execution_configuration=plan['adapter_execution_configuration'],
                   tf32=False, cudnn_benchmark=False, deterministic_algorithms=False,
                   bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                   autocast=False, cuda_events_initialized_before_timing=True, source_pointer_alignment_reproduced=False,
                   source_storage_offset_reproduced=False,
                   measured_latency_scope='CUDA events around one complete operator call; wall samples additionally include host launch and synchronization overhead.',
                   regenerated_operand_values=True, captured_attention_masks=True,
                   primary_batch_invariance_qualification=False,
                   started_at_utc=datetime.now(timezone.utc).isoformat(), full_campaign_complete=False)
    archive = Path(str(args.output) + '.sources')
    archive.mkdir(exist_ok=False)
    for p in sources:
        (archive / source_hashes[str(p.resolve())]).write_bytes(p.read_bytes())
    runtime['source_archive'] = str(archive.resolve())
    runtime_path.write_text(json.dumps(runtime, indent=2) + '\n')
    count = 0
    try:
        with args.output.open('x') as stream, torch.inference_mode():
            for session in range(5):
                for index, case in enumerate(plan['cases']):
                    seed = record_seed(plan['model'], case['id'], session)
                    modes = case['modes']
                    allocation_error = None
                    try:
                        values, kwargs = operands(case, seed, 'cuda')
                        before = tensor_signatures([values, kwargs])
                    except torch.cuda.OutOfMemoryError as error:
                        allocation_error = str(error)
                    for variant_index, variant in enumerate(case['variants']):
                        variant_error = allocation_error
                        if allocation_error is None:
                            try:
                                derived, derived_kwargs = variant_operands(case, values, kwargs, variant, seed, 'cuda')
                                derived_before = tensor_signatures([derived, derived_kwargs])
                            except torch.cuda.OutOfMemoryError as error:
                                variant_error = str(error)
                        shift = (index + session + variant_index) % len(modes)
                        order = modes[shift:] + modes[:shift]
                        for position, mode in enumerate(order):
                            record = dict(model=plan['model'], case_id=case['id'], session=session,
                                          variant=variant, mode=mode, mode_position=position, seed=seed,
                                          probabilities_seed=(seed + 1) % (2 ** 63) if variant == 'attention_pv' else None,
                                          logical_batch_size=case['logical_batch_size'],
                                          plan_sha256=plan_hash, source_trace_sha256=case['source_trace_sha256'],
                                          source_operation_index=case['source_operation_index'],
                                          operation=case['entry']['aten_operation'], gpu_uuid=args.gpu_uuid,
                                          measured_operation=case['entry']['aten_operation'] if variant == 'whole' else 'aten::bmm',
                                          timing_protocol=plan['timing_protocol'])
                            if variant_error is not None:
                                record.update(status='cuda_capacity_failure', phase='operand_allocation', error=variant_error)
                            else:
                                record.update(operand_signatures=derived_before, source_operand_signatures=before)
                                try:
                                    with set_batch_invariant_mode(mode != 'native'):
                                        record.update(measure(operation(case, derived, derived_kwargs, mode, variant)))
                                    if (tensor_signatures([values, kwargs]) != before
                                            or tensor_signatures([derived, derived_kwargs]) != derived_before):
                                        raise ValueError('Operator mutated measured inputs')
                                except torch.cuda.OutOfMemoryError as error:
                                    record.update(status='cuda_capacity_failure', phase='operator_execution', error=str(error))
                            stream.write(json.dumps(record) + '\n')
                            stream.flush()
                            count += 1
                            print(f"session {session} {case['id']} {variant} {mode}: {record['status']}", flush=True)
                            if record['status'] == 'cuda_capacity_failure':
                                gc.collect()
                                torch.cuda.empty_cache()
                        if variant_error is None:
                            del derived, derived_kwargs
                    if allocation_error is None:
                        del values, kwargs
                    gc.collect()
                    torch.cuda.empty_cache()
        validate_plan(args.plan)
        if sha(args.plan) != plan_hash or any(sha(p) != h for p, h in source_hashes.items()):
            raise ValueError('Operator plan or source changed during timing')
        runtime.update(status='completed_raw_operator_timings_independent_acceptance_pending',
                       raw_records=count, raw_sha256=sha(args.output))
    except BaseException as error:
        runtime.update(status='failed', error=repr(error), raw_records=count)
        raise
    finally:
        runtime['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        runtime_path.write_text(json.dumps(runtime, indent=2) + '\n')


if __name__ == '__main__':
    main()

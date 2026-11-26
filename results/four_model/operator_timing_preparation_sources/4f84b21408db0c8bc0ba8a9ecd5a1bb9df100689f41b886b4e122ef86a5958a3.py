"""Reconstruct qualified trace layouts and explicit per-matrix references.

Trace descriptors retain logical dtype/shape/stride, not original pointer
alignment or operand values. Numerical operands are regenerated from a recorded
seed; attention masks are restored from lossless source captures.
"""
import itertools
import math

import torch
import torch.nn.functional as F

from scripts.build_operator_performance_plans import capture_bytes


FUNCTIONS = {'aten::linear': F.linear, 'aten::matmul': torch.matmul,
             'aten::bmm': torch.bmm, 'aten::mm': torch.mm, 'aten::addmm': torch.addmm,
             'aten::conv2d': F.conv2d, 'aten::mean': torch.mean,
             'aten::softmax': torch.softmax,
             'aten::scaled_dot_product_attention': F.scaled_dot_product_attention}


def dtype(name):
    value = getattr(torch, name.removeprefix('torch.'), None)
    if not isinstance(value, torch.dtype):
        raise ValueError('Unknown tensor dtype: ' + name)
    return value


def strided_tensor(descriptor, generator, device):
    shape, stride = tuple(descriptor['shape']), tuple(descriptor['stride'])
    if len(shape) != len(stride) or any(n <= 0 for n in shape) or any(s < 0 for s in stride):
        raise ValueError('Unsupported trace layout')
    kind = dtype(descriptor['dtype'])
    span = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    if kind == torch.bool:
        storage = torch.randint(0, 2, (span,), device=device, generator=generator, dtype=kind)
    elif kind.is_floating_point:
        storage = torch.randn(span, device=device, dtype=kind, generator=generator)
    else:
        raise ValueError('Unsupported operator input precision')
    return storage.as_strided(shape, stride)


def copy_values(destination, values):
    # Collapse expanded axes before writing; copying through overlapping views
    # is invalid. Check all expanded values really agree before collapsing.
    values = torch.broadcast_to(values, destination.shape)
    slices = tuple(slice(0, 1) if n > 1 and s == 0 else slice(None)
                   for n, s in zip(destination.shape, destination.stride()))
    compact = values[slices]
    if not torch.equal(values, compact.expand(destination.shape)):
        raise ValueError('Captured values contradict a broadcast stride')
    destination[slices].copy_(compact)


def materialize(value, generator, device):
    if isinstance(value, dict) and {'shape', 'stride', 'dtype'} <= value.keys():
        return strided_tensor(value, generator, device)
    if isinstance(value, list):
        return [materialize(v, generator, device) for v in value]
    if isinstance(value, dict):
        return {k: materialize(v, generator, device) for k, v in value.items()}
    if isinstance(value, str) and value.startswith('torch.'):
        return dtype(value)
    return value


def operands(case, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    args = materialize(case['entry']['arguments'], generator, device)
    kwargs = materialize(case['entry']['keyword_arguments'], generator, device)
    binding = case['mask_binding']
    if binding is not None:
        capture = binding['capture']
        payload = capture_bytes(capture)
        values = torch.frombuffer(bytearray(payload), dtype=dtype(capture['dtype'])).reshape(capture['values_shape'])
        if binding['selector'] is not None:
            index = binding['selector']
            values = values[index:index + 1]
        copy_values(args[3], values.to(device))
    return args, kwargs


def matrixwise(left, right):
    """Use actual per-matrix views; do not flatten away source strides."""
    leading = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    x = left.expand(*leading, *left.shape[-2:])
    y = right.expand(*leading, *right.shape[-2:])
    indices = itertools.product(*(range(n) for n in leading))
    outputs = [torch.mm(x[index], y[index]) for index in indices]
    return torch.stack(outputs).reshape(*leading, left.shape[-2], right.shape[-1])


def per_matrix_attention(query, key, value, attn_mask=None, dropout_p=0.,
                         is_causal=False, *, scale=None, enable_gqa=False):
    if dropout_p != 0:
        raise ValueError('Inference reference requires zero dropout')
    if enable_gqa:
        if query.shape[-3] % key.shape[-3] or key.shape[-3] != value.shape[-3]:
            raise ValueError('Invalid grouped-query head counts')
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)
    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        raise ValueError('Mismatched attention leading dimensions')
    scores = matrixwise(query, key.transpose(-1, -2))
    scores = scores * (scale if scale is not None else 1 / math.sqrt(query.shape[-1]))
    if is_causal:
        if attn_mask is not None:
            raise ValueError('Causal and explicit masks cannot both be set')
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, float('-inf'))
    elif attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float('-inf')) if attn_mask.dtype == torch.bool else scores + attn_mask
    probabilities = torch.softmax(scores, dim=-1, dtype=query.dtype)
    return matrixwise(probabilities, value)


def operation(case, args, kwargs, mode):
    """Caller controls invariant dispatch outside the measured call region."""
    if mode not in case['modes']:
        raise ValueError('Unplanned operator mode')
    if mode == 'explicit_invariant_per_matrix':
        name = case['entry']['aten_operation']
        if name in ('aten::bmm', 'aten::matmul'):
            return lambda: matrixwise(*args, **kwargs)
        if name == 'aten::scaled_dot_product_attention':
            return lambda: per_matrix_attention(*args, **kwargs)
        raise ValueError('No per-matrix reference for this operation')
    function = FUNCTIONS[case['entry']['aten_operation']]
    return lambda: function(*args, **kwargs)


def tensor_signatures(value):
    if isinstance(value, torch.Tensor):
        import hashlib
        data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                    logical_bytes_sha256=hashlib.sha256(data).hexdigest())
    if isinstance(value, (list, tuple)):
        return [tensor_signatures(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_signatures(v) for k, v in value.items()}
    return str(value) if isinstance(value, torch.dtype) else value

"""Capture an operand's physical span and restore its original tensor layout.

CPU controls can disable pointer-alignment checking. GPU attribution must keep
it enabled; inability to reproduce alignment is a replay failure, not evidence
of a model batching difference.
"""
import hashlib
import math

import torch

DTYPES = {str(dtype): dtype for dtype in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool,
)}


def byte_hash(tensor):
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def capture(tensor):
    check(tensor.layout == torch.strided and str(tensor.dtype) in DTYPES, 'Unsupported operand layout/dtype')
    shape, strides, offset = list(tensor.shape), list(tensor.stride()), tensor.storage_offset()
    check(all(stride >= 0 for stride in strides), 'Negative strides are unsupported')
    span = 0 if tensor.numel() == 0 else 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    check((offset + span) * tensor.element_size() <= tensor.untyped_storage().nbytes(), 'Operand escapes storage')
    payload = (tensor.as_strided((span,), (1,), storage_offset=offset).detach().cpu().clone()
               if span else torch.empty(0, dtype=tensor.dtype))
    return dict(shape=shape, strides=strides, storage_offset=offset, physical_span_elements=span,
                dtype=str(tensor.dtype), source_device=str(tensor.device),
                data_pointer_mod256=tensor.data_ptr() % 256 if tensor.numel() else None,
                span_sha256=byte_hash(payload), logical_sha256=byte_hash(tensor), payload=payload)


def restore(snapshot, device, *, require_pointer_alignment=True, maximum_storage_bytes=None):
    check(snapshot['dtype'] in DTYPES, 'Unsupported captured dtype')
    dtype = DTYPES[snapshot['dtype']]
    shape, strides, offset = snapshot['shape'], snapshot['strides'], snapshot['storage_offset']
    check(len(shape) == len(strides) and all(isinstance(v, int) and v >= 0 for v in shape + strides)
          and isinstance(offset, int) and offset >= 0, 'Invalid captured geometry')
    span = 0 if math.prod(shape) == 0 else 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    check(span == snapshot['physical_span_elements'], 'Captured span differs from geometry')
    payload = snapshot['payload']
    check(payload.device.type == 'cpu' and payload.dtype == dtype and payload.shape == (span,)
          and byte_hash(payload) == snapshot['span_sha256'], 'Captured physical bytes changed')
    itemsize = payload.element_size()
    if maximum_storage_bytes is not None:
        check((offset + span) * itemsize <= maximum_storage_bytes, 'Replay storage exceeds declared capacity')
    storage = torch.empty(offset + span, dtype=dtype, device=device)
    if span:
        storage[offset:offset + span].copy_(payload)
    result = storage.as_strided(shape, strides, storage_offset=offset)
    check(list(result.stride()) == strides and result.storage_offset() == offset, 'Replay tensor layout changed')
    check(byte_hash(result) == snapshot['logical_sha256'], 'Replay logical bytes changed')
    if require_pointer_alignment and result.numel():
        check(result.data_ptr() % 256 == snapshot['data_pointer_mod256'], 'Replay pointer alignment differs')
    return result

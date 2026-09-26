import contextlib
import math
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "matmul_persistent",
    "bmm_persistent",
    "conv1d_batch_invariant",
    "conv2d_batch_invariant",
    "conv3d_batch_invariant",
    "scaled_dot_product_attention_batch_invariant",
    "softmax",
]


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator, input_precision=INPUT_PRECISION)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(launch_metadata=_matmul_launch_metadata)
def bmm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BATCH_TILE_SLOTS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """One persistent matmul grid per batch element.

    Keeping each batch element on its own grid makes the reduction for an
    output element independent of both the batch size and its position in the
    batch.  The kernel intentionally has no cross-batch accumulation.
    """
    flat_pid = tl.program_id(axis=0)
    batch_idx = flat_pid // BATCH_TILE_SLOTS
    start_pid = flat_pid % BATCH_TILE_SLOTS
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    a_batch_ptr = a_ptr + batch_idx * stride_ab
    b_batch_ptr = b_ptr + batch_idx * stride_bb
    c_batch_ptr = c_ptr + batch_idx * stride_cb
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a = tl.load(
                a_batch_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(
                b_batch_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator, input_precision=INPUT_PRECISION)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_batch_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def get_compute_units():
    """
    Returns the number of streaming multiprocessors (SMs) or equivalent compute units
    for the available accelerator. Assigns the value to NUM_SMS.
    """
    NUM_SMS = None
    device_type = _current_accelerator_type()

    # Use match/case for device-specific logic (Python 3.10+)
    match device_type:
        case "cuda":
            device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            NUM_SMS = device_properties.multi_processor_count
        case "xpu":
            device_properties = torch.xpu.get_device_properties(0)
            NUM_SMS = device_properties.max_compute_units
        case _:
            print("No CUDA or XPU device available. Using CPU.")
            # For CPU, you might want to use the number of CPU cores
            NUM_SMS = torch.get_num_threads()

    return NUM_SMS


def _current_accelerator_type() -> str:
    """Return the active accelerator across supported PyTorch releases."""

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        current = accelerator.current_accelerator()
        if current is not None:
            return current.type
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert (
        bias is None or bias.dim() == 1
    ), "Currently assuming bias is 1D, let Horace know if you run into this"

    NUM_SMS = get_compute_units()
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    # print(a.device, b.device, c.device)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        INPUT_PRECISION="ieee",
        **configs[dtype],
    )
    return c


def bmm_persistent(a: torch.Tensor, b: torch.Tensor):
    """Batch-invariant ``torch.bmm`` for dense, equally-sized matrices.

    Unlike ``torch.matmul`` on rank-three tensors, this does not flatten the
    batch into a larger GEMM.  Each batch element has an independent persistent
    grid, so adding an unrelated attention request cannot alter an existing
    request's reduction schedule.
    """
    assert a.ndim == b.ndim == 3, "bmm expects two rank-3 tensors"
    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert a.dtype in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }, f"unsupported dtype: {a.dtype}"

    batch_size, m, k = a.shape
    n = b.shape[2]
    c = torch.empty((batch_size, m, n), device=a.device, dtype=a.dtype)
    if batch_size == 0 or m == 0 or n == 0:
        return c

    num_sms = get_compute_units()
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    def grid(meta):
        tiles = triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"])
        return (min(num_sms, tiles) * batch_size,)

    batch_tile_slots = min(
        num_sms,
        triton.cdiv(m, configs[a.dtype]["BLOCK_SIZE_M"])
        * triton.cdiv(n, configs[a.dtype]["BLOCK_SIZE_N"]),
    )

    bmm_kernel_persistent[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        NUM_SMS=num_sms,
        BATCH_TILE_SLOTS=batch_tile_slots,
        INPUT_PRECISION="ieee",
        **configs[a.dtype],
    )
    return c


@triton.jit
def depthwise_conv1d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    total_outputs,
    input_length,
    output_length,
    channels,
    input_stride_b,
    input_stride_c,
    input_stride_l,
    weight_stride_c,
    weight_stride_k,
    output_stride_b,
    output_stride_c,
    output_stride_l,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_index = offsets % output_length
    channel_index = (offsets // output_length) % channels
    batch_index = offsets // (output_length * channels)
    valid_output = offsets < total_outputs

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kernel_index in range(KERNEL_SIZE):
        input_index = output_index * STRIDE - PADDING + kernel_index * DILATION
        valid_input = valid_output & (input_index >= 0) & (input_index < input_length)
        value = tl.load(
            input_ptr
            + batch_index * input_stride_b
            + channel_index * input_stride_c
            + input_index * input_stride_l,
            mask=valid_input,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + channel_index * weight_stride_c + kernel_index * weight_stride_k,
            mask=valid_output,
            other=0.0,
        )
        accumulator += value.to(tl.float32) * weight.to(tl.float32)
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + channel_index, mask=valid_output, other=0.0).to(
            tl.float32
        )
    tl.store(
        output_ptr
        + batch_index * output_stride_b
        + channel_index * output_stride_c
        + output_index * output_stride_l,
        accumulator,
        mask=valid_output,
    )


def conv1d_batch_invariant(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    """Batch-invariant regular 1-D convolution.

    Depthwise convolution uses a dedicated fixed-order Triton kernel. Other
    group layouts use a per-sample grouped BMM decomposition.
    """
    assert input.ndim == 3 and weight.ndim == 3, "only 1-D convolution is supported"
    assert not transposed, "transposed convolution is not supported"
    assert all(
        value == 0 for value in output_padding
    ), f"output_padding requires transposed convolution, got {output_padding!r}"
    assert input.dtype == weight.dtype and (
        bias is None or bias.dtype == input.dtype
    ), "Incompatible dtypes"
    assert input.shape[1] == weight.shape[1] * groups, "Incompatible channel dimensions"
    assert weight.shape[0] % groups == 0, "output channels must be divisible by groups"

    def single(values, name):
        values = tuple(values)
        assert len(values) == 1, f"{name} must have one value, got {values!r}"
        return values[0]

    stride = single(stride, "stride")
    padding = single(padding, "padding")
    dilation = single(dilation, "dilation")
    kernel_size = weight.shape[-1]
    output_length = (input.shape[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if input.shape[0] == 0:
        return torch.empty(
            (0, weight.shape[0], output_length), device=input.device, dtype=input.dtype
        )

    if groups == input.shape[1] == weight.shape[0] and weight.shape[1] == 1:
        output = torch.empty(
            (input.shape[0], weight.shape[0], output_length),
            device=input.device,
            dtype=input.dtype,
        )
        block_size = 256
        total_outputs = output.numel()
        depthwise_conv1d_kernel[(triton.cdiv(total_outputs, block_size),)](
            input,
            weight,
            bias,
            output,
            total_outputs,
            input.shape[-1],
            output_length,
            input.shape[1],
            input.stride(0),
            input.stride(1),
            input.stride(2),
            weight.stride(0),
            weight.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            KERNEL_SIZE=kernel_size,
            STRIDE=stride,
            PADDING=padding,
            DILATION=dilation,
            HAS_BIAS=bias is not None,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        return output

    channels_per_group = input.shape[1] // groups
    outputs_per_group = weight.shape[0] // groups
    grouped_weight = weight.reshape(groups, outputs_per_group, -1).transpose(1, 2)
    padded = F.pad(input, (padding, padding))
    batch_stride, channel_stride, length_stride = padded.stride()
    patches = padded.as_strided(
        size=(input.shape[0], padded.shape[1], output_length, kernel_size),
        stride=(
            batch_stride,
            channel_stride,
            length_stride * stride,
            length_stride * dilation,
        ),
    )
    grouped_patches = (
        patches.reshape(input.shape[0], groups, channels_per_group, output_length, kernel_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(input.shape[0] * groups, output_length, channels_per_group * kernel_size)
    )
    grouped_weight = (
        grouped_weight.unsqueeze(0)
        .expand(input.shape[0], -1, -1, -1)
        .reshape(input.shape[0] * groups, channels_per_group * kernel_size, outputs_per_group)
    )
    output = bmm_persistent(grouped_patches, grouped_weight).reshape(
        input.shape[0], groups, output_length, outputs_per_group
    )
    if bias is not None:
        output = output + bias.reshape(1, groups, 1, outputs_per_group)
    return output.permute(0, 1, 3, 2).reshape(input.shape[0], weight.shape[0], output_length)


def conv2d_batch_invariant(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    """Batch-invariant implementation of the regular ``aten::convolution`` path.

    This is the same per-sample unfold-and-GEMM strategy used by the PiZero
    SigLIP patch projection.  It supports non-transposed 2-D convolution,
    including grouped and dilated convolution; the implementation is intended
    for CUDA inference, matching the scope of the other operators in this
    package.
    """
    assert input.ndim == 4 and weight.ndim == 4, "only 2-D convolution is supported"
    assert not transposed, "transposed convolution is not supported"
    assert all(
        value == 0 for value in output_padding
    ), f"output_padding requires transposed convolution, got {output_padding!r}"
    assert input.dtype == weight.dtype and (
        bias is None or bias.dtype == input.dtype
    ), "Incompatible dtypes"
    assert input.shape[1] == weight.shape[1] * groups, "Incompatible channel dimensions"
    assert weight.shape[0] % groups == 0, "output channels must be divisible by groups"

    def pair(values, name):
        values = tuple(values)
        if len(values) == 1:
            return values * 2
        assert len(values) == 2, f"{name} must have one or two values, got {values!r}"
        return values

    stride = pair(stride, "stride")
    padding = pair(padding, "padding")
    dilation = pair(dilation, "dilation")
    kernel_size = tuple(weight.shape[-2:])
    output_height = (
        input.shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    output_width = (
        input.shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    if input.shape[0] == 0:
        return torch.empty(
            (0, weight.shape[0], output_height, output_width),
            device=input.device,
            dtype=input.dtype,
        )

    channels_per_group = input.shape[1] // groups
    outputs_per_group = weight.shape[0] // groups
    flattened_weight = weight.reshape(weight.shape[0], -1)
    outputs = []
    for sample in input:
        patches = F.unfold(
            sample.unsqueeze(0),
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        ).squeeze(0)
        group_outputs = [
            matmul_persistent(
                patches[
                    group
                    * channels_per_group
                    * kernel_size[0]
                    * kernel_size[1] : (group + 1)
                    * channels_per_group
                    * kernel_size[0]
                    * kernel_size[1]
                ].transpose(0, 1),
                flattened_weight[
                    group * outputs_per_group : (group + 1) * outputs_per_group
                ].transpose(0, 1),
                bias=(
                    None
                    if bias is None
                    else bias[group * outputs_per_group : (group + 1) * outputs_per_group]
                ),
            ).transpose(0, 1)
            for group in range(groups)
        ]
        output = torch.cat(group_outputs, dim=0)
        outputs.append(output.reshape(1, weight.shape[0], output_height, output_width))
    return torch.cat(outputs, dim=0)


def conv3d_batch_invariant(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    """Batch-invariant regular 3-D convolution via per-sample im2col.

    Qwen3.5-VL uses a ``Conv3d`` patch embedding whose leading dimension is
    the number of image patches.  Running every leading-dimension element
    through the same fixed GEMM prevents that patch count (and therefore the
    request batch size) from changing the reduction schedule.
    """
    assert input.ndim == 5 and weight.ndim == 5, "only 3-D convolution is supported"
    assert not transposed, "transposed convolution is not supported"
    assert all(
        value == 0 for value in output_padding
    ), f"output_padding requires transposed convolution, got {output_padding!r}"
    assert input.dtype == weight.dtype and (
        bias is None or bias.dtype == input.dtype
    ), "Incompatible dtypes"
    assert input.shape[1] == weight.shape[1] * groups, "Incompatible channel dimensions"
    assert weight.shape[0] % groups == 0, "output channels must be divisible by groups"

    def triple(values, name):
        values = tuple(values)
        if len(values) == 1:
            return values * 3
        assert len(values) == 3, f"{name} must have one or three values, got {values!r}"
        return values

    stride = triple(stride, "stride")
    padding = triple(padding, "padding")
    dilation = triple(dilation, "dilation")
    kernel_size = tuple(weight.shape[-3:])
    output_shape = tuple(
        (input.shape[axis + 2] + 2 * padding[axis] - dilation[axis] * (kernel_size[axis] - 1) - 1)
        // stride[axis]
        + 1
        for axis in range(3)
    )
    if input.shape[0] == 0:
        return torch.empty(
            (0, weight.shape[0], *output_shape), device=input.device, dtype=input.dtype
        )

    channels_per_group = input.shape[1] // groups
    outputs_per_group = weight.shape[0] // groups
    kernel_volume = math.prod(kernel_size)
    flattened_weight = weight.reshape(weight.shape[0], -1)
    padded = F.pad(
        input,
        (
            padding[2],
            padding[2],
            padding[1],
            padding[1],
            padding[0],
            padding[0],
        ),
    )
    batch_stride, channel_stride, depth_stride, height_stride, width_stride = padded.stride()
    patches = padded.as_strided(
        size=(input.shape[0], padded.shape[1], *output_shape, *kernel_size),
        stride=(
            batch_stride,
            channel_stride,
            depth_stride * stride[0],
            height_stride * stride[1],
            width_stride * stride[2],
            depth_stride * dilation[0],
            height_stride * dilation[1],
            width_stride * dilation[2],
        ),
    )
    output_volume = math.prod(output_shape)
    grouped_patches = (
        patches.permute(0, 2, 3, 4, 1, 5, 6, 7)
        .reshape(input.shape[0], output_volume, groups, channels_per_group * kernel_volume)
        .permute(0, 2, 1, 3)
        .reshape(input.shape[0] * groups, output_volume, channels_per_group * kernel_volume)
    )
    grouped_weight = (
        flattened_weight.reshape(groups, outputs_per_group, -1)
        .transpose(1, 2)
        .unsqueeze(0)
        .expand(input.shape[0], -1, -1, -1)
        .reshape(
            input.shape[0] * groups,
            channels_per_group * kernel_volume,
            outputs_per_group,
        )
    )
    output = bmm_persistent(grouped_patches, grouped_weight).reshape(
        input.shape[0], groups, output_volume, outputs_per_group
    )
    if bias is not None:
        output = output + bias.reshape(1, groups, 1, outputs_per_group)
    return output.permute(0, 1, 3, 2).reshape(input.shape[0], weight.shape[0], *output_shape)


def convolution_batch_invariant(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    """Dispatch regular 1-D, 2-D and 3-D ``aten::convolution`` calls."""
    if input.ndim == 3 and weight.ndim == 3:
        return conv1d_batch_invariant(
            input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
        )
    if input.ndim == 4 and weight.ndim == 4:
        return conv2d_batch_invariant(
            input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
        )
    if input.ndim == 5 and weight.ndim == 5:
        return conv3d_batch_invariant(
            input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
        )
    raise AssertionError(
        f"only 1-D, 2-D and 3-D convolution are supported, got input rank {input.ndim} "
        f"and weight rank {weight.ndim}"
    )


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)
    >> Stashed changes
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute one softmax row per Triton program.

    The program and reduction tree for an existing row do not depend on the
    number of other rows.  This is the property needed by attention when the
    leading dimensions contain the request batch.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    input_row = input_ptr + row_idx * input_row_stride
    output_row = output_ptr + row_idx * output_row_stride

    max_value = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        columns = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = columns < n_cols
        values = tl.load(input_row + columns, mask=mask, other=-float("inf"))
        max_value = tl.maximum(max_value, tl.max(values))

    denominator = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        columns = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = columns < n_cols
        values = tl.load(input_row + columns, mask=mask, other=-float("inf"))
        shifted = tl.where(max_value == -float("inf"), -float("inf"), values - max_value)
        denominator += tl.sum(tl.where(mask, tl.exp(shifted), 0.0))

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        columns = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = columns < n_cols
        values = tl.load(input_row + columns, mask=mask, other=-float("inf"))
        shifted = tl.where(max_value == -float("inf"), -float("inf"), values - max_value)
        result = tl.where(denominator == 0.0, 0.0, tl.exp(shifted) / denominator)
        tl.store(output_row + columns, result, mask=mask)


def softmax(
    input: torch.Tensor,
    dim: int = -1,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batch-invariant softmax along the last dimension."""
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports softmax along the last dimension")
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported softmax dtype: {input.dtype}")
    if dtype not in {None, torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported softmax output dtype: {dtype}")

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    output = torch.empty(input_2d.shape, dtype=dtype or input.dtype, device=input.device)
    if input_2d.numel() == 0:
        return output.reshape(original_shape)
    _softmax_kernel[(input_2d.shape[0],)](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        input_2d.shape[1],
        BLOCK_SIZE=1024,
    )
    return output.reshape(original_shape)


def scaled_dot_product_attention_batch_invariant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Eager SDPA with fixed per-head GPU reductions.

    Vendor fused-attention kernels may select a different decomposition when
    the request batch changes.  This implementation flattens the leading
    dimensions into independent heads, then uses the batch-invariant BMM and
    row-wise softmax kernels.  Inference requires ``dropout_p == 0``.
    """
    if dropout_p != 0.0:
        raise ValueError("batch-invariant SDPA only supports dropout_p=0")
    if query.ndim < 3 or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("query, key, and value must have matching rank >= 3")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")

    if enable_gqa:
        if query.shape[-3] % key.shape[-3] or key.shape[-3] != value.shape[-3]:
            raise ValueError("invalid grouped-query attention head counts")
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)

    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        raise ValueError("leading query, key, and value dimensions must match")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("incompatible attention dimensions")

    leading_shape = query.shape[:-2]
    query_length, key_length = query.shape[-2], key.shape[-2]
    head_dim, value_dim = query.shape[-1], value.shape[-1]
    flat_query = query.reshape(-1, query_length, head_dim)
    flat_key = key.reshape(-1, key_length, head_dim)
    flat_value = value.reshape(-1, key_length, value_dim)

    scores = bmm_persistent(flat_query, flat_key.transpose(1, 2))
    scores = scores * (scale if scale is not None else 1.0 / math.sqrt(head_dim))
    scores = scores.reshape(*leading_shape, query_length, key_length)

    if is_causal:
        if attn_mask is not None:
            raise ValueError("attn_mask and is_causal cannot both be set")
        causal_mask = torch.ones(
            (query_length, key_length), dtype=torch.bool, device=query.device
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    probabilities = softmax(scores, dim=-1, dtype=query.dtype)
    output = bmm_persistent(probabilities.reshape(-1, query_length, key_length), flat_value)
    return output.reshape(*leading_shape, query_length, value_dim)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def bmm_batch_invariant(a, b):
    return bmm_persistent(a, b)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        if len(dim) == 0:
            dim = list(range(input.ndim))
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return (
            torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32).to(dtype or input.dtype)
            / n_elems
        )


_batch_invariant_MODE = False
_batch_invariant_LIB = None


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    dispatch_key = _current_accelerator_type().upper()
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::convolution", convolution_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl(
        "aten::scaled_dot_product_attention",
        scaled_dot_product_attention_batch_invariant,
        dispatch_key,
    )
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, dispatch_key)


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    previously_enabled = is_batch_invariant_mode_enabled()
    if enabled == previously_enabled:
        yield
        return

    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    try:
        yield
    finally:
        if previously_enabled:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)

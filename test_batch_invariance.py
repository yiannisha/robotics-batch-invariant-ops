"""CUDA correctness and batch-invariance regression tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from batch_invariant_ops import (
    bmm_persistent,
    conv1d_batch_invariant,
    conv2d_batch_invariant,
    conv3d_batch_invariant,
    is_batch_invariant_mode_enabled,
    matmul_persistent,
    scaled_dot_product_attention_batch_invariant,
    set_batch_invariant_mode,
    varlen_scaled_dot_product_attention_batch_invariant,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
TOLERANCES = {
    torch.float32: {"rtol": 5e-5, "atol": 2e-5},
    torch.float16: {"rtol": 2e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 2e-2, "atol": 1e-2},
}


def _assert_first_sample_equal(alone: torch.Tensor, batched: torch.Tensor) -> None:
    assert torch.equal(alone, batched[:1]), (
        f"first sample changed: max diff "
        f"{(alone.float() - batched[:1].float()).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", (2, 3, 5, 17))
def test_mm_is_batch_invariant(dtype: torch.dtype, batch_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1000 + batch_size)
    inputs = torch.randn(batch_size, 257, device="cuda", dtype=dtype, generator=generator)
    weight = torch.randn(257, 129, device="cuda", dtype=dtype, generator=generator)

    alone = matmul_persistent(inputs[:1], weight)
    batched = matmul_persistent(inputs, weight)

    _assert_first_sample_equal(alone, batched)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mm_matches_torch(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(2000)
    left = torch.randn(37, 257, device="cuda", dtype=dtype, generator=generator)
    right = torch.randn(257, 91, device="cuda", dtype=dtype, generator=generator)

    torch.testing.assert_close(
        matmul_persistent(left, right), torch.mm(left, right), **TOLERANCES[dtype]
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", (2, 3, 5, 17))
def test_bmm_is_batch_invariant(dtype: torch.dtype, batch_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(3000 + batch_size)
    left = torch.randn(batch_size, 33, 65, device="cuda", dtype=dtype, generator=generator)
    right = torch.randn(batch_size, 65, 47, device="cuda", dtype=dtype, generator=generator)

    alone = bmm_persistent(left[:1], right[:1])
    batched = bmm_persistent(left, right)

    _assert_first_sample_equal(alone, batched)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_matches_torch_for_noncontiguous_attention_inputs(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(4000)
    query = torch.randn(4, 8, 33, 65, device="cuda", dtype=dtype, generator=generator)
    key = torch.randn(4, 8, 47, 65, device="cuda", dtype=dtype, generator=generator)
    query = query.reshape(-1, 33, 65)
    key = key.reshape(-1, 47, 65).transpose(1, 2)

    torch.testing.assert_close(
        bmm_persistent(query, key), torch.bmm(query, key), **TOLERANCES[dtype]
    )


def test_bmm_supports_more_than_cuda_grid_y_limit() -> None:
    generator = torch.Generator(device="cuda").manual_seed(4500)
    left = torch.randn(70_000, 1, 4, device="cuda", dtype=torch.bfloat16, generator=generator)
    right = torch.randn(70_000, 4, 1, device="cuda", dtype=torch.bfloat16, generator=generator)

    alone = bmm_persistent(left[:1], right[:1])
    batched = bmm_persistent(left, right)

    _assert_first_sample_equal(alone, batched)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("composition", ("duplicate", "unrelated"))
@pytest.mark.parametrize("batch_size", (2, 5, 17, 32))
def test_siglip_patch_conv_is_batch_invariant(
    dtype: torch.dtype, composition: str, batch_size: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(5000 + batch_size)
    target = torch.randn(1, 3, 28, 28, device="cuda", dtype=dtype, generator=generator)
    if composition == "duplicate":
        inputs = target.expand(batch_size, -1, -1, -1).contiguous()
    else:
        companions = torch.randn(
            batch_size - 1, 3, 28, 28, device="cuda", dtype=dtype, generator=generator
        )
        inputs = torch.cat((target, companions))
    weight = torch.randn(32, 3, 14, 14, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(32, device="cuda", dtype=dtype, generator=generator)

    args = (weight, bias, (14, 14), (0, 0), (1, 1), False, (0, 0), 1)
    alone = conv2d_batch_invariant(target, *args)
    batched = conv2d_batch_invariant(inputs, *args)

    _assert_first_sample_equal(alone, batched)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", (2, 5, 17))
def test_depthwise_conv1d_is_batch_invariant(dtype: torch.dtype, batch_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(5800 + batch_size)
    target = torch.randn(1, 32, 73, device="cuda", dtype=dtype, generator=generator)
    companions = torch.randn(
        batch_size - 1, 32, 73, device="cuda", dtype=dtype, generator=generator
    )
    inputs = torch.cat((target, companions))
    weight = torch.randn(32, 1, 4, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(32, device="cuda", dtype=dtype, generator=generator)
    args = (weight, bias, (1,), (3,), (1,), False, (0,), 32)

    _assert_first_sample_equal(
        conv1d_batch_invariant(target, *args), conv1d_batch_invariant(inputs, *args)
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("shape", "weight_shape", "stride", "padding", "dilation", "groups"),
    (
        ((3, 4, 19), (7, 4, 3), (1,), (0,), (1,), 1),
        ((2, 8, 23), (12, 2, 4), (2,), (3,), (2,), 4),
        ((2, 16, 31), (16, 1, 4), (1,), (3,), (1,), 16),
    ),
)
def test_conv1d_matches_torch(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    groups: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(5900)
    inputs = torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)
    weight = torch.randn(*weight_shape, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(weight_shape[0], device="cuda", dtype=dtype, generator=generator)

    actual = conv1d_batch_invariant(
        inputs, weight, bias, stride, padding, dilation, False, (0,), groups
    )
    expected = F.conv1d(inputs, weight, bias, stride, padding, dilation, groups)

    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("shape", "weight_shape", "stride", "padding", "dilation", "groups"),
    (
        ((3, 4, 13, 11), (7, 4, 3, 3), (1, 1), (0, 0), (1, 1), 1),
        ((2, 4, 13, 11), (6, 2, 3, 2), (2, 1), (1, 0), (2, 1), 2),
        ((2, 4, 9, 9), (4, 1, 3, 3), (1, 1), (1, 1), (1, 1), 4),
    ),
)
def test_conv2d_matches_torch(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(6000)
    inputs = torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)
    weight = torch.randn(*weight_shape, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(weight_shape[0], device="cuda", dtype=dtype, generator=generator)

    actual = conv2d_batch_invariant(
        inputs, weight, bias, stride, padding, dilation, False, (0, 0), groups
    )
    expected = F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)

    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("shape", "weight_shape", "stride", "padding", "dilation", "groups"),
    (
        ((3, 3, 2, 16, 16), (12, 3, 2, 16, 16), (2, 16, 16), (0, 0, 0), (1, 1, 1), 1),
        ((2, 4, 7, 9, 8), (6, 2, 3, 2, 2), (2, 1, 2), (1, 0, 1), (2, 1, 1), 2),
    ),
)
def test_conv3d_matches_torch(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(6500)
    inputs = torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)
    weight = torch.randn(*weight_shape, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(weight_shape[0], device="cuda", dtype=dtype, generator=generator)

    actual = conv3d_batch_invariant(
        inputs, weight, bias, stride, padding, dilation, False, (0, 0, 0), groups
    )
    expected = F.conv3d(inputs, weight, bias, stride, padding, dilation, groups)

    # cuDNN and the fixed-order GEMM use different FP32 reduction trees for
    # the large patch dimension (3 * 2 * 16 * 16).  Both agree closely with
    # the explicit im2col reference; allow the corresponding accumulated
    # roundoff here while retaining the usual half-precision tolerances.
    tolerance = {"rtol": 5e-4, "atol": 4e-2} if dtype == torch.float32 else TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", (2, 5, 17))
def test_qwen_patch_conv3d_is_batch_invariant(dtype: torch.dtype, batch_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(6700 + batch_size)
    target = torch.randn(1, 3, 2, 16, 16, device="cuda", dtype=dtype, generator=generator)
    companions = torch.randn(
        batch_size - 1, 3, 2, 16, 16, device="cuda", dtype=dtype, generator=generator
    )
    inputs = torch.cat((target, companions))
    weight = torch.randn(12, 3, 2, 16, 16, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(12, device="cuda", dtype=dtype, generator=generator)
    args = (weight, bias, (2, 16, 16), (0, 0, 0), (1, 1, 1), False, (0, 0, 0), 1)

    _assert_first_sample_equal(
        conv3d_batch_invariant(target, *args), conv3d_batch_invariant(inputs, *args)
    )


def test_mode_overrides_matmul_and_conv1d_conv2d_conv3d() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7000)
    query = torch.randn(5, 19, 23, device="cuda", generator=generator)
    key = torch.randn(5, 23, 17, device="cuda", generator=generator)
    image = torch.randn(5, 3, 28, 28, device="cuda", generator=generator)
    weight = torch.randn(8, 3, 14, 14, device="cuda", generator=generator)
    signal = torch.randn(5, 8, 19, device="cuda", generator=generator)
    signal_weight = torch.randn(8, 1, 4, device="cuda", generator=generator)
    video = torch.randn(5, 3, 2, 16, 16, device="cuda", generator=generator)
    video_weight = torch.randn(8, 3, 2, 16, 16, device="cuda", generator=generator)

    with set_batch_invariant_mode():
        _assert_first_sample_equal(query[:1] @ key[:1], query @ key)
        _assert_first_sample_equal(
            F.conv1d(signal[:1], signal_weight, padding=3, groups=8),
            F.conv1d(signal, signal_weight, padding=3, groups=8),
        )
        _assert_first_sample_equal(
            F.conv2d(image[:1], weight, stride=14), F.conv2d(image, weight, stride=14)
        )
        _assert_first_sample_equal(
            F.conv3d(video[:1], video_weight, stride=(2, 16, 16)),
            F.conv3d(video, video_weight, stride=(2, 16, 16)),
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", (2, 3, 5, 17))
def test_sdpa_is_batch_invariant(dtype: torch.dtype, batch_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(7100 + batch_size)
    target = torch.randn(1, 16, 256, 72, device="cuda", dtype=dtype, generator=generator)
    companions = torch.randn(
        batch_size - 1, 16, 256, 72, device="cuda", dtype=dtype, generator=generator
    )
    query = torch.cat((target, companions))
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    alone = scaled_dot_product_attention_batch_invariant(query[:1], key[:1], value[:1])
    batched = scaled_dot_product_attention_batch_invariant(query, key, value)
    _assert_first_sample_equal(alone, batched)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", (False, True))
def test_sdpa_matches_torch(dtype: torch.dtype, causal: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(7200)
    query = torch.randn(2, 8, 33, 65, device="cuda", dtype=dtype, generator=generator)
    key = torch.randn(2, 8, 47, 65, device="cuda", dtype=dtype, generator=generator)
    value = torch.randn(2, 8, 47, 39, device="cuda", dtype=dtype, generator=generator)

    expected = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
    actual = scaled_dot_product_attention_batch_invariant(query, key, value, is_causal=causal)
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])


@pytest.mark.parametrize("mask_kind", ("boolean", "additive"))
def test_sdpa_matches_torch_with_mask(mask_kind: str) -> None:
    generator = torch.Generator(device="cuda").manual_seed(7250)
    query = torch.randn(2, 4, 17, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
    key = torch.randn(2, 4, 23, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
    value = torch.randn(2, 4, 23, 24, device="cuda", dtype=torch.bfloat16, generator=generator)
    boolean_mask = torch.rand(2, 1, 17, 23, device="cuda", generator=generator) > 0.2
    # Exercise the fully-masked-row behavior as well as ordinary masking.
    boolean_mask[:, :, -1] = False
    mask = (
        boolean_mask
        if mask_kind == "boolean"
        else torch.where(
            boolean_mask,
            torch.tensor(0.0, device="cuda", dtype=torch.bfloat16),
            torch.tensor(float("-inf"), device="cuda", dtype=torch.bfloat16),
        )
    )

    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    actual = scaled_dot_product_attention_batch_invariant(query, key, value, attn_mask=mask)
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.bfloat16])


def test_sdpa_matches_torch_with_grouped_query_attention() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7275)
    query = torch.randn(2, 8, 17, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
    key = torch.randn(2, 2, 23, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
    value = torch.randn(2, 2, 23, 24, device="cuda", dtype=torch.bfloat16, generator=generator)

    expected = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    actual = scaled_dot_product_attention_batch_invariant(query, key, value, enable_gqa=True)
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.bfloat16])


def test_mode_overrides_scaled_dot_product_attention() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7300)
    query = torch.randn(3, 16, 256, 72, device="cuda", dtype=torch.bfloat16, generator=generator)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with set_batch_invariant_mode():
        alone = F.scaled_dot_product_attention(query[:1], key[:1], value[:1])
        batched = F.scaled_dot_product_attention(query, key, value)
    _assert_first_sample_equal(alone, batched)


def _packed_offsets(lengths: tuple[int, ...]) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("causal_type", (None, "TopLeft", "BottomRight"))
def test_varlen_sdpa_matches_independent_torch_calls(causal_type: str | None) -> None:
    generator = torch.Generator(device="cuda").manual_seed(7350)
    q_lengths = (5, 3, 7)
    kv_lengths = (7, 4, 5)
    query = torch.randn(
        1,
        sum(q_lengths),
        4,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        1,
        sum(kv_lengths),
        2,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        1,
        sum(kv_lengths),
        2,
        24,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_offsets = _packed_offsets(q_lengths)
    kv_offsets = _packed_offsets(kv_lengths)
    actual = varlen_scaled_dot_product_attention_batch_invariant(
        query,
        key,
        value,
        q_offsets,
        kv_offsets,
        is_causal=causal_type is not None,
        causal_type=causal_type,
    )

    expected_segments = []
    for index, (q_length, kv_length) in enumerate(zip(q_lengths, kv_lengths)):
        q_start, q_end = q_offsets[index : index + 2].tolist()
        kv_start, kv_end = kv_offsets[index : index + 2].tolist()
        q_item = query[:, q_start:q_end].transpose(1, 2)
        k_item = key[:, kv_start:kv_end].transpose(1, 2)
        v_item = value[:, kv_start:kv_end].transpose(1, 2)
        mask = None
        if causal_type is not None:
            q_indexes = torch.arange(q_length, device="cuda")[:, None]
            kv_indexes = torch.arange(kv_length, device="cuda")[None, :]
            shift = kv_length - q_length if causal_type == "BottomRight" else 0
            mask = kv_indexes <= q_indexes + shift
        expected_segments.append(
            F.scaled_dot_product_attention(
                q_item, k_item, v_item, attn_mask=mask, enable_gqa=True
            ).transpose(1, 2)
        )
    expected = torch.cat(expected_segments, dim=1)
    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.bfloat16])


def test_varlen_sdpa_is_batch_invariant_for_duplicate_and_unrelated_packs() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7375)
    q_lengths = (33, 17, 33)
    kv_lengths = (47, 23, 47)
    query = torch.randn(
        1,
        sum(q_lengths),
        8,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        1,
        sum(kv_lengths),
        8,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn_like(key)

    alone = varlen_scaled_dot_product_attention_batch_invariant(
        query[:, : q_lengths[0]],
        key[:, : kv_lengths[0]],
        value[:, : kv_lengths[0]],
        _packed_offsets(q_lengths[:1]),
        _packed_offsets(kv_lengths[:1]),
        is_causal=True,
        causal_type="BottomRight",
    )
    packed = varlen_scaled_dot_product_attention_batch_invariant(
        query,
        key,
        value,
        _packed_offsets(q_lengths),
        _packed_offsets(kv_lengths),
        is_causal=True,
        causal_type="BottomRight",
    )
    assert torch.equal(alone, packed[:, : q_lengths[0]])


def test_varlen_sdpa_leaves_uncovered_padding_rows_zero() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7390)
    query = torch.randn(
        1,
        11,
        2,
        16,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    output = varlen_scaled_dot_product_attention_batch_invariant(
        query,
        key,
        value,
        torch.tensor([0, 5, 9], device="cuda", dtype=torch.int32),
        torch.tensor([0, 5, 9], device="cuda", dtype=torch.int32),
    )
    assert torch.count_nonzero(output[:, :9]) > 0
    assert torch.count_nonzero(output[:, 9:]) == 0


def test_varlen_sdpa_empty_kv_range_is_zero() -> None:
    query = torch.randn(1, 2, 2, 16, device="cuda", dtype=torch.float16)
    key = torch.empty(1, 0, 2, 16, device="cuda", dtype=torch.float16)
    value = torch.empty_like(key)
    output = varlen_scaled_dot_product_attention_batch_invariant(
        query,
        key,
        value,
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 0], device="cuda", dtype=torch.int32),
    )
    assert torch.count_nonzero(output) == 0


def test_mode_context_is_exception_safe_and_reentrant() -> None:
    assert not is_batch_invariant_mode_enabled()
    with pytest.raises(RuntimeError, match="sentinel"):
        with set_batch_invariant_mode():
            assert is_batch_invariant_mode_enabled()
            with set_batch_invariant_mode():
                assert is_batch_invariant_mode_enabled()
            assert is_batch_invariant_mode_enabled()
            raise RuntimeError("sentinel")
    assert not is_batch_invariant_mode_enabled()

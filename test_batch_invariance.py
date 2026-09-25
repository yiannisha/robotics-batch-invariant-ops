"""CUDA correctness and batch-invariance regression tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from batch_invariant_ops import (
    bmm_persistent,
    conv2d_batch_invariant,
    is_batch_invariant_mode_enabled,
    matmul_persistent,
    set_batch_invariant_mode,
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


def test_mode_overrides_rank_three_matmul_and_conv2d() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7000)
    query = torch.randn(5, 19, 23, device="cuda", generator=generator)
    key = torch.randn(5, 23, 17, device="cuda", generator=generator)
    image = torch.randn(5, 3, 28, 28, device="cuda", generator=generator)
    weight = torch.randn(8, 3, 14, 14, device="cuda", generator=generator)

    with set_batch_invariant_mode():
        _assert_first_sample_equal(query[:1] @ key[:1], query @ key)
        _assert_first_sample_equal(
            F.conv2d(image[:1], weight, stride=14), F.conv2d(image, weight, stride=14)
        )


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

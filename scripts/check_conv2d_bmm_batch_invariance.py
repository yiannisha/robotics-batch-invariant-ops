"""Demonstrate the Conv2d and attention-BMM batch-invariance failure modes.

Run on CUDA from the repository root:

    python scripts/check_conv2d_bmm_batch_invariance.py

The standard results are hardware- and PyTorch-version-dependent: a zero
difference means that the selected cuDNN/cuBLAS algorithm happened to use the
same reduction schedule in that run.  Batch-invariant mode is required to make
the comparison bitwise equal regardless of that selection.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from batch_invariant_ops import set_batch_invariant_mode


def max_difference(alone: torch.Tensor, batched: torch.Tensor) -> float:
    return (alone - batched[:1]).abs().max().item()


def conv2d_case(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # This is the SigLIP patch projection shape that motivated PiZero's
    # per-sample unfold-and-GEMM replacement for torch.conv2d.
    inputs = torch.randn(batch_size, 3, 224, 224, device="cuda", dtype=torch.float32)
    weight = torch.randn(1152, 3, 14, 14, device="cuda", dtype=torch.float32)
    bias = torch.randn(1152, device="cuda", dtype=torch.float32)
    return F.conv2d(inputs[:1], weight, bias, stride=14), F.conv2d(inputs, weight, bias, stride=14)


def attention_matmul_case(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Multihead attention commonly flattens (batch, heads) into the leading
    # dimension.  These rank-3 inputs make torch.matmul dispatch aten::bmm,
    # not aten::mm; consequently an mm-only replacement cannot cover it.
    heads, queries, tokens, head_dim = 8, 4, 281, 256
    query = torch.randn(
        batch_size * heads, queries, tokens, device="cuda", dtype=torch.float32
    )
    key = torch.randn(batch_size * heads, tokens, head_dim, device="cuda", dtype=torch.float32)
    alone = torch.matmul(query[:heads], key[:heads]).reshape(1, heads, queries, head_dim)
    batched = torch.matmul(query, key).reshape(batch_size, heads, queries, head_dim)
    return alone, batched


def profile_attention_dispatch() -> None:
    query = torch.randn(8, 128, 64, device="cuda")
    key = torch.randn(8, 64, 128, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        with record_function("attention:matmul"):
            torch.matmul(query, key)
    operators = {event.key for event in profiler.key_averages()}
    assert "aten::bmm" in operators, f"expected aten::bmm, saw {sorted(operators)}"
    print("attention rank-3 matmul dispatch: aten::bmm (not aten::mm)")


def run_case(name: str, case, batch_size: int, enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with set_batch_invariant_mode(enabled):
        alone, batched = case(batch_size)
    difference = max_difference(alone, batched)
    label = "batch-invariant" if enabled else "standard torch"
    print(f"{name:16} {label:18} max |single - batched[:1]| = {difference:.8g}")
    if enabled:
        assert difference == 0.0, f"{name} is not batch invariant"
    return alone, batched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--require-standard-difference",
        action="store_true",
        help="fail when this CUDA/PyTorch combination does not expose a standard-kernel difference",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This demonstration requires CUDA and Triton.")

    profile_attention_dispatch()
    standard_results = []
    for seed, (name, case) in enumerate(
        (("conv2d (14x14)", conv2d_case), ("attention bmm", attention_matmul_case))
    ):
        torch.manual_seed(seed)
        standard = run_case(name, case, args.batch_size, enabled=False)
        torch.manual_seed(seed)
        invariant = run_case(name, case, args.batch_size, enabled=True)
        # Compare the same batch-one computation.  The standard batched result
        # is intentionally the potentially divergent value in this experiment.
        torch.testing.assert_close(invariant[0], standard[0], rtol=1e-3, atol=1e-3)
        standard_results.append(standard)
    standard_differences = [max_difference(alone, batched) for alone, batched in standard_results]
    if args.require_standard_difference:
        assert any(
            difference != 0.0 for difference in standard_differences
        ), "standard kernels were invariant in this run"


if __name__ == "__main__":
    main()

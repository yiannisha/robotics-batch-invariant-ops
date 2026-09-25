# π0 reference investigation

## Scope

This is the pre-fix PyTorch π0 reference path with stock `nn.Conv2d` and
rank-3 attention matmuls. It uses random weights and synthetic inputs, so the
result is explicitly limited to this numerical regression configuration.

## Baseline

Identical B=1 runs are bitwise repeatable. The target action diverges at every
tested batch size above one, for both duplicate and unrelated compositions.
At B=2, 10 of 28 bfloat16 action values differ and the maximum absolute
difference is 0.0078125. At B=64, the maximum difference is 0.01171875.

The original investigation located the first divergence at the SigLIP 14×14
patch projection (`aten::convolution`), followed after that fix by attention
matmul dispatching to `aten::bmm` rather than the already-overridden
`aten::mm`.

## Replacement operations

- `aten::convolution`: per-sample unfold plus fixed-configuration Triton GEMM
- `aten::bmm`: independent persistent Triton grid for every batch element
- existing invariant `mm`, `addmm`, reductions, and log-softmax overrides

The integration also required compatibility with PyTorch 2.5's accelerator
API and one-element convolution parameter lists, and with Triton 3.1's
`tl.range` signature.

## Result

PASS for this configuration. Every one of the 28 action values is bitwise
identical for B = 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, and 64 in
both duplicate and unrelated composition modes. See `baseline.json` and
`fixed.json` for tensor hashes and exact difference diagnostics.

This is not yet a claim about official π0 or π0.5 checkpoints.

## Operator performance

On the stated environment with bfloat16 inputs, the SigLIP Conv2D replacement
is 1.86× slower at B=1, 2.92× at B=8, and 4.46× at B=32. Incremental peak
allocation is lower than stock cuDNN at B=8 and B=32. The attention BMM is
2.86× slower at B=1, 3.16× at B=8, and 1.13× at B=32. Raw measurements,
shapes, throughput, and memory are in `benchmarks.json`.

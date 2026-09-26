# π0.5 public-weight PyTorch investigation

## Scope and implementation choice

This is the π0.5 path from the PyTorch implementation vendored at commit
`f1eca35` of
[`yiannisha/batch-invariant-pizero`](https://github.com/yiannisha/batch-invariant-pizero).
It executes the 14.47 GB public `lerobot/pi05_base` PyTorch safetensors
checkpoint directly. No JAX code, checkpoint, or conversion is involved.

The vendored loader needed two small compatibility fixes for this checkpoint:
π0.5's time-MLP input is 1024 rather than the state model's 2048, and its
adaptive normalization stores modulation parameters without separate base
norm weights. The same gaps were present in the maintained upstream commit
`37ab0be`. The exact patch is
`scripts/model_invariance/patches/pi_zero_pytorch_pi05_compat.patch`.

## Nondeterminism controls

The model runs in evaluation mode with an explicit per-sample 50×32 flow-noise
tensor. The adapter intercepts the implementation's noise allocation, supplies
the batched explicit tensor, and asserts that the returned original noise is
bitwise equal. Inputs, weights, ten integration time points, precision, and
cache configuration are otherwise fixed. Repeating B=1 produces the same
action hash.

## Baseline and first divergence

Stock CUDA inference failed at every tested request batch above one through
B=64 for duplicate and unrelated companions. The first divergence is the
first SigLIP patch projection. Each request has three views, so changing the
request batch from 1 to 2 changes the Conv2d batch from 3 to 6. Its input for
the target's three views remains exact, while 884,543 of 884,736 float32
outputs differ (maximum `7.791966e-4`).

Profiling proves the specific cause: `aten::cudnn_convolution` selects an NCHW
float32-FFMA implicit GEMM at vision B=3 and an NHWC TF32 tensor-core implicit
GEMM at vision B=6. The layouts, tiles, and reduction arithmetic therefore
depend on the companion samples.

## Iterative propagation

The initial target noise remains exact at ODE evaluation 0, but its predicted
flow already differs by `8.5145235e-5`. From evaluation 1 onward, the
integrator state also differs. The largest observed flow difference is
`6.480217e-4` at evaluation 15; the final 50×32 action differs in 1,577 of
1,600 elements with maximum error `6.914139e-5`. See `iterations.json` for all
18 midpoint evaluations.

## Replacement and end-to-end result

The existing generic `aten::convolution`, MM/addmm, BMM, softmax, log-softmax,
and mean overrides cover this implementation. Conv2d uses a per-sample unfold
and fixed-configuration Triton GEMM; explicit SigLIP and policy attention
einsums dispatch to the invariant BMM path. No π0.5-specific numerical kernel
or per-sample model loop was added.

With `set_batch_invariant_mode()` enabled, every one of the 1,600 target action
values is bitwise identical at B=1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32,
33, and 64, under both duplicate and unrelated composition. All 18 captured
ODE states and predicted flows are also exact at B=2. No later divergence was
found.

## Performance

At the actual float32 patch-convolution shape, the invariant implementation is
3.53×, 8.27×, and 12.02× slower than stock cuDNN at B=1, 8, and 32. For the
actual SigLIP QK BMM it is 1.75×, 1.28×, and 1.33× slower; for attention-value
BMM it is 2.75×, 1.35×, and 1.37× slower. Exact latency, throughput, memory,
shape, and environment records are in `benchmarks.json`.

## Evidence boundaries

This proves numerical batch invariance for the executed PyTorch implementation,
checkpoint, synthetic input configuration, and hardware/software stack. It is
not a policy-quality evaluation and does not assert equivalence between this
open-source implementation and Physical Intelligence's private production
runtime.

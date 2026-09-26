# π0-FAST public-weight PyTorch investigation

## Scope and implementation choice

This experiment uses the maintained PyTorch `pi0_fast` implementation at
LeRobot commit `e595b790` with the public `lerobot/pi0fast-libero` safetensors
checkpoint. Physical Intelligence's OpenPI release does not provide a runnable
PyTorch π0-FAST policy, so this is the closest maintained open-source PyTorch
implementation. No JAX implementation, JAX checkpoint, or conversion is used.

The policy is exercised with real observations from episode zero of the public
`lerobot/libero` dataset: two camera images, the corresponding 8-D state, and
the recorded language task. The third configured image slot is masked. The
adapter performs the checkpoint's state normalization, prompt discretization,
FAST token decoding, inverse DCT, and action unnormalization.

## Nondeterminism controls

The model runs in evaluation mode with greedy decoding and KV caching. Inputs,
precision, tokenizers, maximum decode length, and preprocessing are fixed. The
common harness resets host and CUDA RNG state before each invocation and first
confirms repeated B=1 inference is exact.

## Baseline result

Stock inference is repeatable but not batch-invariant. For frame zero, both
duplicate and unrelated compositions fail at B=3, 4, 5, and 7. B=4 changes all
70 values in the final 10×7 robot action, with maximum absolute difference
`1.1920289993`. Some larger batches happen to select the B=1 greedy sequence;
those isolated passes do not negate the failures at other batch sizes.

The result generalizes beyond one target. Frames 0, 10, and 20 were each tested
at B=3 and B=4 under both composition modes. Stock inference failed all 12
comparisons; the library-enabled run passed all 12 exactly.

## First divergence and cause

At unrelated B=2, the vision patch convolution, complete vision tower,
multimodal projector, layer-0 input normalization, attention Q/K/V, attention
output, MLP gate projection, and MLP up projection are all exact. The first
difference is layer 0's MLP down projection. Its target input is bitwise equal,
but 12,305 of 1,984,512 bfloat16 output values differ, with maximum difference
0.5.

The exact operation is `aten::mm` at `[969,16384] × [16384,2048]`. Because
`nn.Linear` flattens request and sequence dimensions, B=2 changes M from 969 to
1938. Profiling shows that cuBLASLt selects a split-K NVJet GEMM and separate
split-K reduction for M=969, versus a non-split-K NVJet GEMM for M=1938. This
changes accumulation order for the unchanged target rows. The raw profile and
tensor hashes are in `first_divergence_profile.json`; the concise boundary is
in `first_divergence.json`.

The stock first-token language-head output consequently differs in 41,736 of
257,152 values (maximum 1.0). Greedy argmax masks this at B=2, but it changes
the generated action tokens at the failing B=3–7 sizes.

## Independent later divergence

The B=2 trace also exposes a distinct reduction issue on autoregressive call
17. The input to layer 0's post-attention RMSNorm is exact, while its output
differs at one bfloat16 element by `6.103515625e-5`. Transformers' Gemma
RMSNorm computes `x.pow(2).mean(-1, keepdim=True)`; the stock mean reduction is
batch-shape dependent here. This is repaired by the library's fixed reduction
tree registered for `aten::mean.dim`.

## Replacement and end-to-end result

The generic MM and mean overrides repair both proven causes. The other enabled
generic overrides cover the remainder of the execution without model-specific
source changes or per-request model loops. With `set_batch_invariant_mode()`:

- every traced layer-0 boundary and all 24 language-head calls are exact at B=2;
- all three real target frames are exact at B=3 and B=4 in both compositions;
- every value in the final 10×7 action is exact at B=1, 2, 3, 4, 5, 7, 8, 9,
  15, 16, 17, 31, 32, 33, and 64 in both compositions.

No later divergence was found.

## Token-storage batching quirk

LeRobot allocates a fixed 256-token result and stops greedy decoding only when
every batch member has emitted the `|` action delimiter. A companion that runs
longer can therefore cause tokens to be written after the target's delimiter.
Those storage positions are ignored by the official FAST decoder and are not
part of the robot action. The adapter compares the decoded and unnormalized
10×7 action, which is the policy output. With real consecutive LIBERO frames,
the tested target and companions emitted the delimiter at the same step.

## Performance

For the actual PI0-FAST MLP down-projection shape, the invariant GEMM is 1.17×,
1.39×, and 1.35× slower than stock at B=1, 2, and 8. For the decoding RMS mean
shape, the tiny invariant kernel is 4.03×–4.24× slower in latency (roughly
0.030–0.036 ms versus 0.007–0.009 ms). Exact latency, throughput, memory, and
environment data are in `benchmarks.json`.

## Evidence boundaries

This proves numerical batch invariance for the executed PyTorch implementation,
public checkpoint, real LIBERO inputs, and recorded hardware/software stack. It
does not claim equivalence to Physical Intelligence's private production
runtime or measure policy quality.

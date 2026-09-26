# SmolVLA LIBERO PyTorch investigation

## Scope

This experiment uses LeRobot's maintained PyTorch implementation at commit
`e595b790` and the complete public `HuggingFaceVLA/smolvla_libero` checkpoint
at revision `6721902`. Its 604,934,176 state elements cover SmolVLM, the action
expert, and all projections. Real two-camera images, 8D robot state, and task
text come from public LIBERO episode zero. The released preprocessor supplies
tokenization and state normalization; the postprocessor supplies action
unnormalization. The inference output is the complete 50x7 continuous action
chunk after ten Euler flow steps.

The checkpoint is loaded strictly into the official classes. Because its
safetensors are complete, the adapter disables the constructor's redundant
base-VLM weight download before applying the checkpoint; architecture,
processor assets, learned tensors, and inference arithmetic remain unchanged.

## Nondeterminism controls

The model is in evaluation mode and repeated B=1 inference is bitwise exact.
Each example owns a deterministic 50x32 float32 initial flow-noise tensor which
is stacked with the ordinary inputs. Python, NumPy, CPU, and CUDA RNGs are reset
before every inference. The adapter follows the released prefix-cache and
ten-step denoising bodies explicitly so every intermediate flow state can be
fingerprinted without changing their operations.

## Baseline and repaired result

Stock inference fails every duplicate and unrelated composition above B=1 for
B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64`. At B=2, all 350 final action values
differ with maximum absolute difference `0.0052464902`. The largest final
difference in the matrix is `0.018965766` at B=5. Frames 0, 20, and 40 fail all
12 independent duplicate/unrelated B=2/B=4 checks.

With `set_batch_invariant_mode()`, the complete final action is bitwise exact
for all 15 batch sizes through B=64 in both composition modes. All 12
three-target checks are also exact.

## First divergence: SmolVLM modality connector MM

Preprocessing, the 16x16 vision patch convolution, all 12 vision transformer
blocks, and the connector input are exact. The first mismatch is the
bias-free modality projection after spatial pixel shuffle:

```text
policy B=1: [1, 64, 12288] -> flattened M=64
policy B=2: [2, 64, 12288] -> flattened M=128
weight:     [12288, 960]
operator:   aten::linear -> aten::matmul -> aten::mm
```

cuBLAS launches the same named 64x64x8 split-K kernel family, but its grid and
reduction grouping change with flattened M. The target first differs in 58,074
float32 values with maximum absolute difference `4.5776367e-5`. The generic
fixed-schedule MM implementation repairs this boundary.

## Second divergence: state-projection addmm

With only MM repaired, vision embeddings are exact and the next mismatch is
the bias-bearing state projection:

```text
policy B=1: [1, 32] x [32, 960]
policy B=2: [2, 32] x [32, 960]
operator:   aten::linear -> aten::addmm
```

B=1 selects cuBLAS's GEMV kernel while B=2 selects a small-N GEMM kernel. The
output first differs in 329 values with maximum difference `5.9604645e-8`.
Adding the generic fixed-schedule addmm implementation makes every captured
prefix, action-expert, flow-step, and final-action boundary exact. BMM,
convolution, and reduction replacements are not independently required for
this checkpoint.

## Iterative amplification

The connector discrepancy reaches the cached prefix output in 100,993 values
with maximum difference `1.75`. Every one of the 1,600 padded action-state
values differs after the first flow update. Maximum state difference grows
from `0.00338185` after step zero to peaks of `0.0194653` after step seven and
ends at `0.0190591` after step nine. Official unnormalization yields the B=2
final 50x7 maximum difference of `0.00524649`.

## Performance

At the actual float32 connector shape, invariant MM latency is 1.365 ms at
B=1/B=2 and 1.378 ms at B=8, versus 0.060/0.090/0.298 ms for stock. The
resulting latency ratios are 22.68x, 15.25x, and 4.63x. Both implementations
have the same measured incremental output allocation. Full values and software
versions are in `benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned public PyTorch LIBERO
checkpoint, real observations, released pre/postprocessing, prefix KV cache,
all ten flow steps, and 50x7 continuous action output through B=64. It does not
claim task success, training invariance, closed-loop performance, or other
SmolVLA checkpoints.


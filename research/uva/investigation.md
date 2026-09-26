# Unified Video-Action LIBERO-10 PyTorch investigation

## Scope

This experiment uses the authors' official PyTorch implementation at commit
`fe3dec81` and the complete public LIBERO-10 EMA checkpoint. Real 16-frame
windows and task text come from public LIBERO episode zero. The checkpoint
contains 465,177,545 parameters/buffers in float32 and runs the released VAE,
CLIP text encoder, MAR video/action transformer, 100-step action diffusion,
normalization, and 8x10 final action output.

The released serving configuration uses `task_mode="policy_model"`: it runs
the shared video/action representation but returns after action sampling and
intentionally skips video generation. That is the primary full batch-size
matrix. A separate architectural check uses the released
`full_dynamic_model` branch, controls its additional 100-step video diffusion,
and compares the video latent, officially decoded RGB video, and actions.

## Nondeterminism controls

The model is in evaluation mode. Repeated B=1 inference is bitwise exact. Each
example owns explicit VAE posterior noise, initial action noise, and all 100
action-diffusion step noises. The joint adapter additionally owns initial
video-token noise and all 100 video-diffusion step noises. These tensors are
stacked like ordinary inputs, so changing B cannot change the target's RNG
stream. Python, NumPy, CPU, and CUDA RNGs are reset before every inference.
The adapter otherwise follows the official preprocess, MAR sampling,
normalizer, VAE decoder, clamping, and uint8 RGB conversion.

## Action-only baseline and repaired result

Stock inference fails every duplicate and unrelated composition above B=1 for
B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64`. At B=2, 76 of 80 action values
differ with maximum absolute error `6.3404441e-6`; the maximum observed matrix
error is `3.1732023e-5`. Frames 0, 16, and 32 fail all 12 independent
duplicate/unrelated B=2/B=4 checks.

With `set_batch_invariant_mode()`, both composition modes are exact through
B=17 and duplicate composition is also exact at B=31. Unrelated B=31 and
duplicate B=32 exceed GPU memory because the fixed regular-convolution path
requires a large temporary; an OOM is recorded rather than treated as a
numerical failure. All 12 three-target checks are exact.

## First divergence: CLIP block-0 addmm

Inputs, token embeddings, attention projections, and attention output remain
exact. The first stock mismatch is CLIP text block 0's MLP expansion:

```text
policy B=1 input: [1, 30, 512] -> flattened M=30
policy B=2 input: [2, 30, 512] -> flattened M=60
weight:           [2048, 512]
operator:         aten::linear -> aten::addmm
```

At B=1, profiling reports an `sgemm_largek_lds64` kernel plus a separate
epilogue/scaling sequence. At B=2, cuBLAS selects an
`sm80_xmma_gemm_f32...tilesize32x32x8` kernel. The output first differs in
53,921 values with maximum absolute difference `3.0517578e-5`. The reusable
fixed-schedule addmm implementation repairs this boundary.

## Second divergence: VAE downsample Conv2d

With the GEMM family repaired, the CLIP path and first VAE downsample are
exact. The next mismatch is the second VAE encoder downsample convolution. It
processes `[4,128,128,128]` at policy B=1 and `[8,128,128,128]` at policy B=2,
then emits `[4,128,64,64]` versus `[8,128,64,64]`. cuDNN's batch-shaped
arithmetic changes 130,351 target values with maximum difference
`3.4332275e-5`. Adding the generic regular-convolution replacement makes every
captured VAE, MAR encoder/decoder, learned temporal upsampler, all 100 action
diffusion steps, and the final action exact.

## ConvTranspose3d coverage

UVA's action head contains a learned temporal upsampler with input
`[B,1024,4,16,16]`, weight `[1024,1024,4,1,1]`, and stride `(4,1,1)`. The
existing generic dispatcher rejected transposed convolution even after the
causal addmm/Conv2d repairs. A reusable non-overlapping ConvTranspose3d path now
uses a fixed-schedule grouped BMM per sample and group. Unit tests cover all
three supported dtypes, groups 1/2, numerical agreement, dispatcher routing,
and B=2/5/17 invariance.

At UVA's actual shape, cuDNN itself happens to remain exact for every planned
batch size through B=64; thus this operator is required for complete supported
dispatch, not as an observed causal mismatch in this checkpoint. The fixed
operator is 1.99x/2.72x/3.21x/3.96x/4.37x stock latency at B=1/2/4/8/64 and
uses less peak incremental allocation in the recorded cases.

## Joint video/action branch

At B=2, stock joint inference changes 16,378 of 16,384 video-latent values
(maximum difference `0.0021321774`), 6,080 decoded RGB bytes, and 76 of 80
action values. With the invariant context, duplicate and unrelated B=2 are
bitwise exact for the 4x16x16x16 latent, the official 4x256x256 RGB video, and
the 8x10 action trajectory across all 100 video and 100 action diffusion
steps.

## Evidence boundary

This proves numerical batch invariance for the pinned public PyTorch
LIBERO-10 checkpoint, real observations, released preprocessing, action-only
deployment path through the maximum fitting batches above, and joint decoded
video/action generation at B=2. It does not claim task success, training
invariance, closed-loop performance, or other UVA checkpoints.


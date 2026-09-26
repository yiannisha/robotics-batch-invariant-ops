# X-VLA LIBERO PyTorch investigation

## Scope

This experiment uses the maintained LeRobot PyTorch implementation at commit
`e595b790` and the complete public `lerobot/xvla-libero` safetensors snapshot
at commit `12e8783e`. Real two-camera observations, state, and task text come
from public LIBERO episode zero. The 879,482,456-parameter float32 policy runs
Florence-2, a 24-block action transformer, ten flow updates, and the official
ee6d-to-7D LIBERO action conversion.

## Nondeterminism controls

The policy is in evaluation mode and repeated B=1 inference is bitwise exact.
Each fixture frame owns a deterministic 30x20 float32 flow-noise tensor which
is stacked like any other input. This is necessary because the released
`predict_action_chunk` method accepts a noise argument but does not consume it;
its internal generator otherwise draws a batch-shaped tensor. The adapter
mirrors the official PyTorch inference body to inject that explicit noise and
uses the released preprocessors and final rotation processor unchanged.

## Baseline

Stock inference fails every duplicate and unrelated composition above B=1 for
B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64`. At B=2, 166 of 210 final action
values differ with maximum absolute difference `1.6093254e-6`. At B=33/B=64,
179 values differ with maximum difference `3.272295e-5`. Frames 0, 20, and 40
also fail all duplicate and unrelated B=2/B=4 checks (12 of 12 failures).

## First divergence: Florence stage-3 convolution

Preprocessing, the explicit noise, Florence stages zero and one, and the input
to stage two are exact. The first mismatch is Florence-2's third patch/downsample
convolution:

```text
policy B=1: input [2, 512, 28, 28]
policy B=2: input [4, 512, 28, 28]
weight:     [1024, 512, 3, 3], stride 2, padding 1
operator:   aten::convolution -> aten::cudnn_convolution
```

The two leading dimensions account for the two valid camera views. cuDNN uses
a 64x64 tiled, segment-K-on kernel for B=1 and a 128x128 tiled, segment-K-off
kernel for B=2. The target first differs in 32,703 float32 values, with maximum
difference `4.2438507e-5`. The generic fixed-schedule Conv2d override repairs
this boundary.

## Second divergence: multimodal projector MM

With only convolution repaired, all Florence vision-tower boundaries are
exact, but 154 final action values still differ. The next mismatch is the
bias-free image projection:

```text
policy B=1: [2, 50, 2048] -> flattened M=100
policy B=2: [4, 50, 2048] -> flattened M=200
weight:     [2048, 1024]
operator:   aten::matmul -> aten::mm
```

The stock cuBLAS split-K launch geometry depends on flattened M. Its target
output first differs in 93,093 values with maximum difference `2.3841858e-6`.
Adding the generic fixed-schedule MM/addmm/BMM family makes all 591 captured
boundaries and the final trajectory exact. No additional attention or
reduction repair is required for this checkpoint.

## End-to-end result

With `set_batch_invariant_mode()`, every final 30x7 action is bitwise identical
for all 15 batch sizes through B=64 in duplicate and unrelated composition.
Frames 0, 20, and 40 pass all 12 independent B=2/B=4 checks. The staged trace
shows zero differences under both Conv+GEMM-only repair and the complete
public context.

## Conv2d optimization and performance

The initial per-image Conv2d implementation was prohibitively slow at this
large float32 shape. It was generalized to form all image/group patches at
once and execute one batch-independent persistent BMM. All Conv2d correctness
tests and the 150-test suite pass after the change. The optimization reduces
invariant convolution latency from 2.35/4.68/18.68 ms to 1.20/1.21/2.45 ms at
policy B=1/2/8, leaving ratios of 14.12x/11.44x/12.70x versus cuDNN. Incremental
memory is lower than stock at B=1/B=2 and modestly higher at B=8.

At the projector MM, invariant latency is 7.93x/5.95x/2.44x stock at B=1/2/8
with identical incremental output allocation. Full values and environment are
in `benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned public PyTorch LIBERO
checkpoint, real observations, released preprocessing, ten flow steps, and
official 30x7 action output. It does not claim task success, training
invariance, closed-loop performance, or other X-VLA embodiments/checkpoints.

# OpenVLA-OFT LIBERO-Spatial PyTorch investigation

## Scope

This experiment uses the authors' official PyTorch repository at commit
`e4287e94`, its official bidirectional-attention Transformers fork at
`bc339d9a`, and the complete public
`moojink/openvla-7b-oft-finetuned-libero-spatial` checkpoint at revision
`6d0231af`. The 7,709,173,191 learned elements cover the dual DINOv2/SigLIP
vision backbone, multimodal projector, 7B VLM, proprio projector, and L1
continuous action head.

Inputs are real two-camera images, 8D robot state, and task text from public
LIBERO episode zero. The official tokenizer and dual-tower image processor are
used. A PyTorch implementation of the released centered 90%-area inference
crop avoids pulling TensorFlow's training-data stack into this PyTorch-only
investigation. Released Q01/Q99 state normalization and action
unnormalization are applied. The robot output is the complete 8x7 continuous
action chunk.

The upstream `predict_action()` helper reshapes its result to 8x7 and is
therefore batch-1-only. The adapter preserves the leading dimension while
following the same model operations; it never serializes a batch into separate
model calls. At B=1, the adapter's complete output is bitwise identical to the
released helper for the same processed tensors.

## Nondeterminism controls

The model and all three learned components are in evaluation mode. This L1
regression policy has no inference-time sampling. Python, NumPy, CPU, and CUDA
RNGs are reset before each run. Repeated B=1 inference is bitwise exact.
Preprocessing happens before batch composition, so the target tensors are
identical in every context.

## Baseline and repaired result

Stock inference fails every duplicate and unrelated composition above B=1 for
B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64`. At B=2, 51 of 56 final action
values differ and the maximum absolute difference after official
unnormalization is `0.0078125`. Across the matrix, the target's maximum
difference reaches `0.015625`.

With `set_batch_invariant_mode()`, the complete action chunk is bitwise exact
for all 15 sizes through B=64 in both composition modes. Independent target
frames 0, 20, and 40 fail all 12 stock B=2/B=4 checks, with a largest final
difference of `0.3515625`; all 12 fixed checks are exact.

## First divergence: DINO block-0 fused SDPA

Input embeddings and both DINO patch-convolution calls are exact. The first
mismatch is the first DINO transformer block:

```text
Q/K/V:       [B, 16, 261, 64]
operator:    aten::scaled_dot_product_attention
B=1 kernel:  flash split-K plus split-K reduction
B=2 kernel:  non-split flash forward with a different tile
```

The block output is `[B,261,1024]` bfloat16. Its target sample first differs
in 6,021 elements with maximum absolute difference `0.03125`. Profiling also
shows batch-shape-dependent cuBLASLt algorithms for the block's bias-bearing
linear layers, but replacing convolution, MM/addmm, and BMM while leaving
SDPA stock does not move the first boundary. The generic fixed-schedule SDPA
implementation makes both vision towers, the multimodal projector, and all 32
VLM blocks exact.

## Second divergence: continuous action-head addmm

With only SDPA repaired, the first remaining mismatch is the action head's
wide first linear layer:

```text
logical input: [B, 8, 28672]
flattened B=1: [8, 28672]
flattened B=2: [16, 28672]
weight:        [28672, 4096]
operator:      aten::addmm
```

B=1 selects a 128x8 split-K cuBLASLt kernel, while B=2 selects a 64x16 split-K
kernel. The layer output differs in 64 bfloat16 values with maximum absolute
difference `0.25`. SDPA plus the generic invariant MM/addmm implementation
makes every captured boundary and final action exact. BMM and convolution are
not independently required for this checkpoint.

## Performance

At the actual DINO block-0 shape, invariant SDPA is 7.56x, 8.30x, and 2.91x
slower than stock at B=1, B=2, and B=8. At the actual action-head FC1 shape,
invariant addmm is 4.25x, 4.41x, and 3.28x slower. Full latency and incremental
allocation results are in `benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned public PyTorch
LIBERO-Spatial checkpoint, real observations, official learned components,
dual-camera backbone, full 32-layer VLM, and 8x7 continuous action output
through B=64. It does not claim task success, closed-loop performance,
training invariance, diffusion-head support, FiLM support, or other
OpenVLA-OFT checkpoints.

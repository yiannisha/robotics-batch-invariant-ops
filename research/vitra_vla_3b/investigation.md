# VITRA-VLA-3B batch-invariance investigation

## Result

The official native PyTorch VITRA-VLA-3B policy is deterministic for repeated
identical B=1 runs but is not batch-invariant with stock CUDA operators. It
fails every tested non-unit batch through B=64 in both duplicate and unrelated
composition modes. With `set_batch_invariant_mode()`, the complete policy is
bitwise exact at B=1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, and 64.

The compared robot output is the released postprocessed 16x102 float64 hand
action trajectory. At stock B=2, 794 elements differ, with maximum absolute
difference 2.1309424482751638e-6. By B=5, all 816 predicted right-hand values
differ and the maximum error has amplified to roughly 1.02e-3.

## Checkpoint and reference-path validity

The 15.07 GB official checkpoint contains 899 tensors and 3,766,775,472
elements. Its key set and every shape exactly match the model constructed from
PaliGemma2 metadata: there are no missing, unexpected, converted, or substitute
learned weights. The adapter's explicit-noise implementation is bitwise equal
to VITRA's released private `_forward_act_model` path at B=1 and unrelated B=2.
The only bypassed public-method restriction is its artificial `assert B == 1`.

## First divergence and cause

The image processor, input IDs, state, FOV, DDIM noise, SigLIP patch
convolution, complete vision tower, multimodal projector, and first FOV linear
remain exact. The earliest mismatch is:

```text
exact input:  fov_encoder.projector.2 input, [B, 2304], float32
operation:    aten::linear
weight:       [2304, 2304], float32
first output: 1,169 of 2,304 values differ at B=2
maximum:      2.9802322387695312e-8
```

CUDA profiling confirms the batch-shape-dependent arithmetic choice. B=1 uses
a cuBLAS GEMV kernel; B=2 and B=8 use different small-N GEMM kernels; B=64 uses
a CUTLASS 64x64 SGEMM kernel. Those kernels tile and reduce the shared 2,304
dimension differently, so the same target row receives different float32
rounding. The reusable fixed-schedule `aten::linear` override alone repairs
the final VITRA output; no model-specific numerical special case and no second
operator repair are required.

## Iterative propagation

The stock VLM cognition token already differs in 2,265 elements. With identical
initial DDIM noise, DDIM step 0 differs in 1,565 state values (maximum
7.867813110351562e-6). The maximum difference grows across the ten states,
reaching 2.008676528930664e-5 at step 9. The final normalized action differs in
794 elements before released denormalization. The complete per-step values are
in `first_divergence.json` under `stock_iterative_trace`.

With the full library, preprocessing, VLM hidden state, cognition token, every
one of the ten DDIM states, normalized actions, and unnormalized actions are
all bitwise exact at B=2. All three released images also pass duplicate and
unrelated B=2/B=4 checks as targets.

## Performance

At the actual `[B,2304] x [2304,2304]` FP32 boundary, invariant linear takes
0.518, 0.505, 0.503, and 0.499 ms at B=1, 2, 8, and 64, versus stock 0.011,
0.014, 0.023, and 0.071 ms: 46.16x, 36.46x, 22.31x, and 6.99x slower. This
projection runs once per policy call, so its roughly 0.5 ms absolute cost is
small beside end-to-end VLM plus ten-step DiT inference. Exact latency,
throughput, allocation, shape, and environment records are in
`benchmarks.json`.

## Reproduction artifacts

- `baseline.json`: stock full batch sweep
- `fixed.json`: repaired full batch sweep
- `first_divergence.json`: staged boundary and DDIM-step traces
- `official_equivalence.json`: released-path equality
- `multiple_inputs.json`: three-image target regression
- `benchmarks.json`: actual FOV projection benchmark
- `backend_kernels.json`: CUDA kernel-family evidence

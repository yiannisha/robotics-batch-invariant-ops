# OpenPI π0.5 batch-invariance investigation

## Scope

This run uses the official OpenPI PyTorch π0.5 architecture and inference code
at commit `215abfb`, with deterministic random parameters. It is an
architecture/operator regression, not an official-checkpoint policy claim.
The public `pi05_droid` checkpoint was enumerated directly in GCS and is
12.43 GB in JAX form; the available disk could not hold it alongside the
additional converted PyTorch checkpoint.

## Nondeterminism controls

The adapter uses evaluation mode, disables compilation and augmentation, and
passes flow noise explicitly as part of every sample. Repeating the exact B=1
batch produced the same action hash. Every comparison therefore reuses the
same target images, prompt, state, noise, number of Euler steps, weights, and
precision.

## Baseline

Stock CUDA inference failed every tested B greater than one in both duplicate
and unrelated composition. At B=2 all 480 output action scalars differed from
B=1; the maximum absolute difference was 0.0029878616. The target output was
otherwise repeatable. The first differing module boundary was the first VLM
MLP `down_proj`, a bias-free linear layer that dispatches to `aten::mm` after
flattening the request batch and 968-token sequence into GEMM M.

## Iteration 1: existing operator replacements

The previously integrated MM, addmm, BMM, convolution, log-softmax, and mean
overrides repaired B=2. They did not complete the model: every B from 3 through
64 produced the same non-reference action hash. Leaf-module tracing found
exact Q/K/V projections in SigLIP vision layer 0 followed by a different input
to `out_proj`. Profiling the vision attention shape showed the route
`aten::scaled_dot_product_attention` →
`aten::_scaled_dot_product_flash_attention` →
`pytorch_flash::flash_fwd_kernel`. OpenPI forces eager attention for the VLM
and action expert during sampling, but leaves the SigLIP tower configured for
SDPA.

## Iteration 2: batch-invariant SDPA

The library now overrides the public `aten::scaled_dot_product_attention`
CUDA operator. The replacement computes each flattened attention head with
the batch-invariant BMM kernel, applies a Triton softmax with one fixed
reduction program per row, and performs the value product with another
batch-invariant BMM. It supports evaluation inference (`dropout_p=0`), causal
masking, additive/boolean masks, and grouped-query head expansion.

With this replacement enabled, the final `[B,15,32]` action trajectory was
bitwise identical for the target sample at every tested B through 64, for both
duplicate and unrelated companions, across all 10 Euler steps. No later
divergence was found.

## Performance

For the actual SigLIP shape `[B,16,256,72]` in bfloat16, the replacement was
7.27×, 4.27×, and 6.23× slower than fused SDPA at B=1, 8, and 32 respectively.
Its incremental allocation at B=32 was 171,966,464 bytes versus 38,274,048
bytes for stock SDPA. Exact measurements and the full software environment are
in `benchmarks.json`.

# MolmoAct2 public-weight PyTorch investigation

## Scope

This experiment uses the official PyTorch MolmoAct2 repository at commit
`66b87e64` and the public `allenai/MolmoAct2-LIBERO` Transformers/PyTorch
safetensors snapshot at commit `0d24a92b`. The 5.44-billion-parameter model is
loaded directly with its released custom Transformers code. No JAX source,
weight, or conversion is involved.

The executed policy is the intended continuous-action configuration: its
25-block image encoder and 36-block language backbone produce per-layer KV
context for a 36-block action expert, followed by ten flow-matching updates.

## Upstream B>1 compatibility correction

The released attention-bias builder creates its causal mask with leading
dimension one and then updates it in place with a B-wide image-token mask.
Consequently the otherwise public batch path raises a broadcast error for every
B greater than one. The adapter applies a minimal runtime correction: expand
and clone that causal mask to B before the same image-mask update. At B=1 the
corrected bias and official result are bitwise identical. This fixes batching
correctness only; stock PyTorch remains numerically batch-dependent afterward.

## Inputs and nondeterminism controls

Targets and companions are real camera frames and states from episode zero of
the public LIBERO dataset. The official processor constructs two-view visual
tokens plus the task, setup/control tags, and discretized normalized state. The
checkpoint's `libero` statistics normalize raw 8-D state and unnormalize the
final 10x7 robot action.

The model is in evaluation mode, CUDA graphs and dropout are disabled, and all
host/device RNGs are reset before each invocation. The public action API accepts
a CUDA generator rather than a trajectory tensor. A freshly seeded generator
is used on every invocation; for the checkpoint's 10x32 bfloat16 trajectory,
its first-sample random tensor was independently verified bitwise identical for
B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64`. Repeated B=1 inference is bitwise
exact.

## Baseline results

Stock PyTorch fails both duplicate and unrelated composition at every tested B
above one, through B=64. At B=2, 41 of the 70 final float32 action values differ
with maximum absolute difference `0.00390625`. From B=17 upward, 42 values
differ by the same maximum. Frames 0, 10, and 20 also fail all duplicate and
unrelated B=2/B=4 checks (12 of 12 failures).

## First divergence and root cause

For B=1 versus unrelated B=2, preprocessing, the initial flow trajectory, ViT
patch projection, all 25 vision blocks, visual pooling, and the image-projector
input are bitwise exact. Within the projector, `w1`, SiLU, `w3`, and the product
fed to `w2` also remain exact. The first difference is:

```text
image_projector.w2:
    [B * 392, 9728] @ [9728, 2560]
    -> [B * 392, 2560]
```

The bias-free PyTorch `Linear` lowers through `aten::matmul` to `aten::mm`.
For B=1, cuBLASLt chooses an NVJet split-K kernel plus a separate split-K
reduction. At B=2 it chooses a different non-split-K NVJet kernel. The identical
target input first changes in 3,686 bfloat16 output elements (mean absolute
difference `7.501e-5`; the largest isolated value difference is `32.0`). The
complete tensor hashes and exact CUDA kernel names are in
`first_divergence.json`.

The existing generic fixed-schedule `matmul_persistent`/`aten::mm` replacement
addresses the causal operation. No model-specific numerical kernel was needed.

## Iterative propagation

The changed visual context reaches the first action block at flow step zero.
The first normalized velocity differs in 95 of 320 values with maximum
difference `0.0078125`. By step nine, 126 values differ and the maximum reaches
`0.046875`. Every recorded vision, language, action, per-step velocity, and
final-action boundary is exact with batch-invariant operators. See
`iterations.json`.

## End-to-end result

With `set_batch_invariant_mode()`, every final 10x7 action value is bitwise
identical for B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64` in both duplicate and
unrelated composition. Fixed mode also passes all 12 checks across target
frames 0, 10, and 20 at B=2/B=4. No later divergence was found.

## Performance

For the exact image-projector MM shape on the recorded H100, the invariant
kernel costs 2.38x, 1.75x, and 1.41x stock latency at B=1, B=2, and B=8,
respectively. Absolute invariant latencies are `0.122`, `0.202`, and `0.441 ms`;
incremental output memory is unchanged. Full measurements are in
`benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned public PyTorch LIBERO
checkpoint, real observations, continuous action mode, and recorded
environment. Depth reasoning is disabled in this checkpoint, and the separate
discrete action-tokenizer path was not tested.

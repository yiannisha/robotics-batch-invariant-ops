# GR00T N1.7 public-weight PyTorch investigation

## Scope

This experiment uses NVIDIA's official PyTorch Isaac-GR00T repository at
commit `51d4c89f` and the public `nvidia/GR00T-N1.7-LIBERO` `libero_10`
safetensors snapshot at commit `2ea293aa`. The complete released backbone and
action-head weights are loaded into the official `Gr00tN1d7` classes. No JAX
source, weight, or conversion is involved.

The executed policy consumes the official two-camera LIBERO prompt and state,
runs the Qwen3-VL/Cosmos-derived visual-language backbone and 32-block action
DiT, performs four flow-matching updates over a 40x132 normalized trajectory,
then decodes and applies the official LIBERO gripper convention to a 16x7
action result.

## Gated-base loader workaround

NVIDIA's constructor first downloads `nvidia/Cosmos-Reason2-2B`, then replaces
that model with all tensors in the GR00T checkpoint. The Cosmos repository is
gated and its license was not accepted in this environment. During
construction only, the adapter supplies an uninitialized Qwen3-VL module from
the exact public architecture config and the corresponding public processor.
The official Hugging Face loader then fills NVIDIA's model with the complete
GR00T shards. The Qwen snapshot contributes no weights; this bypasses a
redundant gated download rather than changing or converting the model.

## Inputs and nondeterminism controls

Targets and companions are real camera frames and states from episode zero of
the public LIBERO dataset. The released processor constructs two-view visual
tokens, task text, and normalized 8-D state. Its checkpoint statistics decode
the final action, followed by the official LIBERO gripper binarization and sign
inversion.

The model is in evaluation mode and all host/device RNGs are reset before each
invocation. Initial flow noise is generated independently per example as an
explicit 40x132 bfloat16 input, then stacked; it therefore cannot change when
batch composition changes. Repeated B=1 inference is bitwise exact.

## Baseline results

Stock PyTorch fails duplicate and unrelated composition at every tested B
above one, through B=64. At B=2, 78 of the 112 final float32 action values
differ with maximum absolute difference `0.012660384`. Frames 0, 10, and 20
also fail all duplicate and unrelated B=2/B=4 checks (12 of 12 failures).

## First divergences and root cause

For B=1 versus unrelated B=2, preprocessing, initial flow noise, the complete
visual-language backbone, projected context, state encoder, and the first
action-encoder input are bitwise exact. The first difference is the
category-conditioned second action projection:

```text
action_encoder.W2:
    [B, 40, 3072] @ [B, 3072, 1536]
    -> [B, 40, 1536]
```

This operation lowers to `aten::bmm`. At B=1, cuBLASLt selects an NVJet split-K
kernel plus a separate split-K reduction; at B=2 it selects a different
non-split-K kernel. Two bfloat16 output elements first differ, with maximum
absolute difference `0.00048828125`.

Repairing only BMM is insufficient: 76 final action values still differ. With
BMM fixed, the next difference is the first action-DiT cross-attention key
projection. Its bias-bearing PyTorch `Linear` lowers to `aten::addmm`:

```text
action_dit.block0.cross_attention.k:
    [B * 156, 2048] @ [2048, 1536] + bias
    -> [B * 156, 1536]
```

The B=1 and B=2 calls select different NVJet kernels. The first target output
then differs in 210,049 bfloat16 elements, with maximum absolute difference
`0.140625`. The existing generic fixed-schedule BMM and MM/addmm replacements
address both causal operations; no model-specific numerical kernel was added.
Exact tensor hashes and full CUDA kernel names are in `first_divergence.json`.

## Iterative propagation

At flow step zero, the normalized velocity differs in 3,764 of 5,280 values
with maximum difference `0.017578125`. By step three, 4,298 values differ and
the maximum reaches `0.046875`. Every recorded backbone, action-DiT, flow, and
final-action boundary is exact with the complete batch-invariant operator
context. See `iterations.json`.

## End-to-end result

With `set_batch_invariant_mode()`, every final 16x7 action value is bitwise
identical for B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64` in both duplicate and
unrelated composition. Fixed mode also passes all 12 checks across target
frames 0, 10, and 20 at B=2/B=4. No later divergence was found.

## Performance

For the exact action-encoder BMM shape on the recorded H100, the invariant
kernel costs 2.78x, 3.78x, and 1.80x stock latency at B=1, B=2, and B=8.
For the exact action-DiT addmm shape, the ratios are 2.86x, 3.53x, and 1.69x.
Incremental output memory is unchanged in all cases. Full measurements are in
`benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned public PyTorch LIBERO-10
checkpoint, real observations, official continuous-action path, four released
flow steps, and recorded environment. It does not claim task success or cover
other embodiments, GR00T checkpoints, training, or closed-loop environment
rollouts.

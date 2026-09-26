# InternVLA-A1.5 public-weight PyTorch investigation

## Scope

This experiment uses the official PyTorch InternVLA-A1.5 policy at upstream
commit `e6fc904f` and the public `InternRobotics/InternVLA-A1.5-Libero`
safetensors checkpoint. Qwen3.5-2B is loaded from its pinned public PyTorch
safetensors snapshot. No JAX implementation, JAX weight, or conversion is used.

Both public action inference implementations were executed:

- `standard`: the ordinary bfloat16 action-expert path with eager full attention;
- `optimized`: the upstream recommended action-only path, with a float32 action
  expert, SDPA, and its optimized denoising implementation.

The A1.5 checkpoint uses foresight during training but discards the WAN branch
for recommended action inference. The external Wan2.2 video generator is not a
robot-policy output of either executed action-only configuration, so no video
invariance claim is made. The three WAN-related checkpoint keys reported as
unexpected are precisely the omitted training/video projection state.

## Inputs and nondeterminism controls

The target and companions are real frames and states from episode zero of the
public LIBERO dataset. Each sample contains two camera views, the corresponding
8-D robot state, and the recorded instruction. The official processor creates
the Qwen multimodal prompt and discretized normalized state. Checkpoint
`libero_10` statistics normalize state and unnormalize/clip the final 8x7 robot
action.

The model is in evaluation mode. Host and CUDA RNGs are reset before every
invocation, and flow noise is an explicit per-sample input seeded by source
frame, so target noise never depends on B. Preprocessing, precision, prompt
length, ten Euler flow steps, task, cameras, state, and normalization metadata
are fixed. Repeated B=1 execution is bitwise exact in both backends.

## Baseline results

The standard backend fails both duplicate and unrelated composition at every
tested B above 1: `2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64`. At B=2,
46 of 56 final float32 action values differ with maximum absolute difference
`0.0010518134`. The largest observed maximum across the sweep is
`0.0017530024` at B=3.

The optimized backend happens to match B=1 at B=2, 3, 5, and 7, but fails at
B=4 and every tested B from 8 onward. A failing batch changes 37 of 56 action
values with maximum difference `0.0009723902`. Isolated stock passes do not
establish batch invariance.

The result generalizes to other observations. Frames 0, 10, and 20 were tested
at B=2 and B=4 in both composition modes. Standard stock fails all 12 cases.
Optimized stock passes the six B=2 cases and fails the six B=4 cases. Fixed
mode passes all 24 cases exactly.

## First divergence and root cause

For standard B=1 versus unrelated B=2, the following target tensors are all
bitwise exact: Conv3d visual patch embedding, complete vision output, complete
language prefix, action layers 0–18 at flow step zero, layer-19 input norm,
post-RoPE Q/K/V, QK-transpose attention scores, the masked scores, and float32
softmax probabilities.

The first difference is the second eager-attention matrix product in action
expert layer 19:

```text
probabilities [B, 8, 58, 708]
    @ values [B, 8, 708, 256]
    -> attention value product [B, 8, 58, 256]
```

Rank-4 `torch.matmul` lowers to `aten::bmm` after flattening requests and heads.
B=1 therefore runs 8 matrices and B=2 runs 16. The inputs for the target are
identical, but stock CUDA selects a 32x32 WMMA CUTLASS kernel for batch-heads 8
and a 64x64 tensor-op CUTLASS kernel for batch-heads 16. Two bfloat16 output
elements first differ, by at most `6.103515625e-5`. The full boundary trace and
CUDA kernel names are in `standard_first_divergence_trace.json`; the concise
causal record is in `first_divergence.json`.

The existing generic fixed-schedule `bmm_persistent` repair addresses this
cause. Qwen3.5 also exercises regular Conv3d patch projection and grouped
causal Conv1d. Because the global `aten::convolution` override must preserve
those calls, this milestone adds generic 1-D and 3-D convolution handling. The
Qwen depthwise Conv1d has a dedicated fixed-order Triton kernel; other supported
Conv1d/Conv3d cases use batch-independent im2col/BMM decomposition. These
convolutions were not the first stock divergence in this input, but must remain
correct when invariant mode is enabled.

## Iterative amplification

The first flow velocity differs in 36 of 256 normalized values with maximum
difference `0.0078125`. The mismatch feeds the next Euler state. By flow step 9,
215 values differ and the maximum reaches `0.04296875`. All ten velocity tensors
are exact in fixed mode. See `iterations.json`.

## End-to-end result

With `set_batch_invariant_mode()` both standard and optimized action paths are
bitwise exact at every value of the final 8x7 action for B=`1, 2, 3, 4, 5, 7,
8, 9, 15, 16, 17, 31, 32, 33, 64`, under both duplicate and unrelated
composition. The fixed trace also makes every recorded vision, language,
action-layer, attention, per-step velocity, and final-action boundary exact.
No later divergence was found.

## Performance

On the recorded H100/PyTorch 2.8 environment, the actual action-attention BMM
replacement costs 2.45–6.40x stock latency across request B=1, 2, and 8
(`0.034–0.116 ms` invariant). The Conv3d patch replacement costs 2.24–4.74x
(`0.624–4.884 ms`). After adding the specialized depthwise kernel, invariant
Qwen causal Conv1d is faster than stock for these shapes: 0.49–0.59x stock
latency (`0.034–0.274 ms`). Exact throughput, memory, and shapes are in
`benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned official PyTorch policy,
public checkpoints, real LIBERO samples, two upstream action backends, and the
recorded environment. It does not claim invariance for the unexecuted external
WAN video generator or equivalence to private deployment runtimes.

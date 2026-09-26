# WSA Base LIBERO PyTorch investigation

## Executed path

This investigation uses official WSA source commit `bfee742c`, the public
`zaleni/WSA-Base-LIBERO` checkpoint at revision `4e992636`, and its released
causal WSA Base inference path. The adapter performs the same Qwen3-VL image and
text preprocessing, state normalization, learned 3D query construction, 28
joint und/gen/action transformer layers, ten flow updates, and action
unnormalization as the official LIBERO server. A direct B=1 and unrelated B=2
comparison against `WSABaseModel.sample_actions` is bitwise exact.

The checkpoint's 1,288 tensors cover all 3,156,147,248 saved learned elements.
No JAX weights are used or converted. The small inference-only source patch
constructs the Qwen architecture without first downloading redundant base
weights, then the official loader fills the model from WSA safetensors. A key
audit rejects any missing learned tensor or unexpected tensor; only the frozen
external Cosmos module and Qwen's tied input-embedding alias are absent by
design.

## Determinism and inputs

The target and companions are the first 64 real synchronized observations from
public LIBERO episode zero. Each sample has independently fixed 10x32 float32
flow noise. Repeating the identical B=1 inference produces identical bits.
Three distinct target frames additionally pass duplicate and unrelated B=2 and
B=4 checks. These are numerical inference regressions, not task-success claims.

## Iterative divergence search

Stock inference fails at every B greater than one through B=64 for both
duplicate and unrelated composition. At unrelated B=2 all 70 final action
values differ, with maximum absolute difference `0.0030884146690368652`.

The first divergence is layer 0 of the understanding expert. Its MLP gate and
up projections and the exact input to `down_proj` still agree, while the
`aten::linear` output for logical input `[B,246,6144]` and weight
`[2048,6144]` differs in 1,500 bfloat16 elements for the target. The generic
rank-3 invariant linear path repairs that boundary.

Linear repair exposes the next independent boundary in layer 0 of the action
expert. The input to `post_attention_layernorm` is exact, but its float32
RMSNorm `aten::mean.dim` reduction over `[B,11,1024]` differs in two bfloat16
outputs after scaling. The existing fixed-tree invariant mean repairs it.
Linear plus mean alone makes the final B=1/B=2 action exact; enabling the full
library also makes all ten recorded velocities and all ten evolving flow states
bitwise identical.

## Result

With `set_batch_invariant_mode()`, the final 10x7 unnormalized trajectory is
bitwise identical at every required batch size through B=64, for duplicate and
unrelated companions. The result uses the public learned checkpoint, real
observations, explicit per-sample randomness, the full iterative policy path,
and the public context-manager API.

Operator microbenchmarks at the real WSA shapes are in `benchmarks.json`; they
report both latency/memory and the stock/fixed invariance result. The complete
stock/fixed sweeps, official equivalence, multi-target checks, and per-step
trace are committed beside this document.

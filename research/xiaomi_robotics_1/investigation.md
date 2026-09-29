# Xiaomi-Robotics-1 RoboCasa batch-invariance investigation

## Scope and checkpoint audit

This investigation uses Xiaomi's official native PyTorch repository at commit
`0dd7aef8`, its complete public `Xiaomi-Robotics-1-RoboCasa` checkpoint at
revision `82b7c221`, and the checkpoint-bundled Transformers model and
processor. The three native safetensor shards contain 1,120 tensors and
5,053,149,696 unique learned elements. A strict key audit finds no unexpected
tensors and only the expected tied `vlm.lm_head.weight` alias absent.

The policy combines a Qwen3-VL backbone with a 36-layer, 1024-wide action DiT.
It caches the VLM's keys and values, conditions a 10x60 action trajectory on a
60-D state, and runs five Euler flow steps. The adapter makes the released
batch-shaped random noise explicit per sample. For three real observations,
its decoded 10x7 float32 actions are bitwise identical to the unmodified
released B=1 seeded `forward` path.

Inputs use 64 consecutive frames from a public RoboCasa episode with the same
left-base, right-base, and wrist camera roles and the released multi-view prompt
template. The episode stores EEF pose, not Xiaomi's expected joint state, so
the adapter uses deterministic plausible Panda joint/gripper values padded to
60 dimensions. This is a numerical inference test, not a policy-quality claim.

## Stock result and first divergence

Repeated stock B=1 calls are exact. Stock fails every B above one through 64
for both duplicate and unrelated compositions. At B=2, 60 of 70 decoded robot
action values differ, with maximum absolute difference 0.0171875954.

The vision path, layer-0 language attention, and input to Qwen layer 0's MLP
down-projection are exact. The first mismatch is
`vlm.model.language_model.layers.0.mlp.down_proj`, an `aten::linear` with input
`[B,259,9728]` and weight `[2560,9728]`: 2,414 BF16 outputs differ at B=2,
with maximum absolute difference 0.0078125.

After invariant linear is enabled, VLM caches and the first two DiT velocities
are exact. The next mismatch is `aten::mean.dim` inside the RMSNorm at DiT
layer 7 during Euler step 2. Its `[B,12,1024]` input is exact, while eight BF16
normalized values differ by as much as 0.0078125. The iterative flow then
amplifies that error into changed decoded action values.

## Repair and coverage

Invariant linear plus fixed-tree mean is the minimal sufficient repair at
B=2. Full-library mode makes the VLM cache, state embedding, explicit noise,
all five velocity fields, final normalized trajectory, and decoded robot
actions bitwise exact. It passes duplicate and unrelated composition at sizes
1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, and 64. Three distinct real
episode frames additionally pass duplicate and unrelated B=2/B=4 checks.

No new kernel was needed: Xiaomi exercises the existing reusable invariant
rank-3 linear and fixed-tree mean implementations.

## Performance

At the actual `[B,259,9728]` Qwen MLP input, invariant linear is 2.99x, 2.12x,
1.70x, and 1.40x stock latency at B=1, 2, 8, and 64. Absolute latency is
0.120-2.285 ms. At the actual `[B,12,1024]` FP32 RMS reduction, invariant mean
is 2.92-4.31x stock but remains approximately 0.027 ms at every measured size.
The complete measurements and memory counters are in `benchmarks.json`.

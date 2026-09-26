# RDT-1B official PyTorch investigation

## Scope and inference fidelity

This experiment uses the authors' official PyTorch repository at commit
`cd79363`, official fine-tuned ManiSkill checkpoint at revision `9622afab`,
and official SigLIP SO400M tower at revision `9fdffc58`. The BF16 policy has
1,228,319,872 learned elements: a 2,048-wide, 28-block Robotics Diffusion
Transformer with 32 attention heads, a 64-step action horizon, and a unified
128-dimensional state/action space. SigLIP adds 428,225,600 learned elements.

The released ManiSkill wrapper has six image slots (two history positions by
three cameras), of which the current exterior image is populated and five use
the processor-mean background. It selects the official unified-state indices
0-6 and 10, runs five DPM-Solver steps, selects the same eight dimensions from
the 64x128 prediction, and applies the released action bounds. The adapter
preserves all of those operations. It generalizes the upstream B=1 wrapper to
request batches and makes its initial BF16 diffusion noise an explicit
per-sample input.

With CUDA seed 73001, that controlled sampler is bitwise identical to the
unchanged official `RDTRunner.predict_action` output at B=1, including the
final float32 64x8 denormalized trajectory. The checkpoint also publishes the
exact cached PickCube-v1 T5 embedding used by official evaluation; the adapter
loads it directly rather than loading a redundant T5 model.

## Inputs and determinism

The target has normalized zero proprioception (the midpoint of every released
ManiSkill state range), a real public LIBERO image in the current exterior
slot, the official PickCube-v1 embedding, control frequency 25, and fixed
per-sample noise. Companion samples use other real LIBERO frames, deterministic
valid normalized proprioception, and their own fixed noise. These are
off-domain numerical fixtures, not a robot-task evaluation.

Each request's six images are processed by SigLIP with the same B=6 layout as
the official B=1 wrapper before policy batching. A separate check also flattens
two requests into B=12 images: both duplicate and unrelated row-zero vision
tokens happen to be stock-exact, and remain exact in invariant mode. Repeated
complete policy inference is bitwise exact.

## Baseline failure and first divergence

Stock inference changes the final action for every non-unit batch at B=`2,3,4,
5,7,8,9,15,16,17,31,32,33,64` in both duplicate and unrelated composition.
At B=2, 21 of 512 float32 action values differ, with maximum absolute change
0.0188322.

The first stock divergence occurs on the first diffusion evaluation in RDT
block 1, the first block conditioned on the 4,374 image tokens. Exact BF16
query and key/value projections enter `aten::scaled_dot_product_attention`:

```text
query:  [1, 32, 67, 64]
key:    [1, 32, 4374, 64]
value:  [1, 32, 4374, 64]
output before projection: [1, 67, 2048]
differences at B=2: 11,067
maximum absolute difference: 0.0625
```

The stock fused CUDA SDPA selects arithmetic based on the overall batch
geometry. The library's blockwise attention fixes the query/key and
attention/value reduction schedules independently of B. An SDPA-only override
makes the complete B=2 trajectory exact.

At B=64, SDPA-only reveals a second independent threshold in
`lang_adaptor.0`. Its exact `[1,20,4096]` input produces 32 differing BF16
values in the `[1,20,2048]` output (maximum 0.0078125) because the biased
matrix multiply changes from M=20 to M=1280. Adding only invariant `addmm` to
invariant SDPA makes both duplicate and unrelated B=64 actions exact. Thus no
additional operator family is necessary for this pinned policy.

## Final result

With the public invariant context, the complete float32 64x8 robot trajectory
is bitwise exact for duplicate and unrelated batches at B=`1,2,3,4,5,7,8,9,
15,16,17,31,32,33,64`. Three independent real image/state/noise targets pass
all duplicate and unrelated B=2/B=4 checks; stock fails all twelve.

## Operator performance

On the recorded H100, invariant SDPA for the actual block-1 cross-attention
shape is 5.83x, 5.41x, and 4.90x stock latency at B=1, 2, and 8. It uses more
temporary memory because it materializes fixed-schedule blockwise partials.
The language-adapter addmm is 3.23x stock latency at B=1, 0.65x at B=2, and
2.32x at B=64; its incremental allocation is lower at all three sizes.

## Evidence boundary

This proves numerical batch invariance for the pinned official PyTorch
ManiSkill policy, cached task embedding, five-step sampler, specified real-image
fixtures, explicit per-sample noise, and final unnormalized trajectory through
B=64 on the recorded H100. It does not claim closed-loop ManiSkill success,
semantic suitability of LIBERO images for PickCube, training invariance, or
other RDT checkpoints.

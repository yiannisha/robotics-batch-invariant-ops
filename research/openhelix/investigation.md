# OpenHelix official PyTorch investigation

## Scope and fidelity

This experiment uses the official OpenHelix PyTorch repository at commit
`d88867f` and the authors' complete `prompt_tuning_aux` checkpoint at revision
`1a6fabb`. The inference path contains a roughly 7.1B-parameter BF16
LLaVA/LISA planner and the released 48.0M-parameter FP32 DiffuserActor policy.
All planner shards and the policy state load strictly; construction-only LLaVA
and CLIP metadata contributes no learned weight.

Three real public CALVIN ABC transitions supply static/gripper RGB, depth, and
proprioception. Point clouds are reconstructed from the released calibration.
The fixed prompt is `push the sliding door to the right side`. The complete
path produces a 512D action-token embedding, encodes both camera views and
point clouds, runs 25 DDPM steps, and converts the resulting relative pose to
a 19x7 absolute robot trajectory.

## Determinism and official equivalence

Initial 19x9 diffusion noise and every position/rotation scheduler variance
draw are explicit per-sample inputs. With CUDA seed 91000, this controlled path
is bitwise identical to the unchanged released
`DiffuserActorACTS.conditional_sample` result: all 190 FP32 values match after
25 steps. Repeated complete B=1 inference is also bitwise exact.

## Baseline and first divergence

Stock inference changes the final action for every non-unit B in
`2,3,4,5,7,8,9,15,16,17,31,32,33,64`, in duplicate and unrelated
composition. At B=2, 114 of 133 values differ, with maximum absolute error
0.000860393. These are the six continuous action dimensions at all 19 steps;
thresholded gripper values remain equal.

The CLIP patch embedding and layer-0 Q/K/V projections are exact. The first
divergence is the layer-0 attention-score `aten::bmm`:

```text
logical Q/K/V: [1, 16, 257, 64] BF16
attention scores [16, 257, 257]:
    6 differences, maximum 0.015625
attention module output [1, 257, 1024]:
    183 differences, maximum 0.00048828125
```

The batch dimension is folded into Bx16 independent GEMMs. Stock CUDA chooses
arithmetic based on that leading count; the existing fixed-schedule BMM repair
keeps each head's reduction decomposition unchanged.

After all existing generic operator repairs, the full CLIP output and the
multimodal projector input `[1,256,1024]` are exact. The next first divergence
is direct rank-3 `aten::linear`: 268,177 of the `[1,256,4096]` BF16 projector
outputs differ at B=2, with maximum difference 0.5. This call does not
decompose through the already overridden `aten::mm` or `aten::addmm` dispatcher
entries. The new generic linear implementation flattens arbitrary leading
dimensions, uses the fixed-schedule persistent matrix multiplication, adds the
bias there, and restores the original leading shape.

## Final result

With the new linear override added to the existing operator set, the complete
19x7 FP32 absolute action is bitwise exact at every B=`1,2,3,4,5,7,8,9,15,16,
17,31,32,33,64` in both composition modes. All twelve checks spanning three
real CALVIN targets, duplicate/unrelated composition, and B=2/B=4 also pass;
stock fails all twelve.

## Performance

At the actual `[B,256,1024] x [4096,1024]` BF16 projector boundary, invariant
linear is 2.96x, 2.51x, and 1.08x stock latency at B=1, 2, and 8. At B=8 it
runs in 0.0656 ms versus 0.0610 ms stock. Incremental output/workspace memory
is 5.18/6.23 MB at B=1 and 37.75/33.55 MB at B=8 for invariant/stock.

## Evidence boundary

This proves numerical batch invariance for the pinned official PyTorch
checkpoint, real CALVIN inputs, fixed prompt, full planner and policy encoder,
all 25 stochastic denoising steps, and final actions through B=64 on the
recorded H100. It does not claim closed-loop task success, other OpenHelix
checkpoints, training invariance, or batch invariance for source paths outside
the tested CALVIN policy.

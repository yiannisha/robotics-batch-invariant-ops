# CogACT-Small official PyTorch investigation

## Scope and fidelity

This experiment uses Microsoft's official PyTorch repository at commit
`b174a1b`, its authors' pinned-era OpenVLA dependency at commit `5603207`, and
the official `CogACT/CogACT-Small` checkpoint at revision `ca302a3`. The model
has 7,553,713,991 learned parameters: a fused DINOv2-L/SigLIP-SO400M vision
tower, a three-layer multimodal projector, a 32-layer Llama-2 7B cognition
backbone, and a six-layer 384-wide DiT-S action head.

The target exactly follows the official Azure example: bundled
`scripts/aml/test_image.png`, prompt `move sponge near apple`, the released
RT-1 `fractal20220817_data` q01/q99 statistics, CFG scale 1.5, and ten DDIM
steps. The documented deployment precision is used: BF16 VLM and FP32 action
head. The initial FP32 16x7 diffusion noise is an explicit per-sample input.

With CUDA seed 74000, the controlled progressive DDIM loop is bitwise identical
to the authors' unchanged `predict_action_batch` method at B=1, including the
complete float64 unnormalized 16x7 output. The checkpoint strictly supplies all
learned tensors. The public Llama mirror supplies only construction metadata and
tokenizer files because the original metadata repository is gated; no mirror
weight is loaded.

## Determinism and inputs

Repeated B=1 inference is bitwise exact. Duplicate batches repeat the official
target, including its fixed noise. Unrelated batches retain that target in row
zero and use other real public LIBERO images plus independently fixed noise.
The prompt is held fixed to isolate numerical batch effects. The two additional
regression targets are LIBERO frames. They are deliberately only numerical
fixtures; no CogACT task-success claim is made for off-domain images.

## Baseline and first divergence

Stock inference changes the final action for every non-unit B in
`2,3,4,5,7,8,9,15,16,17,31,32,33,64`, for both duplicate and unrelated
composition. At B=2, 96 of 112 output values differ (all six continuous
dimensions at all 16 steps; the thresholded gripper remains unchanged), with
maximum absolute difference 0.002151882.

The DINO and SigLIP patch convolutions are exact. The first divergence is
DINO block 0 attention:

```text
operator: aten::scaled_dot_product_attention
Q/K/V: [1, 16, 261, 64] BF16
attention input: exact
QKV projection [1, 261, 3072]: exact
pre-projection attention output [1, 261, 1024]:
    7,590 differences, maximum 0.015625
```

The fused CUDA SDPA changes its reduction schedule with request B. The generic
fixed-block SDPA replacement makes the complete DINO and SigLIP towers, the
multimodal projector, and Llama layer-0 attention inputs/outputs exact.

SDPA alone reveals the next independent boundary in Llama layer 0. Its exact
`[1,278,11008]` SwiGLU activation enters the bias-free MLP down projection;
`aten::mm` produces 4,156 differing BF16 `[1,278,4096]` values, maximum
0.001953125. The generic invariant MM replacement fixes this boundary. Adding
BMM after invariant SDPA and GEMM does not change the already exact result.

## Final result

Invariant SDPA plus MM/addmm is sufficient for both duplicate and unrelated
B=64. The public full context makes the complete float64 16x7 robot trajectory
bitwise exact at every tested B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64` in
both compositions. Three real image/noise targets pass all twelve B=2/B=4
checks; stock fails all twelve.

## Performance

On the recorded H100, invariant SDPA at the actual DINO block-0 shape is 3.17x,
8.87x, and 5.42x stock latency at B=1, 2, and 8. The invariant Llama layer-0
down-projection MM is 2.30x, 2.16x, and 1.36x stock latency. Both paths preserve
the exact target across benchmark batch shapes; the persistent MM uses the same
incremental output allocation as stock for this case.

## Evidence boundary

This proves numerical batch invariance for the pinned official CogACT-Small
PyTorch checkpoint, official published example, fixed prompt/configuration,
explicit noise, complete cognition path, all ten DDIM steps, and final
unnormalized action through B=64 on the recorded H100. It does not claim
closed-loop success, CogACT-Base/Large invariance, training invariance, or
semantic suitability of the LIBERO companion images.

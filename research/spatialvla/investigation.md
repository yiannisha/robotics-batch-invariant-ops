# SpatialVLA 4B PyTorch investigation

## Scope

This experiment uses the authors' official PyTorch repository at commit
`18fd74b`, official `IPEC-COMMUNITY/spatialvla-4b-224-pt` checkpoint at
revision `886c4557`, and the checkpoint's released Transformers model and
processor code. The model has 4,027,854,731 parameter elements and combines a
27-layer SigLIP image tower, a 24-layer BEiT/ZoeDepth geometry tower, 3-D
position embeddings, and a Gemma 2 autoregressive policy. The two BF16
safetensors shards are loaded directly without conversion.

The primary target exactly follows the published quickstart: `example.png`,
the prompt `What action should the robot take to pick the cup?`, greedy
generation with the released 256-token cap, and Bridge `bridge_orig/1.0.0`
action statistics. The model emits twelve spatial action tokens plus EOS,
which the official tokenizer decodes into a float64 4x7 action chunk. The
adapter fixes the upstream decoder's row-zero assumption by invoking that
unchanged decoder once per generated row.

## Determinism and inputs

The model runs in evaluation and inference mode with fixed Python, NumPy, CPU,
and CUDA seeds. Generation is greedy and has no sampled latent. The target's
processor output is identical at B=1 and B=2. Repeated B=1 inference is
bitwise exact.

Duplicate batches repeat the published target. Unrelated batches retain that
target in row zero and use real public LIBERO observations as companions; the
companion prompt is deliberately held fixed so language and generation
configuration cannot confound the numerical check. The additional-target
regression also checks two LIBERO observations as row-zero targets at B=2 and
B=4.

## Baseline failure and staged repair

Stock inference changes the final action at B=2, 3, 4, 5, 7, 8, 9, 15, 31,
32, 33, and 64 for both duplicate and unrelated compositions. B=16 and B=17
happen to select the B=1 action tokens, which is not a batch-invariance
guarantee. At B=2, six of 28 decoded action scalars differ and the maximum
absolute action change is 0.0276571.

The first traced stock divergence is ZoeDepth's final reassembly downsampler:

```text
module:    vision_zoe_model.neck.reassemble_stage.layers.3.resize
operator:  aten::convolution
input:     [1, 1024, 24, 24] BF16, exact
parameters: kernel 3x3, stride 2, padding 1
output:    [1, 1024, 12, 12]
differences: 250 elements, maximum 0.0625
```

The staged B=1/B=2 score experiment then enables operator families
cumulatively. Convolution alone leaves all 13 generation score calls
different. Adding fixed-tree `mean` still leaves all 13 different; adding
invariant `mm` makes the prefill score exact but leaves 12 cached-decode calls
different. Adding invariant `bmm` makes every one of the 13 full 265,347-way
score tensors, generated tokens, and decoded actions exact. Removal ablations
from full mode independently make convolution and MM fail all score calls,
BMM fail 11 cached-decode calls, and mean fail 12 calls. With BMM removed, an
exact layer-2 cached-decode attention input produces 11 differing BF16 output
elements (maximum 0.0009765625), confirming the rank-3/4 matmul boundary.

SpatialVLA also exercises direct non-last-dimension `torch.softmax` in
ZoeDepth. The generic mode previously covered softmax only inside its SDPA
replacement. This investigation therefore adds an `aten::_softmax` override
and generalizes the fixed per-row kernel to arbitrary dimensions through a
canonical permutation. Its removal does not change this particular target
once all other invariant operators are active, so it is recorded as a real
dispatch-coverage repair rather than claimed as a necessary target-specific
ablation.

The released geometry tower also contains 4x4/stride-4 and 2x2/stride-2
ConvTranspose2D layers. Generic invariant mode previously rejected those
calls. A reusable non-overlapping grouped-BMM ConvTranspose2D implementation
now covers both layers. Stock happens to be exact for the benchmarked released
shape, but the new implementation is required for the generic dispatcher to
run the complete model and provides a batch-independent decomposition.

## Final result

With invariant mode enabled, the complete float64 4x7 robot action is bitwise
exact for duplicate and unrelated batches at B=`1,2,3,4,5,7,8,9,15,16,17,31,
32,33,64`. The full generation-score trace is exact at B=2. All twelve checks
covering the published target and two additional real targets at B=2/B=4 also
pass in repaired mode; stock fails all four checks for the published target.

## ConvTranspose2D performance

The benchmark loads the actual released
`vision_zoe_model.neck.reassemble_stage.layers.0.resize` BF16 weight with input
shape `[B,256,24,24]`, weight `[256,256,4,4]`, and stride four. Relative to
stock cuDNN, the invariant grouped-BMM path is 1.80x slower at B=1, 1.21x
slower at B=2, and 0.93x stock latency at B=8. Its measured incremental memory
is lower at all three sizes. Both stock and invariant paths happen to be exact
across these benchmark batch sizes.

## Evidence boundary

This proves numerical batch invariance for the pinned official PyTorch
224-pixel pretrained policy, released example and processor, greedy spatial
token sequence, and complete unnormalized action through B=64 on the recorded
H100. It does not claim robot task success, closed-loop behavior, training
invariance, other SpatialVLA checkpoints, or semantic independence of
ZoeDepth's released batch-wide domain-selection branch for adversarial mixed
NYU/KITTI batches.

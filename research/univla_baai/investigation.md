# BAAI UniVLA PyTorch investigation

## Scope

This experiment uses the authors' official PyTorch repository at commit
`a91ee5`, the official `UNIVLA_LIBERO_IMG_BS192_8K` checkpoint at revision
`df6c94ff`, and the official Emu3 vision tokenizer at revision `c81f916a`.
The policy contains 8,492,011,520 learned elements in 291 tensors and is loaded
directly from its four BF16 safetensors shards without conversion. Despite the
released class name `Emu3MoE`, this checkpoint has `action_experts=false` and
is a dense 32-layer autoregressive policy.

Inference follows the released LIBERO wrapper: primary and wrist RGB frames
are resized to 200x200, converted to 25x25 VQ grids, formatted in Emu3 `VLA`
mode, greedily decoded under the official FAST action-token constraint, decoded
to a 10x7 action chunk, unnormalized with the released LIBERO constants, and
mapped to binary gripper commands. Inputs are real observations from the
public LeRobot LIBERO episode-zero fixture.

## Preprocessing and determinism controls

The released wrapper sets `image_processor.min_pixels = 80 * 80` after loading
the vision tokenizer. This assignment is essential: without it, the processor
expands each 200px camera image to 512px, produces two 64x64 VQ grids, and
creates an 8,235-token prompt for a model configured for 1,600 tokens. The
adapter preserves the released assignment and obtains a 1,293-token two-camera
prompt.

The policy uses greedy decoding (`do_sample=false`) and has no stochastic
inference input. Every image is VQ-encoded individually, matching online B=1
preprocessing, before request batches are composed. Finished rows are trimmed
at their own first end-of-action token so a longer companion cannot add pad
tokens to the target FAST decode. Repeated B=1 inference is bitwise exact.

## Baseline and repaired result

The released public robot action happens to be bitwise exact for every
duplicate and unrelated composition at B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33`.
B=64 exceeds memory because the official generator materializes the full
184,622-way prefill logits. Three target frames also pass all duplicate and
unrelated B=2/B=4 action checks.

This action-level result hides a real numerical boundary. At B=2, every greedy
token remains unchanged, but 43 allowed action-token scores across the decode
differ from B=1, with maximum absolute difference 0.0625. The trace over all
184,622 logits locates the first divergence at cached decode call 1:

```text
transformer/final-norm input: exact
lm_head input:               [B, 1, 4096] BF16, exact
lm_head weight:              [184622, 4096] BF16
operator:                    aten::mm
B=2 target:                  151 / 184622 logits differ
maximum error:               0.0625
```

The 1,293-token prefill is exact through the LM head. Every captured transformer
stage on the first cached decode step is also exact; only the wide bias-free LM
head changes when its flattened matrix height changes from one to two.
Isolated operator staging confirms that an invariant `addmm` override leaves
the 151 differences, while the existing reusable invariant `mm` makes all
184,622 logits exact. With full invariant mode, every captured prefill/decode
intermediate, every constrained score, every generated token, and every final
action is exact for duplicate and unrelated batches through B=33.

No UniVLA-specific numerical kernel was required. This model validates the
generic persistent MM replacement on an autoregressive vocabulary projection
that is much wider than the robotics heads tested previously.

## Performance

Using the checkpoint's actual 4,096x184,622 LM-head weight, the invariant MM is
1.47x, 1.38x, and 1.40x slower than stock at B=1, B=2, and B=8, respectively.
Both paths have identical incremental output allocation. Measurements use 50
warmups and 1,000 repetitions; complete latency, throughput, shape, and memory
records are in `benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned official PyTorch LIBERO
image policy, official VQ and FAST tokenizers, BF16/FlashAttention inference,
real two-camera inputs, greedy token sequence, and complete unnormalized 10x7
robot action through B=33. It does not claim task success, closed-loop behavior,
training invariance, the separately released video-SFT policy, other UniVLA
checkpoints, or B=64 support.

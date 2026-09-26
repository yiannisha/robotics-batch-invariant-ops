# DexVLA PyTorch investigation

## Scope

This experiment uses the authors' official PyTorch repository at commit
`fc21a822`, the closest public complete checkpoint (`kuromivv/DexVLA`, revision
`f99bdd40`), and the authors' public example episode. The checkpoint contains
3,412,385,806 learned elements across 1,081 tensors and loads directly into the
official Qwen2-VLA plus 32-layer ScaleDP classes without conversion.

Inputs are real RGB frames from three cameras, normalized 14D robot state, and
the episode task text. Official preprocessing, greedy language reasoning,
FiLM conditioning, ten DDIM denoising steps, and released min/max action
unnormalization are executed. The robot output is the complete 50x14 action
chunk.

The released ScaleDP inference helper hard-codes `B = 1`. The adapter preserves
its operations while deriving B from inputs and making initial diffusion noise
explicit per example. It also trims every generated row at its own first EOS
before FiLM; otherwise Hugging Face generation adds companion-dependent EOS
steps until the longest batch member finishes. With identical explicit noise,
the generalized B=1 path is bitwise identical to the official `evaluate()`
method for all 700 normalized action values and its generated reasoning text.

## Nondeterminism controls

The complete model is in evaluation mode; language generation is greedy; and
Python, NumPy, CPU, and CUDA RNGs are reset before each run. Each episode frame
has fixed GPU-generated diffusion noise stored as an explicit CPU fixture.
Repeated B=1 inference is bitwise exact. The official README's float32 DDIM
schedule fix is applied after model loading to prevent its documented
bfloat16-underflow NaNs.

## Baseline and repaired result

Stock inference fails every completed duplicate and unrelated composition
above B=1 (B=2,3,4,5), then its quadratic packed-image attention mask OOMs at
B=7. At B=2, 323 of 700 unnormalized action values differ with maximum
absolute difference `0.05386042594909668`.

With invariant operators and the packed-attention integration described below,
the complete action chunk is bitwise exact at
B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33` for both composition modes. B=64 is the
first OOM, so the maximum completed batch is 33. Independent target frames 0,
20, and 40 fail all 12 stock B=2/B=4 checks (maximum difference
`0.09425568580627441`); all 12 repaired checks are exact, including companions
whose generated reasoning has a different length.

## First divergence: Qwen2-VL vision SDPA

Pixel inputs, 3D patch convolution, and block-0 QKV projection are exact. The
first mismatch is the first vision block's SDPA. Official Qwen2-VL packs all
three images per observation along the token dimension:

```text
B=1 packed tokens: 4,692 = 3 x 1,564
B=2 packed tokens: 9,384 = 6 x 1,564
heads / head dim:  16 / 80
block output:      [packed_tokens, 1,280] bfloat16
operator:          aten::scaled_dot_product_attention
```

The first sample differs in 1,265 block-output values with maximum absolute
difference `0.001953125`. Convolution, MM, and addmm repairs do not move this
boundary.

The generic dispatcher cannot recover logical image boundaries from a dense
block-diagonal attention mask. In invariant mode only, the adapter forwards
the `cu_seqlens` already supplied by official Qwen2-VL to the library's generic
batch-invariant varlen SDPA. Stock mode retains the original method. This is
required beyond B=3: treating the entire packed token axis as one reduction
can still depend on the number of image segments and retains quadratic
cross-image memory. The varlen path evaluates true segments independently and
raises the practical limit from stock B=5 to fixed B=33.

## Second divergence: ScaleDP conditioning addmm

With attention repaired, the first remaining mismatch is ScaleDP's first
conditioning linear:

```text
input:     [B, 1,550] (1,536D FiLM condition + 14D state)
weight:    [1,550, 1,024]
operator:  aten::addmm
```

Its `[B,1024]` output differs in 507 bfloat16 values with maximum absolute
difference `0.0078125`. Adding invariant `mm` does not repair it; invariant
`addmm` does. SDPA plus addmm makes every captured boundary, all ten diffusion
steps, and the final action exact. Convolution and BMM are not independently
required for this checkpoint.

## Performance

At the actual three-image-per-observation attention geometry, invariant SDPA
is 5.11x, 5.39x, and 6.08x slower than stock at robot B=1, B=2, and B=8. Its
eager fixed reductions use about 495 MB, 988 MB, and 3.95 GB of incremental
memory. The invariant ScaleDP conditioning addmm is 3.72x, 2.99x, and 2.85x
slower. Full measurements are in `benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned official PyTorch classes,
closest public complete checkpoint, real three-camera episode inputs, generated
reasoning, FiLM, ten-step ScaleDP sampler, and 50x14 action output through B=33.
It does not claim official provenance for the complete checkpoint, task
success, closed-loop performance, training invariance, other checkpoints, or
B=64 execution.

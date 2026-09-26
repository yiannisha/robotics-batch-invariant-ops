# Cosmos 3 Edge Policy DROID PyTorch investigation

## Scope

This experiment uses NVIDIA's official PyTorch Cosmos framework at commit
`cf5d68c0` and the public `nvidia/Cosmos3-Edge-Policy-DROID` safetensors
snapshot at commit `a7c7288f`. Real three-camera observations, joint state,
gripper state, and task text come from the pinned public `Cosmos3-DROID`
dataset. No JAX source, weight, or conversion is involved.

The official Edge serving configuration uses JSON prompts and the inclusive
guidance interval `[960, 1001]`. Four UniPC steps jointly generate a 32x8
action trajectory and a 48x9x33x40 future-video latent. A separate check passes
the latent through the released Wan2.2 VAE to produce 33x528x640 RGB frames.

## Inputs and nondeterminism controls

Each real DROID observation has an explicit diffusion seed, so the target's
noise is unchanged by batch size and companion selection. The official sample
transform, tokenizer, action decoder, and gripper convention are used.
Repeated B=1 inference is bitwise exact.

## Baseline result

Stock inference fails every duplicate and unrelated composition above B=1 for
all tested sizes through B=64. At B=2, 215 of 256 final float32 action values
differ with maximum absolute difference `0.034447193`, and 505,738 future
latent values differ with maximum difference `1.5768205`. At B=64 the maxima
are `0.016654491` and `0.69438481`, respectively.

The decoded-video baseline also fails. At B=2, 13,356,836 values differ in the
uint8 33x528x640 RGB output.

## Generation divergence and repair

The generation trace finds exact Q, K, and V values for the shared 158 real
tokens, followed by five differing bfloat16 attention outputs with maximum
difference `0.00006103515625`:

```text
B=1: dense cuDNN attention
      Q [1, 159, 16, 128], K/V [1, 159, 8, 128]

B=2: packed variable-length CUTLASS attention
      Q/K/V total sequence 317, target real sequence length 158
```

The alignment token is excluded by the packed offsets. The stock branch is
repeated in cached text-KV attention during later flow evaluations.

The reusable packed variable-length attention operator introduced for Cosmos
Nano transfers without architecture-specific numerical changes. The adapter
uses identical varlen geometry at B=1 and B>1, routes ordinary and cached
text-KV attention through the fixed-reduction operator, and supplies B=1 cache
reorder metadata. In repaired mode, the first attention output, all eight
conditional/unconditional velocity evaluations, the final action, and the
future latent are exact.

## Decoder divergence and repair

Exact generated latents initially still produced 15,927 differing uint8 video
pixels at B=2. An exact leaf-module trace found 687 matching calls before the
first mismatch in the Wan decoder's channel-wise `RMS_norm`, at shape
`[1,1024,4,132,160]`. `torch.nn.functional.normalize` dispatches this reduction
to `aten::linalg_vector_norm`; stock CUDA selected batch-shaped reduction
behavior even though each sample was identical.

The reusable `linalg_vector_norm_batch_invariant` implementation assigns a
fixed Triton reduction tree to every output vector. It supports the real CUDA
float types, Euclidean norms, arbitrary single dimensions without a layout
copy, multi-dimension reductions through a canonical layout, `keepdim`, and
the public `torch.linalg.vector_norm` dispatch path. After this repair, the
action, latent, and every decoded uint8 video value are exact at B=2.

## End-to-end result

With `set_batch_invariant_mode()`, every action and future-latent value is
bitwise identical for B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64` under duplicate
and unrelated composition. The decoded video is exact for both compositions
at B=2. Target frames 0, 20, and 40 independently pass duplicate and unrelated
B=2/B=4 checks. A decoded-video regression on the previously completed Nano
policy also remains exact after enabling the generic norm override.

## Performance

At Edge's actual 158-token causal attention shape, invariant attention is
1.53x, 1.75x, and 2.13x stock latency at B=1, B=2, and B=8. At the dominant
3093-query/3251-KV full-attention shape, ratios are 6.78x, 5.57x, and 6.46x.

At the decoder norm shape, the invariant vector norm is 1.51x, 1.87x, and
2.06x stock latency. Its incremental allocation is about 0.34 MB and 0.68 MB
at B=1/B=2, versus about 173 MB for stock at those sizes. Full timings,
throughput, environment, and peak-memory measurements are in
`benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned public Edge DROID
policy, real observations, official joint action/world generation, four
released flow steps, and the released VAE decoder in the recorded environment.
It does not claim task success, training invariance, closed-loop performance,
or support for unreleased/multi-GPU Cosmos variants.

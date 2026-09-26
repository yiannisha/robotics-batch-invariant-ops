# Cosmos 3 Nano Policy DROID PyTorch investigation

## Scope

This experiment uses NVIDIA's official PyTorch Cosmos framework at commit
`cf5d68c0` and the public `nvidia/Cosmos3-Nano-Policy-DROID` safetensors
snapshot at commit `805c0d6d`. Real three-camera observations, joint state,
gripper state, and task text come from the pinned public `Cosmos3-DROID`
dataset. No JAX source, weight, or conversion is involved.

The official Robolab service transforms the observations, runs four UniPC
flow steps with guidance 3.0 and shift 5.0, and jointly generates a 32x8
action trajectory and a 48x9x33x40 future-video latent. A separate check passes
the latent through the released Wan2.2 VAE to produce 33x528x640 RGB frames.

## Inputs and nondeterminism controls

The adapter assigns each DROID frame an explicit diffusion seed. The target's
seed is therefore unchanged by batch size or companion selection. The official
sample transform, tokenizer, action decoder, and gripper convention are used.
Repeated B=1 inference is bitwise exact.

Direct diagnostics showed that the VAE conditioning latent, raw and tokenized
actions, packed-sequence plan, caption tokens, complete initial diffusion
noise, conditioning reference, and mask are all exact between B=1 and B=2.
The first divergence occurs inside the first denoising-network call.

## Baseline result

Stock inference fails every duplicate and unrelated composition above B=1 for
all tested sizes through B=64. At B=2, 255 of 256 final float32 action values
differ with maximum absolute difference `0.113338232`, while 506,866 future
latent values differ with maximum difference `2.8412516`. At B=64 the maxima
are `0.1680040` and `2.9505436`, respectively.

The released VAE decode is also affected. At B=2, stock inference changes
25,410,747 values in the uint8 33x528x640 RGB video.

## Divergence chain

The stock trace first observes ordinary projection differences: the shared
real-token query is exact, while 131 key and 204 value elements differ. The
library's existing invariant matrix operators make Q, K, and V exact.

With those generic operators repaired but the Cosmos attention dispatch left
stock, the next boundary is the first causal attention result. Its exact Q,
K, and V produce 14 differing bfloat16 outputs with maximum difference
`0.000244140625`:

```text
B=1: dense cuDNN flash attention
      Q [1, 96, 32, 128], K/V [1, 96, 8, 128]

B=2: packed variable-length CUTLASS attention
      Q [1, 191, 32, 128], K/V [1, 191, 8, 128]
      first real sequence length 95
```

The one-token shape difference is trailing alignment padding, excluded by the
varlen offsets; the trace compares the shared 95 real tokens. CUDA kernel names
and exact hashes are in `first_divergence.json`.

Cosmos repeats this batch-shaped branch after the first flow evaluation. Its
text-KV cache uses dense attention for a single sample but constructs reordered
per-sample KV and varlen metadata when multiple samples are present. Fixing
only the first attention path is therefore insufficient.

## Repairs

The new reusable
`varlen_scaled_dot_product_attention_batch_invariant` operator accepts Cosmos'
packed `[1, total_tokens, heads, dim]` convention and cumulative offsets. It
evaluates each logical sequence with the library's fixed-schedule BMM and
row-wise softmax. Equal geometries share bounded launches, but every sequence,
head, and row keeps an independent reduction tree. The 512 MiB score bound
prevents transient memory from scaling with the whole packed batch.

The adapter makes both B=1 and B>1 use varlen geometry while invariant mode is
active, routes standard and cached text-KV attention through the generic
operator, and supplies identity reorder metadata for the one-sample cache. The
baseline remains the unmodified upstream branches.

Decoded-video validation found one later implementation issue in an existing
generic operator. Wan2.2's largest Conv3D im2col matrix has more than 2^31
elements (`[1, 84480, 27648]` at the failing layer). Persistent BMM pointer
offsets overflowed 32 bits. BMM now applies the same conditional 64-bit indexing
guards already used by persistent MM. This is a generic large-tensor repair,
not a Cosmos numerical special case.

## Iterative propagation

With only generic projection operators fixed, the first conditional velocity
already differs in 404,695 values with maximum difference `0.12890625`. By the
last conditional/unconditional evaluations, roughly 490,000 velocity values
differ with maxima above 3.5. The final action differs in 255 values and the
future latent in 506,849 values. The trace records all eight conditional and
unconditional network evaluations across four flow steps.

With the complete repair, first-attention Q/K/V/output, every recorded flow
input, every velocity, the final action, and the future latent are bitwise
exact.

## End-to-end result

With `set_batch_invariant_mode()`, every action and future-latent value is
bitwise identical for B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64` under duplicate
and unrelated composition. The decoded uint8 video is also exact for duplicate
and unrelated B=2. No later divergence was found.

After the large-index BMM repair, `large_batch_recheck.json` reruns duplicate
and unrelated B=64 with the final kernel revision; both outputs remain exact.

The multi-input artifact additionally checks target frames 0, 20, and 40 at
B=2 and B=4.

## Performance

At the actual 96-token causal-attention shape, the invariant operator is
1.34x, 1.75x, and 2.05x stock latency at B=1, B=2, and B=8. At the dominant
3094-query/3188-KV full-attention shape, the ratios are 5.22x, 5.61x, and
6.75x. The eager implementation materializes scores and therefore uses more
temporary memory than fused attention; launches are bounded so the complete
model still fits B=64 on the recorded H100. Full timings and peak incremental
memory are in `benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned public DROID policy,
real observations, official joint action/world generation, four released flow
steps, and the released VAE decoder in the recorded environment. It does not
claim task success, training invariance, closed-loop performance, or support
for unreleased/multi-GPU Cosmos variants.

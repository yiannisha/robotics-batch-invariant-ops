# Batch Invariant Ops

A companion library release to https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/. This library contains some batch-invariant kernels as well as an example of achieving deterministic vLLM inference.

## Overview

This library primarily leverages torch.Library to sub out existing PyTorch kernels with "batch-invariant" ones. This allows many existing PyTorch models to use the batch-invariant ops with low overhead and non-intrusive code changes.

Batch invariance means that the output for a fixed sample is bitwise identical
whether it is evaluated alone or beside other samples. This is stricter than
ordinary run-to-run determinism: a deterministic CUDA library can consistently
choose one reduction schedule for B=1 and a different, equally deterministic
schedule for B=32. Floating-point reduction then produces different bits.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode

# Enable batch-invariant mode
with set_batch_invariant_mode():
    # Your inference code here
    model = YourModel()
    output = model(input_tensor)
```

## Testing Batch-Invariance

The following example shows how batch size can affect results in standard PyTorch:

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode
torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)
    
    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]
    
    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

```

## Deterministic Inference in vLLM
`deterministic_vllm_inference.py` shows an proof of concept of validating that vLLM can be made deterministic with a minor upstream PR to use this library. Without the upstream PR, we see that out of 1000 random length 100 completions we see 18 unique samples. After the upstream PR, there is only one unique sample.

## Supported Operations

### Matrix Operations
- `torch.mm()` - Matrix multiplication
- `torch.addmm()` - Matrix multiplication with bias addition
- `torch.bmm()` - Batch matrix multiplication, including rank-3 attention
  `torch.matmul()` calls that dispatch to `aten::bmm`

### Mixture-of-Experts Operations

- `deterministic_token_choice_moe()` - deterministic top-k SwiGLU routing for
  fused expert weights, using dense fixed-schedule expert BMMs, top-k gather,
  and fixed-order route accumulation instead of atomic packing/accumulation

### Attention Operations
- `torch.nn.functional.scaled_dot_product_attention()` - evaluation-time SDPA
  (`dropout_p=0`) using independent batch-invariant BMM and Triton softmax
  reductions; causal, boolean/additive mask, and grouped-query paths are
  supported
- `varlen_scaled_dot_product_attention_batch_invariant()` - packed
  variable-length attention with cumulative sequence offsets, GQA, and
  top-left/bottom-right causal alignment; equal geometries use memory-bounded
  launches while retaining independent per-sequence reductions

### Convolution Operations
- `torch.nn.functional.conv1d()` / `nn.Conv1d` - Regular 1-D convolution,
  including a fixed-order Triton depthwise path used by Qwen3.5
- `torch.nn.functional.conv2d()` / `nn.Conv2d` - Regular (non-transposed) 2-D
  convolution using batch-independent grouped im2col/BMM
- `torch.nn.functional.conv3d()` / `nn.Conv3d` - Regular 3-D convolution using
  batch-independent grouped im2col/BMM, including Qwen3.5-VL patch embedding
- `torch.nn.functional.conv_transpose3d()` / `nn.ConvTranspose3d` -
  Non-overlapping 3-D learned upsampling (`stride == kernel_size`) using a
  fixed-schedule grouped BMM, including UVA's temporal action upsampler

### Activation Functions
- `torch.log_softmax()` - Log-softmax activation
- `batch_invariant_ops.softmax()` - last-dimension softmax used by the SDPA
  replacement

### Reduction Operations
- `torch.mean()` - Mean computation along specified dimensions
- `torch.linalg.vector_norm(..., ord=2)` - Euclidean vector norms with a fixed
  reduction tree per output vector, including the `torch.nn.functional.normalize`
  path used by RMS normalization

The current kernels cover CUDA float32, float16, and bfloat16 paths exercised
by the tests. Conv1D/2D/3D cover regular non-transposed convolution, including
groups and dilation. ConvTranspose3D currently covers the non-overlapping,
zero-padding case. Unsupported operator overloads and dtypes are not claimed.

## Conv2d and attention BMM demonstration

On CUDA, run `python scripts/check_conv2d_bmm_batch_invariance.py`.  It
compares a sample evaluated alone with the same sample at the start of a larger
batch for a 14x14 patch convolution and an attention-shaped rank-3 matmul.  It
also profiles the latter to show why an `mm` override does not cover it:
rank-3 `torch.matmul` dispatches `aten::bmm`.

## Robotics model status

Only configurations that have been executed end to end are marked PASS.

| Model configuration | Precision | Batch sizes | Baseline | With this library |
|---|---:|---:|---:|---:|
| π0 pre-fix PyTorch reference, random weights and synthetic inputs | bfloat16 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| π0.5 `pi-zero-pytorch` reference, public `lerobot/pi05_base` weights and controlled synthetic inputs | float32 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| π0-FAST LeRobot PyTorch, public `lerobot/pi0fast-libero` weights and real LIBERO inputs | mixed float32/bfloat16 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| InternVLA-A1.5 official PyTorch standard action backend, public LIBERO weights and real inputs | bfloat16 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| InternVLA-A1.5 official PyTorch optimized action-only backend, public LIBERO weights and real inputs | mixed float32/bfloat16 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| MolmoAct2 official PyTorch, public `allenai/MolmoAct2-LIBERO` weights and real inputs | bfloat16 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| GR00T N1.7 official NVIDIA PyTorch, public `libero_10` weights and real inputs | bfloat16 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| Cosmos 3 Nano Policy official NVIDIA PyTorch, public DROID weights and real inputs | bfloat16 network, float32 outputs | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| Cosmos 3 Edge Policy official NVIDIA PyTorch, public DROID weights and real inputs | bfloat16 network, float32 outputs | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| X-VLA maintained LeRobot PyTorch, public `lerobot/xvla-libero` weights and real inputs | float32 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| UVA official PyTorch action-only LIBERO-10, public checkpoint and real inputs | float32 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31 (duplicate) | FAIL | PASS |
| UVA official PyTorch joint video/action LIBERO-10, decoded RGB output | float32 | 1, 2 | FAIL | PASS |
| SmolVLA maintained LeRobot PyTorch, public LIBERO checkpoint and real inputs | mixed float32/bfloat16 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| OpenVLA-OFT official PyTorch, public LIBERO-Spatial checkpoint and real inputs | bfloat16 network, float64 unnormalized actions | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |
| DexVLA official PyTorch classes, closest public complete checkpoint and real inputs | bfloat16 network, float32 unnormalized actions | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33 (64 OOM) | FAIL | PASS |
| LingBot-VLA 2.0 official PyTorch, official RoboTwin checkpoint and real three-camera inputs | float32 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |

The π0 entry is a numerical regression of the public companion investigation.
The π0.5 row uses the PyTorch implementation vendored by
[`batch-invariant-pizero`](https://github.com/yiannisha/batch-invariant-pizero)
and loads the public 14.47 GB PyTorch safetensors checkpoint directly. It runs
all 10 integration time points (18 midpoint ODE evaluations) with explicit
per-sample noise. Its deterministic inputs are synthetic, so the result is a
learned-weight numerical inference regression rather than a policy-quality
claim. The π0-FAST row uses LeRobot's maintained PyTorch implementation with
real observations from the public LIBERO dataset. Stock inference changed the
final action at B=3, 4, 5, and 7; the library was exact through B=64. Other
planned model families are not marked supported until their checks have run.
DreamZero is recorded as blocked under the single-H100 constraint:
its released PyTorch checkpoint is 45.85 GB and its official inference path
requires at least two GPUs.
DeVA is recorded as access-blocked: its official PyTorch inference is
single-GPU, but the public post-training checkpoint still requires gated
Cosmos-Predict2 base assets unavailable in this environment. See
[`research/deva/constraint_audit.md`](research/deva/constraint_audit.md).

The InternVLA rows use the official A1.5 PyTorch repository and public policy
and Qwen3.5-2B safetensors directly. Both upstream action paths run ten
flow-matching steps with explicit per-sample noise. Standard stock fails every
B above 1; optimized stock first fails at B=4. Both are exact through B=64 with
the library, including three real target frames. A1.5's WAN foresight branch is
training-only for the recommended action deployment and was not loaded, so no
world/video-output PASS is claimed.

The MolmoAct2 row uses the official PyTorch Transformers implementation and
five-shard public LIBERO checkpoint. Its released B>1 image-attention mask has
a broadcast bug, so the adapter applies the documented semantics-preserving
batch-dimension correction before numerical testing. Stock then fails every B
above one. The first divergence is the image projector's final `aten::mm`:
cuBLASLt selects split-K at B=1 and a different non-split-K kernel at B=2. The
existing generic invariant MM repair makes all ten continuous flow steps and
the final 10x7 action exact through B=64.

The GR00T row uses NVIDIA's official N1.7 PyTorch classes and complete public
LIBERO-10 checkpoint. Because the constructor redundantly resolves the gated
Cosmos base before overwriting it, the adapter constructs the exact public
Qwen3-VL architecture without base weights and then lets the official loader
fill every tensor from NVIDIA's shards. Stock fails every B above one. Its
first divergence is a category-conditioned `aten::bmm`; after BMM alone is
repaired, an action-DiT `aten::addmm` is the next divergence. The generic
invariant BMM and MM/addmm paths make all four flow steps and the final 16x7
action exact through B=64.

The Cosmos Nano row uses NVIDIA's official Cosmos framework, complete public
`Cosmos3-Nano-Policy-DROID` checkpoint, released Wan2.2 VAE, and real
three-camera observations from `Cosmos3-DROID`. Stock inference switches from
dense cuDNN attention at B=1 to packed variable-length CUTLASS attention at
B>1, and repeats that switch in its cached text-KV path. The generic packed
varlen-attention replacement makes all four UniPC flow steps, the 32x8 action,
and the 48x9x33x40 world latent exact through B=64. The separately decoded
33x528x640 RGB video is exact at B=2 after adding 64-bit large-tensor indexing
to persistent BMM for the Wan2.2 Conv3D path. Three DROID targets also pass
duplicate and unrelated B=2/B=4 checks.

The Cosmos Edge row uses the same official PyTorch serving path with the
complete public `Cosmos3-Edge-Policy-DROID` checkpoint and its released JSON
prompt plus `[960, 1001]` guidance-interval configuration. Its generation
path has the same dense-versus-packed attention switch and is exact through
B=64 after the reusable varlen repair. Exact latents exposed a later decoder
boundary in `aten::linalg_vector_norm`: a Wan2.2 RMS-normalization call was the
first of 1,169 traced decoder calls to differ. The new generic fixed-tree
Euclidean vector norm makes the decoded 33x528x640 RGB video exact at B=2.
Three real DROID targets pass duplicate and unrelated B=2/B=4 checks.

The X-VLA row uses LeRobot's maintained PyTorch policy, complete public LIBERO
checkpoint, released processors, and real LIBERO observations. Stock fails
every B above one. Its first divergence is Florence-2's stage-3 patch
`aten::convolution`, where cuDNN switches both tiling and segment-K behavior.
After convolution alone is repaired, the bias-free multimodal projector's
`aten::mm` is the next boundary. The existing generic Conv2d and GEMM family
make all ten flow steps and the official 30x7 trajectory exact through B=64.
Three real targets pass duplicate and unrelated B=2/B=4 checks.

The UVA rows use the authors' official PyTorch repository and complete public
LIBERO-10 EMA checkpoint. All VAE, video-diffusion, and action-diffusion random
draws are explicit per-example inputs. Stock action-only inference fails every
batch above one through B=64; the invariant path is exact for both composition
modes through B=17 and additionally at duplicate B=31 (unrelated B=31 and
duplicate B=32 exceed memory). Three real target windows pass all B=2/B=4
checks. The first stock divergence is CLIP's block-0 `aten::addmm`; after GEMM
repair, the VAE's second downsample `aten::convolution` is next. The released
action-serving mode intentionally bypasses video generation, so the separate
joint-path check executes 100 controlled video-diffusion steps and verifies
the 4x16x16x16 video latent, official VAE-decoded 4x256x256 RGB video, and
8x10 action output at B=2.

The SmolVLA row uses the maintained LeRobot PyTorch implementation and the
complete public `HuggingFaceVLA/smolvla_libero` checkpoint. Real two-camera
observations, 8D state, task text, released normalization, and explicit
per-example flow noise drive the official ten-step sampler. Stock fails every
batch above one; the first divergence is the bias-free SmolVLM modality
connector's `aten::mm`. With MM alone repaired, the state projection's
`aten::addmm` is next. The generic MM/addmm paths make the complete 50x7 action
chunk exact through B=64, and three target frames pass all B=2/B=4 checks.

The OpenVLA-OFT row uses the authors' official PyTorch implementation,
official bidirectional-attention Transformers fork, complete public
LIBERO-Spatial checkpoint, and real two-camera observations. Stock fails every
batch above one. Its first divergence is DINO block 0, where B=1 selects
split-K flash SDPA and B=2 selects a different non-split flash kernel. With
SDPA repaired, the wide continuous action-head `aten::addmm` is the next
boundary. The existing generic SDPA and MM/addmm paths make the complete 8x7
action chunk exact through B=64. Three target frames pass all B=2/B=4 checks.

The DexVLA row uses the authors' official PyTorch classes with the closest
public complete Qwen2-VLA plus ScaleDP checkpoint; the authors do not publish a
complete end-to-end checkpoint, so that provenance limitation is explicit in
the evidence. Real three-camera observations drive official greedy reasoning,
FiLM conditioning, and ten DDIM steps with explicit per-example noise. Stock
fails every completed batch above one and OOMs at B=7. Its first divergence is
vision block-0 packed SDPA; with that repaired, ScaleDP's conditioning
`aten::addmm` is next. In invariant mode, the adapter passes official Qwen2-VL
`cu_seqlens` to the reusable varlen attention operator and trims each row's
FiLM context at its own EOS. The complete 50x14 action is exact through B=33;
B=64 OOMs. Three target frames pass all B=2/B=4 checks.

The LingBot-VLA 2.0 row uses the authors' official PyTorch repository and
complete official 6B RoboTwin checkpoint without weight conversion. The
released inference MoE uses relaxed atomic route packing and accumulation and
is not even repeatable at B=1, so the adapter substitutes the reusable
deterministic token-choice MoE reference before measuring batch effects. It
also clones explicit per-frame flow noise and clears upstream shape-only
vision cache state between calls. The controlled stock path fails every batch
above one. Its first divergence is the flow state projection's FP32
`aten::addmm`; invariant addmm plus MM still leaves eager-attention BMM as a
later boundary. Generic addmm, MM, and BMM make all ten flow steps and the
complete 50x55 action chunk exact through B=64. Three target frames pass all
B=2/B=4 checks. The real three-camera fixture conforms to the official
RoboTwin transform but is not claimed as a distribution-matched task-success
evaluation.

Raw evidence is in [`research/pi0`](research/pi0),
[`research/pi05`](research/pi05), and
[`research/pi_fast`](research/pi_fast), and
[`research/internvla_a15`](research/internvla_a15), including environments,
upstream commits, baseline/fixed hashes, first-divergence diagnostics,
multiple-input checks, iteration notes, and operator benchmarks. MolmoAct2 evidence is in
[`research/molmoact2`](research/molmoact2), and GR00T N1.7 evidence is in
[`research/groot_n17`](research/groot_n17). Cosmos 3 Nano evidence is in
[`research/cosmos3_nano`](research/cosmos3_nano), and Cosmos 3 Edge evidence
is in [`research/cosmos3_edge`](research/cosmos3_edge). X-VLA evidence is in
[`research/xvla`](research/xvla), UVA evidence is in
[`research/uva`](research/uva), SmolVLA evidence is in
[`research/smolvla`](research/smolvla), and OpenVLA-OFT evidence is in
[`research/openvla_oft`](research/openvla_oft). DexVLA evidence is in
[`research/dexvla`](research/dexvla), and LingBot-VLA 2.0 evidence is in
[`research/lingbot_vla2`](research/lingbot_vla2). The earlier random-weight
official OpenPI architecture experiment is retained separately in
[`research/pi05_openpi_architecture`](research/pi05_openpi_architecture) and is
not the basis of the π0.5 support row. The DreamZero constraint audit is in
[`research/dreamzero`](research/dreamzero).

## Reproduction

Run the operator regression suite and the focused stock-kernel demonstration:

```bash
python -m pytest -q
python scripts/check_conv2d_bmm_batch_invariance.py --require-standard-difference
```

The reusable model harness accepts an adapter module and tests repeatability,
duplicate batches, and unrelated batches. For the π0 reference adapter, set
`PIZERO_SOURCE` and `PIZERO_CONFIG` to the companion checkout before running:

```bash
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.pi0_reference \
  --batch-invariant-ops \
  --output research/pi0/fixed.json
```

For π0.5, check out the pinned companion reference, apply the recorded loader
compatibility patch, and download the public PyTorch checkpoint:

```bash
git clone https://github.com/yiannisha/batch-invariant-pizero
git -C batch-invariant-pizero checkout f1eca35dd0e493f7d2b5f2cedf5b86d4403a9841
git -C batch-invariant-pizero apply \
  "$PWD/scripts/model_invariance/patches/pi_zero_pytorch_pi05_compat.patch"

# Download config.json and model.safetensors from lerobot/pi05_base, then:
PIZERO_PYTORCH_SOURCE="$PWD/batch-invariant-pizero/pi-zero-pytorch" \
PIZERO_PI05_CHECKPOINT=/path/to/pi05_base \
PIZERO_PI05_STEPS=10 \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.pizero_pi05_reference \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/pi05/fixed.json
```

Omit `--batch-invariant-ops` and write to `baseline.json` for the stock CUDA
comparison. The adapter copies tensors directly from safetensors into the
model, so it does not create a second 14 GB converted checkpoint.

For π0-FAST, check out the LeRobot commit and download the pinned public model,
tokenizers, and LIBERO episode-zero files recorded in
[`research/pi_fast/checkpoint.txt`](research/pi_fast/checkpoint.txt). Extract
the first 64 frames from each camera video with ffmpeg into `frames/image_%03d.png`
and `frames/image2_%03d.png`, then run:

```bash
mkdir -p /path/to/libero-episode-zero/frames
ffmpeg -i /path/to/image/file-000.mp4 -frames:v 64 \
  /path/to/libero-episode-zero/frames/image_%03d.png
ffmpeg -i /path/to/image2/file-000.mp4 -frames:v 64 \
  /path/to/libero-episode-zero/frames/image2_%03d.png

LEROBOT_PI0FAST_CHECKPOINT=/path/to/pi0fast-libero \
LEROBOT_PI0FAST_SAMPLE_DIR=/path/to/libero-episode-zero \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.lerobot_pi0fast \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/pi_fast/fixed.json
```

Use `scripts/trace_pi0fast_batch_invariance.py` for layer-boundary hashes and
`scripts/check_pi0fast_multiple_inputs.py` for the three-target regression.

For InternVLA-A1.5, check out the commit and download the pinned policy and
Qwen PyTorch snapshots recorded in
[`research/internvla_a15/checkpoint.txt`](research/internvla_a15/checkpoint.txt).
Use the same LIBERO episode-zero fixture as π0-FAST, then run either official
action backend:

```bash
INTERNVLA_A15_CHECKPOINT=/path/to/InternVLA-A1.5-Libero \
INTERNVLA_A15_QWEN=/path/to/Qwen3.5-2B \
INTERNVLA_A15_SAMPLE_DIR=/path/to/libero-episode-zero \
INTERNVLA_A15_BACKEND=standard \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.internvla_a15 \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/internvla_a15/standard_fixed.json
```

Set `INTERNVLA_A15_BACKEND=optimized` for the recommended optimized action-only
path. `scripts/trace_internvla_a15_batch_invariance.py` records the precise
attention boundary and flow-step propagation;
`scripts/check_internvla_a15_multiple_inputs.py` runs the three-target check.

For MolmoAct2, download the pinned `allenai/MolmoAct2-LIBERO` PyTorch snapshot
recorded in [`research/molmoact2/checkpoint.txt`](research/molmoact2/checkpoint.txt).
Reuse the LIBERO episode-zero fixture above, then run:

```bash
MOLMOACT2_CHECKPOINT=/path/to/MolmoAct2-LIBERO \
MOLMOACT2_SAMPLE_DIR=/path/to/libero-episode-zero \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.molmoact2 \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/molmoact2/fixed.json
```

Use `scripts/trace_molmoact2_batch_invariance.py` for the first-divergence and
flow-step trace, `scripts/check_molmoact2_multiple_inputs.py` for the
three-target regression, and `scripts/benchmark_molmoact2_ops.py` for the exact
projector-MM benchmark.

For GR00T N1.7, check out NVIDIA's pinned Isaac-GR00T commit and download the
`libero_10` subdirectory of the pinned `nvidia/GR00T-N1.7-LIBERO` PyTorch
snapshot recorded in
[`research/groot_n17/checkpoint.txt`](research/groot_n17/checkpoint.txt).
The separately gated Cosmos base is not needed for its weights: download only
the config and processor/tokenizer assets of the pinned public
`Qwen/Qwen3-VL-2B-Instruct` snapshot. Reuse the LIBERO fixture above, add the
official repository to `PYTHONPATH`, then run:

```bash
GROOT_N17_CHECKPOINT=/path/to/GR00T-N1.7-LIBERO/libero_10 \
GROOT_N17_PROCESSOR=/path/to/Qwen3-VL-2B-Instruct-processor-assets \
GROOT_N17_SAMPLE_DIR=/path/to/libero-episode-zero \
PYTHONPATH="$PWD:/path/to/Isaac-GR00T" \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.groot_n17 \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/groot_n17/fixed.json
```

Use `scripts/trace_groot_n17_batch_invariance.py` for the two causal
operator boundaries and four-step trace,
`scripts/check_groot_n17_multiple_inputs.py` for the three-target regression,
and `scripts/benchmark_groot_n17_ops.py` for the exact BMM/addmm benchmarks.

For Cosmos 3 Nano, check out the pinned NVIDIA Cosmos framework and download
the pinned policy, DROID data, and Wan2.2 VAE snapshots recorded in
[`research/cosmos3_nano/checkpoint.txt`](research/cosmos3_nano/checkpoint.txt).
Extract 64 synchronized frames from the wrist and two exterior video streams,
add the official repository to `PYTHONPATH`, then run:

```bash
COSMOS3_NANO_CHECKPOINT=/path/to/Cosmos3-Nano-Policy-DROID \
COSMOS3_DROID_SAMPLE_DIR=/path/to/cosmos3-droid-sample \
HF_HOME=/path/to/huggingface-cache \
PYTHONPATH="$PWD:/path/to/cosmos-framework" \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.cosmos3_nano_policy \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/cosmos3_nano/fixed.json
```

Set `COSMOS3_DECODE_VIDEO=1` and use B=1,2 to include the released VAE's
decoded RGB output. `scripts/trace_cosmos3_nano_batch_invariance.py` records
the stock, generic-operators-only, and fully repaired boundaries;
`scripts/check_cosmos3_nano_multiple_inputs.py` runs the three-target check;
and `scripts/benchmark_cosmos3_nano_ops.py` benchmarks both actual attention
geometries. The adapter defaults to eager official inference because compiled
graphs capture kernels before runtime dispatcher selection;
`COSMOS3_TORCH_COMPILE=1` restores the upstream compiled serving switch.

For Cosmos 3 Edge, use the same pinned framework, DROID fixture, and VAE, then
download the Edge checkpoint recorded in
[`research/cosmos3_edge/checkpoint.txt`](research/cosmos3_edge/checkpoint.txt):

```bash
COSMOS3_EDGE_CHECKPOINT=/path/to/Cosmos3-Edge-Policy-DROID \
COSMOS3_DROID_SAMPLE_DIR=/path/to/cosmos3-droid-sample \
HF_HOME=/path/to/huggingface-cache \
PYTHONPATH="$PWD:/path/to/cosmos-framework" \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.cosmos3_edge_policy \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/cosmos3_edge/fixed.json
```

The generic Cosmos trace and multiple-input scripts accept the Edge adapter
through `--adapter`. `scripts/trace_cosmos3_video_decoder.py` fingerprints the
Wan decoder leaf calls, and `scripts/benchmark_cosmos3_edge_ops.py` benchmarks
Edge's actual attention and vector-norm shapes.

For X-VLA, check out the pinned LeRobot commit and download the public policy
and LIBERO episode-zero files recorded in
[`research/xvla/checkpoint.txt`](research/xvla/checkpoint.txt). Extract the
first 64 synchronized frames as for π0-FAST, then run:

```bash
LEROBOT_XVLA_CHECKPOINT=/path/to/xvla-libero \
LEROBOT_XVLA_SAMPLE_DIR=/path/to/libero-episode-zero \
PYTHONPATH="$PWD:/path/to/lerobot/src" \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.lerobot_xvla \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/xvla/fixed.json
```

Use `scripts/trace_lerobot_xvla_batch_invariance.py` for both staged operator
boundaries, `scripts/check_lerobot_xvla_multiple_inputs.py` for the three-target
regression, and `scripts/benchmark_lerobot_xvla_ops.py` for the exact Conv2d
and projector-MM benchmarks.

For UVA, check out the official commit and download the LIBERO-10 checkpoint
recorded in [`research/uva/checkpoint.txt`](research/uva/checkpoint.txt). With
the same real LIBERO fixture, run:

```bash
UVA_LIBERO_CHECKPOINT=/path/to/libero10.ckpt \
UVA_LIBERO_SAMPLE_DIR=/path/to/libero-episode-zero \
PYTHONPATH="$PWD:/path/to/unified_video_action" \
python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.uva_libero10 \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/uva/fixed.json
```

Use the `uva_libero10_joint` adapter for joint decoded-video/action checks,
`scripts/trace_uva_libero10_batch_invariance.py` for the staged addmm/Conv2d
trace, `scripts/check_uva_libero10_multiple_inputs.py` for the three-target
regression, and `scripts/benchmark_uva_libero10_ops.py` for the actual learned
temporal-upsample shape.

For SmolVLA, use the official `HuggingFaceVLA/smolvla_libero` checkpoint and
the pinned LeRobot commit recorded under `research/smolvla`, then run the
generic harness with adapter
`scripts.model_invariance.adapters.lerobot_smolvla`. The corresponding trace,
three-target regression, and connector benchmark are
`scripts/trace_lerobot_smolvla_batch_invariance.py`,
`scripts/check_lerobot_smolvla_multiple_inputs.py`, and
`scripts/benchmark_lerobot_smolvla_ops.py`.

For OpenVLA-OFT, use the official LIBERO-Spatial checkpoint and pinned source
commits under `research/openvla_oft`. Apply the PyTorch-inference-only import
patch in `scripts/model_invariance/patches`, then run the generic harness with
adapter `scripts.model_invariance.adapters.openvla_oft`. The corresponding
staged trace, three-target regression, and operator benchmark are
`scripts/trace_openvla_oft_batch_invariance.py`,
`scripts/check_openvla_oft_multiple_inputs.py`, and
`scripts/benchmark_openvla_oft_ops.py`.

For DexVLA, use the official source commit and community full-checkpoint
revision recorded under `research/dexvla`, plus the official example HDF5
episode. Set `DEXVLA_CHECKPOINT` and `DEXVLA_SAMPLE`, add the official source
and `policy_heads` directory to `PYTHONPATH`, and run the generic harness with
adapter `scripts.model_invariance.adapters.dexvla`. The official-equivalence,
staged trace, three-target regression, and operator benchmark are
`scripts/check_dexvla_official_equivalence.py`,
`scripts/trace_dexvla_batch_invariance.py`,
`scripts/check_dexvla_multiple_inputs.py`, and
`scripts/benchmark_dexvla_ops.py`.

For LingBot-VLA 2.0, use the official source, RoboTwin checkpoint, and pinned
Qwen3-VL assets recorded under `research/lingbot_vla2`. Set
`LINGBOT_VLA2_CHECKPOINT`, `LINGBOT_VLA2_SAMPLE`, and `QWEN3VL_PATH`, add the
official source to `PYTHONPATH`, and run the generic harness with adapter
`scripts.model_invariance.adapters.lingbot_vla2`. The released-kernel
comparison, staged trace, three-target regression, and operator benchmark are
`scripts/check_lingbot_vla2_official_equivalence.py`,
`scripts/trace_lingbot_vla2_batch_invariance.py`,
`scripts/check_lingbot_vla2_multiple_inputs.py`, and
`scripts/benchmark_lingbot_vla2_ops.py`.

## Performance

Batch invariance changes the arithmetic decomposition and can be slower than
vendor kernels. On an NVIDIA H100 NVL with PyTorch 2.5.0, CUDA 12.4, Triton
3.1.0, and bfloat16 π0 shapes, the Conv2D replacement was 1.86–4.46× slower
and the BMM replacement was 1.13–3.16× slower across B=1, 8, and 32. See
[`research/pi0/benchmarks.json`](research/pi0/benchmarks.json) for latency,
throughput, memory, shapes, and exact environment data.

For the π0.5 float32 reference on PyTorch 2.14.0/CUDA 13.0, the actual SigLIP
patch Conv2d replacement was 3.53–12.02× slower across B=1, 8, and 32. The
explicit SigLIP QK BMM replacement was 1.28–1.75× slower and its
attention-value BMM was 1.35–2.75× slower. See
[`research/pi05/benchmarks.json`](research/pi05/benchmarks.json).

For π0-FAST on PyTorch 2.8.0/CUDA 12.8, the invariant layer-0 MLP down GEMM
was 1.17–1.39× slower across B=1, 2, and 8. The invariant RMS mean took about
0.030–0.036 ms versus 0.007–0.009 ms for stock. See
[`research/pi_fast/benchmarks.json`](research/pi_fast/benchmarks.json).

For InternVLA-A1.5 on PyTorch 2.8.0/CUDA 12.8, the actual action-attention BMM
was 2.45–6.40× slower and the Qwen visual Conv3d was 2.24–4.74× slower across
B=1, 2, and 8. The specialized invariant Qwen depthwise Conv1d was 0.49–0.59×
stock latency. See
[`research/internvla_a15/benchmarks.json`](research/internvla_a15/benchmarks.json).

For MolmoAct2 on the same environment, the invariant image-projector MM was
2.38×, 1.75×, and 1.41× slower than stock at B=1, 2, and 8, respectively. See
[`research/molmoact2/benchmarks.json`](research/molmoact2/benchmarks.json).

For GR00T N1.7 on PyTorch 2.9.0/CUDA 12.8, the invariant action-encoder BMM
was 2.78×, 3.78×, and 1.80× slower at B=1, 2, and 8. The action-DiT addmm was
2.86×, 3.53×, and 1.69× slower. See
[`research/groot_n17/benchmarks.json`](research/groot_n17/benchmarks.json).

For Cosmos 3 Nano on PyTorch 2.13.0/CUDA 13.0, invariant packed causal
attention was 1.34–2.05× slower and dominant full attention was 5.22–6.75×
slower across B=1, 2, and 8. The full path materializes scores but bounds each
launch's score memory; complete inference still fits B=64 on the recorded
H100. See [`research/cosmos3_nano/benchmarks.json`](research/cosmos3_nano/benchmarks.json).

For Cosmos 3 Edge in the same environment, invariant causal attention was
1.53–2.13× slower and full attention was 5.57–6.78× slower across B=1, 2, and
8. The invariant Wan decoder vector norm was 1.51–2.06× slower, while avoiding
stock's roughly 173 MB temporary allocation at B=1/B=2. See
[`research/cosmos3_edge/benchmarks.json`](research/cosmos3_edge/benchmarks.json).

For X-VLA on PyTorch 2.14.0/CUDA 13.0, the optimized invariant Florence
stage-3 Conv2d is 14.12x, 11.44x, and 12.70x slower than cuDNN at B=1, 2, and
8, while reducing incremental memory at B=1/B=2. The invariant image-projector
MM is 7.93x, 5.95x, and 2.44x slower. See
[`research/xvla/benchmarks.json`](research/xvla/benchmarks.json).

For UVA on the same environment, its float32 1024-channel temporal
ConvTranspose3d is 1.99× slower at B=1 and 4.37× slower at B=64 with the
fixed-schedule implementation. Both paths happen to be exact at this shape
through B=64, but the replacement supplies a batch-independent decomposition
and lets the generic convolution dispatcher cover the full model. See
[`research/uva/benchmarks.json`](research/uva/benchmarks.json).

For SmolVLA on the same environment, the actual float32 modality-connector MM
replacement is 22.68x, 15.25x, and 4.63x slower than stock at B=1, 2, and 8,
respectively, with identical incremental output allocation. See
[`research/smolvla/benchmarks.json`](research/smolvla/benchmarks.json).

For OpenVLA-OFT on PyTorch 2.8.0/CUDA 12.8, invariant DINO block-0 SDPA is
7.56x, 8.30x, and 2.91x slower than stock at B=1, 2, and 8. The invariant
continuous action-head addmm is 4.25x, 4.41x, and 3.28x slower. See
[`research/openvla_oft/benchmarks.json`](research/openvla_oft/benchmarks.json).

For DexVLA on the same environment, invariant segmented vision SDPA is 5.11x,
5.39x, and 6.08x slower than stock at robot B=1, 2, and 8. The invariant
ScaleDP conditioning addmm is 3.72x, 2.99x, and 2.85x slower. See
[`research/dexvla/benchmarks.json`](research/dexvla/benchmarks.json).

For LingBot-VLA 2.0 FP32 inference on the same environment, invariant
state-projection addmm is 4.46x, 4.18x, and 4.67x slower at B=1, 2, and 8;
action-expert MM is 4.49x, 3.02x, and 1.15x slower; action-attention BMM is
2.52x, 3.22x, and 2.67x slower; and the dense deterministic MoE fallback is
8.22x, 8.31x, and 14.04x slower than the released atomic kernel. See
[`research/lingbot_vla2/benchmarks.json`](research/lingbot_vla2/benchmarks.json).

## Attribution

The operator approach and original matrix/reduction kernels come from Thinking
Machines Lab's [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
The Conv2D/BMM investigation and π0 tracing methodology build on
[Batch-Invariant VLAs](https://yiannisha.dev/blog/batch-invariant-vlas) and the
[`batch-invariant-pizero`](https://github.com/yiannisha/batch-invariant-pizero)
companion repository. π0 and π0.5 are from Physical Intelligence; these
regressions use the cited open PyTorch implementations and exact commits
recorded in the research files.

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

### Attention Operations
- `torch.nn.functional.scaled_dot_product_attention()` - evaluation-time SDPA
  (`dropout_p=0`) using independent batch-invariant BMM and Triton softmax
  reductions; causal, boolean/additive mask, and grouped-query paths are
  supported

### Convolution Operations
- `torch.nn.functional.conv1d()` / `nn.Conv1d` - Regular 1-D convolution,
  including a fixed-order Triton depthwise path used by Qwen3.5
- `torch.nn.functional.conv2d()` / `nn.Conv2d` - Regular (non-transposed) 2-D
  convolution, implemented as an independent unfold-and-GEMM for each sample
- `torch.nn.functional.conv3d()` / `nn.Conv3d` - Regular 3-D convolution using
  batch-independent grouped im2col/BMM, including Qwen3.5-VL patch embedding

### Activation Functions
- `torch.log_softmax()` - Log-softmax activation
- `batch_invariant_ops.softmax()` - last-dimension softmax used by the SDPA
  replacement

### Reduction Operations
- `torch.mean()` - Mean computation along specified dimensions

The current kernels cover CUDA float32, float16, and bfloat16 paths exercised
by the tests. Conv1D/2D/3D cover regular non-transposed convolution, including
groups and dilation. Unsupported operator overloads and dtypes are not claimed.

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

The π0 entry is a numerical regression of the public companion investigation.
The π0.5 row uses the PyTorch implementation vendored by
[`batch-invariant-pizero`](https://github.com/yiannisha/batch-invariant-pizero)
and loads the public 14.47 GB PyTorch safetensors checkpoint directly. It runs
all 10 integration time points (18 midpoint ODE evaluations) with explicit
per-sample noise. Its deterministic inputs are synthetic, so the result is a
learned-weight numerical inference regression rather than a policy-quality
claim. The π0-FAST row uses LeRobot's maintained PyTorch implementation with
real observations from the public LIBERO dataset. Stock inference changed the
final action at B=3, 4, 5, and 7; the library was exact through B=64. GR00T,
Cosmos, and the other planned model families are not yet marked supported.
DreamZero is recorded as blocked under the single-H100 constraint:
its released PyTorch checkpoint is 45.85 GB and its official inference path
requires at least two GPUs.

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

Raw evidence is in [`research/pi0`](research/pi0),
[`research/pi05`](research/pi05), and
[`research/pi_fast`](research/pi_fast), and
[`research/internvla_a15`](research/internvla_a15), including environments, upstream
commits, baseline/fixed hashes, first-divergence diagnostics, multiple-input
checks, iteration notes, and operator benchmarks. MolmoAct2 evidence is in
[`research/molmoact2`](research/molmoact2). The earlier random-weight
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

## Attribution

The operator approach and original matrix/reduction kernels come from Thinking
Machines Lab's [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
The Conv2D/BMM investigation and π0 tracing methodology build on
[Batch-Invariant VLAs](https://yiannisha.dev/blog/batch-invariant-vlas) and the
[`batch-invariant-pizero`](https://github.com/yiannisha/batch-invariant-pizero)
companion repository. π0 and π0.5 are from Physical Intelligence; these
regressions use the cited open PyTorch implementations and exact commits
recorded in the research files.

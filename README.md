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
- `torch.nn.functional.conv2d()` / `nn.Conv2d` - Regular (non-transposed) 2-D
  convolution, implemented as an independent unfold-and-GEMM for each sample

### Activation Functions
- `torch.log_softmax()` - Log-softmax activation
- `batch_invariant_ops.softmax()` - last-dimension softmax used by the SDPA
  replacement

### Reduction Operations
- `torch.mean()` - Mean computation along specified dimensions

The current kernels cover CUDA float32, float16, and bfloat16 paths exercised
by the tests. Conv2D covers regular non-transposed 2-D convolution, including
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
| Official OpenPI π0.5 PyTorch architecture, random weights and controlled synthetic inputs | mixed bfloat16/float32 | 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64 | FAIL | PASS |

The π0 entry is a numerical regression of the public companion investigation.
The π0.5 entry runs the current official OpenPI implementation, including all
10 flow steps, but uses random parameters: OpenPI distributes the checkpoint
in a 12.43 GB JAX form and the available workspace could not also hold the
converted PyTorch copy. Neither row is an official-checkpoint policy-quality
claim. Official-weight π0/π0.5, π0-FAST, DreamZero, InternVLA-A, MolmoAct,
GR00T, Cosmos, and the other planned model families are not yet marked
supported.

Raw evidence is in [`research/pi0`](research/pi0) and
[`research/pi05`](research/pi05), including environments, upstream commits,
baseline/fixed hashes, first-divergence diagnostics, iteration notes, and
operator benchmarks.

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

For the official OpenPI PyTorch architecture, run from an OpenPI environment
with its documented Transformers replacement installed and this repository on
`PYTHONPATH`:

```bash
OPENPI_NUM_STEPS=10 python scripts/check_model_batch_invariance.py \
  --adapter scripts.model_invariance.adapters.openpi_pi05 \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output research/pi05/fixed.json
```

Set `OPENPI_PYTORCH_WEIGHTS` to a converted `model.safetensors` path to run the
same adapter with released parameters.

## Performance

Batch invariance changes the arithmetic decomposition and can be slower than
vendor kernels. On an NVIDIA H100 NVL with PyTorch 2.5.0, CUDA 12.4, Triton
3.1.0, and bfloat16 π0 shapes, the Conv2D replacement was 1.86–4.46× slower
and the BMM replacement was 1.13–3.16× slower across B=1, 8, and 32. See
[`research/pi0/benchmarks.json`](research/pi0/benchmarks.json) for latency,
throughput, memory, shapes, and exact environment data.

On the OpenPI PyTorch 2.7.1/CUDA 12.6 environment, batch-invariant SDPA was
4.27–7.27× slower than fused flash attention for the π0.5 SigLIP shape across
B=1, 8, and 32, with a larger materialized attention matrix. See
[`research/pi05/benchmarks.json`](research/pi05/benchmarks.json).

## Attribution

The operator approach and original matrix/reduction kernels come from Thinking
Machines Lab's [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
The Conv2D/BMM investigation and π0 tracing methodology build on
[Batch-Invariant VLAs](https://yiannisha.dev/blog/batch-invariant-vlas) and its
companion repositories. π0 is from Physical Intelligence; the regression uses
the cited open PyTorch reimplementation commit recorded in the research files.
The π0.5 architecture regression uses Physical Intelligence's official OpenPI
PyTorch implementation at the exact commit recorded in `research/pi05`.

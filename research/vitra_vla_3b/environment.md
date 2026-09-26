# VITRA-VLA-3B environment

- Date: 2026-09-26
- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 4.47.1 (the version pinned by upstream VITRA)
- tokenizers: 0.21.4
- timm: 0.9.10
- NumPy: 2.1.2
- SciPy: 1.14.1
- Pillow: 11.0.0

The official inference builder leaves `use_bf16=False` and moves the native
float32 checkpoint to CUDA without casting. This investigation follows that
released behavior: PaliGemma2, the FOV encoder, and DiT run in float32; the
released NumPy statistics postprocessing promotes the returned 16x102 action
trajectory to float64.

Inputs use all three real images shipped in the official repository, the
published prompt, published zero-state example, 60-degree FOV, released action
mask and normalization statistics, CFG 5.0, and ten DDIM steps. Initial DDIM
noise is an explicit per-sample input. Dropout is disabled with `eval()`.

Required environment variables:

```bash
VITRA_SOURCE=/path/to/VITRA
VITRA_CHECKPOINT=/path/to/VITRA-VLA-3B
VITRA_PALIGEMMA_METADATA=/path/to/paligemma2-3b-mix-224-metadata
```

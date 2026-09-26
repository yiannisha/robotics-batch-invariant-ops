# MolmoAct2 execution environment

- GPU: NVIDIA H100 NVL, 95,830 MiB
- NVIDIA driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 5.2.0
- NumPy: 2.1.2
- PyArrow: 25.0.1
- Pillow: 11.0.0

The official PyTorch remote-code model is loaded from the pinned checkpoint
snapshot. The existing isolated environment at
`models/internvla-a-series/.venv` supplies the compatible PyTorch and
Transformers runtime; `batch_invariant_ops` is imported from this checkout.

Inference uses bfloat16 weights and autocast, evaluation mode, two real LIBERO
camera views, raw 8-D robot state, the checkpoint's `libero` normalization
metadata, ten continuous flow-matching steps, and a 10x7 unnormalized final
action trajectory. CUDA graphs are disabled so shape-specialized graph capture
does not obscure the eager operator path under investigation.

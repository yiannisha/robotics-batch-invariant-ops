# SpatialVLA execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 4.47.0
- Accelerate: 1.0.1
- safetensors: 0.8.0
- NumPy: 2.1.2
- SciPy: 1.14.1
- Pillow: 11.0.0

The official checkpoint is loaded directly in BF16 with `trust_remote_code=True`,
`local_files_only=True`, and no FlashAttention package. Transformers therefore
uses the checkpoint's released eager/SDPA-compatible PyTorch code without a
JAX path or checkpoint conversion.

# GR00T N1.7 execution environment

- GPU: NVIDIA H100 NVL, 95,830 MiB
- NVIDIA driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.9.0+cu128
- TorchVision: 0.24.0+cu128
- CUDA reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.5.0
- Transformers: 4.57.3
- Diffusers: 0.35.1
- NumPy: 1.26.4
- PyArrow: 25.0.1
- Pillow: 12.3.0
- Albumentations: 1.4.18

The official NVIDIA PyTorch source and complete two-shard LIBERO-10 checkpoint
are loaded from pinned local snapshots. The isolated environment at
`models/internvla-a-series/.venv` supplies the official GR00T-compatible stack;
`batch_invariant_ops` is imported from this checkout.

Inference uses bfloat16 weights, evaluation mode, two real LIBERO camera views,
raw 8-D robot state, the released processor and normalization statistics, four
continuous flow-matching steps, explicit per-example bfloat16 noise, and the
official LIBERO gripper postprocessing. The decoded result is the first 16
steps of the 7-D action trajectory. Flash Attention is not installed, so the
official backbone takes its supported PyTorch SDPA fallback.

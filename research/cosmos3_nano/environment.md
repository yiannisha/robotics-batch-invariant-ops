# Cosmos 3 Nano Policy execution environment

- GPU: NVIDIA H100 NVL, 95,830 MiB
- NVIDIA driver: 580.126.09
- Python: 3.13.8
- PyTorch: 2.13.0+cu130
- CUDA reported by PyTorch: 13.0
- cuDNN: 9.2.0
- Triton: 3.7.1
- Transformers: 4.57.6
- Diffusers: 0.39.0
- NATTEN: 0.21.6+cu130.torch213
- NumPy: 2.2.6
- PyArrow: 25.0.1
- Pillow: 12.3.0

The official NVIDIA PyTorch source, full Cosmos3 Nano DROID checkpoint, and
Wan2.2 VAE are loaded from pinned local snapshots. `batch_invariant_ops` is
imported from this checkout.

Inference uses the released bfloat16 network, official DROID transforms,
three real camera streams, joint and gripper state, four UniPC flow steps,
guidance 3.0, shift 5.0, and explicit per-example diffusion seeds. The model
jointly produces a 32x8 float32 action trajectory and a 48x9x33x40 float32
future-video latent. The optional decoded-video check uses the official Wan2.2
VAE and compares the resulting uint8 frames.

The adapter disables the upstream `torch.compile` serving switch because
compiled graphs capture lower-level attention kernels before the library's
runtime dispatcher context is selected. `COSMOS3_TORCH_COMPILE=1` restores the
unaltered upstream compiled path for baseline experiments.

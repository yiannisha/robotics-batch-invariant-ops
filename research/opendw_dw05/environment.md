# OpenDW DW05 execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Model dtype: bfloat16; returned actions: float32; decoded video: uint8
- Official source commit: `e33befa8005a1585e0140dbf464566e90bc79aa1`
- Official checkpoint revision: `6ab5f9e2636610cba440d08264663efe70c3f761`

The two-line inference-only patch enables memory-mapped, weights-only loading
of the official PyTorch `.pt`/`.pth` files. It does not change model arithmetic.

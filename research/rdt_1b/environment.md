# RDT-1B execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 4.41.0
- Diffusers: 0.27.2
- timm: 1.0.3
- huggingface-hub: 0.23.2
- NumPy: 2.1.2
- Pillow: 11.0.0

The official PyTorch state dictionary is loaded directly into the released
`RDTRunner` BF16 architecture. The official SigLIP safetensors are loaded by
the released `SiglipVisionTower`. No JAX implementation, checkpoint conversion,
or randomly initialized learned component is used.

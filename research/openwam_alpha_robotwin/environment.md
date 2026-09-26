# OpenWAM Alpha RoboTwin execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Model dtype: bfloat16
- Physical action output: float32, `[B, 32, 20]`
- Joint video latent: float32, `[B, 48, 3, 24, 20]`
- Decoded video: uint8, `[B, 3, 9, 384, 320]`
- Official source commit: `c51935fb24c0d4a206ebb41f02841372c6cf2f7e`
- Official checkpoint revision: `04b96af53eeeb111c64631f822efcbb82f4b186e`

OpenWAM currently requires newer pure-Python Transformers and Diffusers
packages than the other integrations. They were installed in an isolated
runtime path; the repository's PyTorch 2.8 installation and the model source
were not modified.

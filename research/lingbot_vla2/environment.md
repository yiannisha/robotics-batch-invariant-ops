# LingBot-VLA 2.0 execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- transformers: 4.57.3
- tokenizers: 0.22.2
- safetensors: 0.5.3
- flash-attn: 2.8.3
- h5py: 3.16.0
- official LingBot-VLA 2.0 source: `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`

The checkpoint is evaluated in the authors' released FP32 inference mode.
Model construction still uses the official FlashAttention-2 vision/text
configuration; the action/VLM joint attention path is the repository's eager
FP32 implementation. The source checkout remains unmodified. Runtime controls
and the MoE compatibility bridge live only in this repository's adapter.

# CogACT-Small execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 4.40.1
- timm: 0.9.10
- PEFT: 0.11.1
- tokenizers: 0.19.1
- sentencepiece: 0.1.99
- draccus: 0.3.1
- NumPy: 2.1.2
- Pillow: 11.0.0

The official 30.2 GB PyTorch checkpoint is memory-mapped and loaded strictly
into the authors' released architecture. Following the documented deployment
option, the VLM runs in BF16 and DiT-S remains FP32. This occupies about 15.2
GB after loading and peaked near 15.54 GB for B=1 inference.

The complete checkpoint includes every learned DINO, SigLIP, projector, Llama,
and action-head tensor. The inference-only construction patch therefore avoids
downloading placeholder vision weights that would immediately be overwritten.
No JAX code, conversion, random learned component, or substitute weight is used.

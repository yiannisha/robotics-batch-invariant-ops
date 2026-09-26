# OpenHelix execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 4.40.1
- Diffusers: 0.29.2
- PEFT: 0.11.1
- Accelerate: 0.31.0
- safetensors: 0.4.5
- torchvision: 0.23.0+cu128
- sentencepiece: 0.2.1
- einops: 0.8.1
- NumPy: 2.1.2
- Pillow: 11.0.0

The seven planner shards are streamed directly into a meta-constructed model,
so a second full checkpoint copy is never materialized. Planner parameters run
in BF16 and the policy runs in its released FP32 precision. The loaded pair
occupies about 14.4 GB of device memory.

The checkpoint supplies all learned Llama, CLIP vision, multimodal projector,
text/action head, and diffusion-policy tensors. Public LLaVA and CLIP snapshots
supply only tokenizer/configuration metadata. No JAX code, conversion, random
learned component, or substitute learned weight is used.

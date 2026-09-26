# DexVLA execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- transformers: 4.45.2
- diffusers: 0.11.1
- timm: 0.9.10
- peft: 0.13.2
- h5py: 3.16.0
- tokenizers: 0.20.1
- official DexVLA source: `fc21a822f4c774e242eb6f1ab4a235788de7aba9`

The released stack uses an old DDIM scheduler whose beta schedule is cast to
the model's bfloat16 dtype during model loading. As documented in the official
README, its cumulative alpha underflows and the last denoising step returns
NaNs. The adapter reconstructs the same scheduler after loading so its schedule
remains float32. This is the authors' documented inference fix, not a change to
the learned model or denoising configuration.

The official repository remains clean. The model-specific batch
generalizations and packed-attention bridge live only in the adapter in this
repository.


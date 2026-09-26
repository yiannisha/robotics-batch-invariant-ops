# BAAI UniVLA execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- transformers: 4.44.0
- tokenizers: 0.19.1
- safetensors: 0.8.0
- FlashAttention: 2.8.3
- SciPy: 1.15.3
- accelerate: 0.34.2
- UniVLA source: `a91ee5f269d7ad46d46635ee6481d2eb655b4af9`

The released requirements pin PyTorch 2.4.0/CUDA 12.4 and FlashAttention
2.5.7. This investigation retains the official PyTorch model, processor, and
FlashAttention-2 inference path while using PyTorch 2.8.0 and a compatible
FlashAttention build so the model exercises the same current dispatcher and
Triton implementation as `batch_invariant_ops` on H100.

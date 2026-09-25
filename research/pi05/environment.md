# OpenPI π0.5 investigation environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes (95,830 MiB)
- Driver: 580.126.09
- Python: 3.11.13
- PyTorch: 2.7.1+cu126
- CUDA runtime packaged with PyTorch: 12.6
- cuDNN: 9.5.1
- Triton: 3.3.1
- Transformers: 4.53.2 plus OpenPI's required `transformers_replace` files
- OpenPI: `215abfb217dbac7d5f1273282331b9b1866c0479`
- `batch_invariant_ops`: editable install from this branch
- dtype: OpenPI's default mixed bfloat16/float32 inference precision
- device: CUDA
- model configuration: `Pi0Config(pi05=True, action_horizon=15, pytorch_compile_mode=None)`
- weights: deterministic random initialization, seed 42
- inputs: three 224×224 RGB views, 200 prompt tokens, 32-D state
- generative configuration: 10 Euler flow steps; explicit per-sample `[15,32]` noise
- batch sizes: 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64
- compositions: duplicate and unrelated companions

Compilation was disabled during investigation so that dynamic-shape compilation
did not obscure eager operator dispatch. Dropout and image augmentation were
disabled by evaluation inference. Python, NumPy, CPU PyTorch, and CUDA RNGs
were reset by the common harness before every invocation.

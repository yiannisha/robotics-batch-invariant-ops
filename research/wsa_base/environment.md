# WSA Base PyTorch investigation environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime packaged with PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 4.57.1 plus WSA's bundled Qwen3-VL replacement
- Accelerate: 1.15.0
- safetensors: 0.4.5
- `batch_invariant_ops`: this branch
- implementation: official PyTorch WSA Base causal LIBERO action path
- weights: public `zaleni/WSA-Base-LIBERO` native safetensors
- precision: bfloat16 network, float32 flow state and unnormalized actions
- device: CUDA
- inputs: two real 224x224 LIBERO camera views duplicated into the released
  two-frame request history, one zero/masked missing camera, normalized 8-D
  state, and the episode's language task
- output: 10x7 unnormalized robot action trajectory
- flow integration: 10 explicit Euler steps with fixed per-sample 10x32 noise
- batch sizes: 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64
- compositions: duplicate and unrelated real LIBERO observations

Evaluation mode is enabled. The common harness resets Python, NumPy, CPU
PyTorch, and CUDA RNG state before every inference. The flow noise is an
explicit sample field and is reused unchanged for the target. The official
evaluation setting disables the training-only DA3 teacher (`lambda_3d=0`) but
retains the checkpoint's learned 3D query modules and released causal attention
path. The frozen Cosmos tokenizer is loaded as required, although the official
causal action-only fast path omits its visual-middle tokens and relies on the
Qwen current-frame prefix plus learned query block.

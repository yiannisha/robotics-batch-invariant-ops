# π0-FAST PyTorch investigation environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime packaged with PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 5.5.4
- LeRobot: 0.6.2, editable at the commit in `model_commit.txt`
- `batch_invariant_ops`: editable install from this branch
- implementation: LeRobot's maintained PyTorch `pi0_fast` policy
- weights: public `lerobot/pi0fast-libero` PyTorch safetensors
- precision: bfloat16 language model, float32 vision path, float32 decoded actions
- device: CUDA
- inputs: two real 224×224 LIBERO camera frames, one masked image slot,
  normalized 8-D robot state, and the episode's language task
- output: 10×7 unnormalized robot action trajectory
- decoding: greedy autoregressive generation with KV cache and a 256-token
  maximum; the primary target emits its action delimiter at token 24
- batch sizes: 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64
- compositions: duplicate and unrelated real LIBERO frames

Evaluation mode was enabled. The common harness resets Python, NumPy, CPU
PyTorch, and CUDA RNG state before every inference. Generation is greedy, so
there is no sampling RNG. The primary sweep uses frame zero from episode zero
as the target and the first 64 real episode frames as unrelated companions.
The multi-input check additionally uses frames 10 and 20 as targets.

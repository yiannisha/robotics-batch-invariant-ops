# Xiaomi-Robotics-1 RoboCasa execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime packaged with PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 4.57.1
- Accelerate: 1.15.0
- safetensors: 0.4.5
- `batch_invariant_ops`: this branch
- implementation: checkpoint-bundled official PyTorch `MiBoTForActionGeneration`
- weights: complete public `Xiaomi-Robotics-1-RoboCasa` native safetensors
- attention backend: checkpoint-bundled eager implementation
- precision: bfloat16 network/noise, float32 decoded physical actions
- device: CUDA
- inputs: three real RoboCasa simulator camera views, released multi-view chat
  template, deterministic plausible 8-D Panda joint/gripper state padded to 60
- output: 10x7 decoded RoboCasa action trajectory
- flow integration: five explicit Euler steps with fixed per-sample 10x60 noise
- batch sizes: 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64
- compositions: duplicate and unrelated real RoboCasa observations

Evaluation mode is enabled. The common harness resets Python, NumPy, CPU
PyTorch, and CUDA RNG state before every inference. Flow noise is an explicit
sample field and is reused unchanged for the target. The public episode stores
EEF pose rather than the joint-space state consumed by Xiaomi's evaluator, so
the adapter uses deterministic in-range Panda joint states. This is a numerical
inference test, not a policy-quality evaluation.

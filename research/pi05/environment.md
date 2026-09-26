# π0.5 PyTorch reference investigation environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes (95,830 MiB)
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.14.0+cu130
- CUDA runtime packaged with PyTorch: 13.0
- cuDNN: 9.24.0
- Triton: 3.8.0
- `batch_invariant_ops`: editable install from this branch
- implementation: `pi-zero-pytorch` vendored by `batch-invariant-pizero`
- weights: public `lerobot/pi05_base` PyTorch safetensors
- dtype: float32, as specified by the checkpoint configuration
- device: CUDA
- input configuration: three 224×224 RGB views, 200 prompt-token IDs, 32-D state
- output configuration: 50×32 action trajectory
- generative configuration: 10 time points / 18 midpoint ODE evaluations; explicit per-sample 50×32 noise
- batch sizes: 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64
- compositions: duplicate and unrelated companions

Evaluation mode was enabled. The adapter supplies flow noise explicitly per
sample and verifies that the implementation consumed that exact tensor. The
common harness resets Python, NumPy, CPU PyTorch, and CUDA RNG state before
each invocation. Inputs are deterministic synthetic tensors so this is a
numerical inference regression with learned public weights, not a robot-task
quality evaluation.

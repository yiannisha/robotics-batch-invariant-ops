# X-VLA LIBERO PyTorch investigation environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- NVIDIA driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.14.0+cu130
- CUDA reported by PyTorch: 13.0
- cuDNN: 9.24.0
- Triton: 3.8.0
- Transformers: 5.5.4
- LeRobot: 0.6.2, editable at the commit in `model_commit.txt`
- `batch_invariant_ops`: imported from this checkout
- implementation: LeRobot's maintained PyTorch X-VLA policy
- weights: public `lerobot/xvla-libero` PyTorch safetensors
- precision: float32 model and actions
- device: CUDA
- input: two real 256x256 LIBERO camera frames (officially padded/resized to
  224x224), one empty camera slot, an 8-D state padded to 20 channels, and the
  released language/task preprocessing
- generation: 10 flow-matching updates from explicit per-example 30x20 noise
- output: official ee6d decoding and rotation conversion to a 30x7 action
  trajectory
- batch sizes: 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64
- compositions: duplicate and unrelated real LIBERO observations

Evaluation mode disables the model's dropout. The common harness resets
Python, NumPy, CPU PyTorch, and CUDA RNG state before every invocation. The
released `predict_action_chunk(..., noise=...)` signature currently ignores
its `noise` argument, so the adapter follows the official `generate_actions`
body while supplying the initial flow noise as an explicit stacked input.
This is the only inference-body adaptation and prevents a batch-shaped random
draw from being misclassified as numerical batch dependence.

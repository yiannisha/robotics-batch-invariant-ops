# InternVLA-A1.5 execution environment

- GPU: NVIDIA H100 NVL, 95,830 MiB
- NVIDIA driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- Transformers: 5.2.0
- flash-linear-attention / fla-core: 0.5.0
- NumPy: 2.1.2
- PyArrow: 25.0.1
- Pillow: 11.0.0

The isolated environment is at `models/internvla-a-series/.venv`. The official
project is installed editable at commit `e6fc904f`; `batch_invariant_ops` is
also installed editable. The Qwen3.5 Transformers files are installed exactly
as directed by the upstream InternVLA repository.

`fla-core` is installed, but the optional `causal-conv1d` CUDA extension is not,
so Transformers reports `is_fast_path_available=False` and executes its public
PyTorch causal-convolution fallback. The FLA chunk gated-delta-rule kernel is
available. This fallback is recorded because it is part of the executed graph;
no JAX implementation or converted weight is involved.

Inference uses bfloat16 for the Qwen/action model in the standard backend. The
official optimized backend keeps the action expert and action projections in
float32. Both use ten flow-matching steps, explicit per-sample float32 noise,
650 prompt tokens, two 224x224 LIBERO camera views, normalized 8-D robot state,
and an 8x7 unnormalized/clipped final action trajectory.

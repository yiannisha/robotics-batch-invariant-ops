# π0 reference environment

- GPU: NVIDIA H100 NVL, 95,830 MiB
- Driver: 580.126.09
- Python: 3.10.18
- PyTorch: 2.5.0+cu124
- CUDA runtime: 12.4
- cuDNN: 9.1.0
- Triton: 3.1.0
- Transformers: 4.47.1
- Precision: bfloat16
- Model weights: deterministic random initialization, seed 42
- Input: deterministic synthetic 224×224 RGB image, text token, proprioception, and explicit initial flow noise
- Flow integration: 10 steps
- GPU deterministic repeat check: exact

The environment uses the pre-fix PyTorch reference implementation from the
companion π0 investigation. It does not use an official policy checkpoint and
therefore validates the numerical model path, not policy quality.

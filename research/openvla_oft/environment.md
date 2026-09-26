# OpenVLA-OFT execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 9.10.2
- Triton: 3.4.0
- transformers: official OpenVLA-OFT fork 4.40.1 at `bc339d9a`
- timm: 0.9.10
- peft: 0.11.1
- diffusers: 0.30.3
- tokenizers: 0.19.1
- OpenVLA-OFT source: `e4287e94541f459edc4feabc4e181f537cd569a8`

The upstream package initializers eagerly import its TensorFlow RLDS training
stack even when only PyTorch inference modules are requested. The reproducible
inference-only patch under `scripts/model_invariance/patches` makes those three
initializers lazy; it does not alter any model, processor, or inference
operation. The upstream checkout was restored to a clean state after the run.

The released environment pins PyTorch 2.2.0. This investigation uses PyTorch
2.8.0 so the same current dispatcher and Triton stack as the operator library
can be exercised on H100. The official custom Transformers fork and model code
remain pinned exactly.

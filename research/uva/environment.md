# UVA execution environment

- GPU: NVIDIA H100 NVL, 99,949,674,496 bytes
- Driver: 580.126.09
- Python: 3.12.3
- PyTorch: 2.14.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- cuDNN: 9.24.0
- Triton: 3.8.0
- transformers: 5.5.4
- hydra-core: 1.3.7
- omegaconf: 2.3.1
- dill: 0.4.1
- zarr: 2.16.1
- numcodecs: 0.11.0

The official source pins an older PyTorch/Transformers environment. Under
Transformers 5.5, `CLIPModel.get_text_features` returns a structured output;
the adapter selects its `pooler_output`. The checkpoint has no missing keys.
Transformers reports only its two obsolete deterministic `position_ids`
buffers as unexpected. Neither compatibility adaptation changes a learned
tensor or model arithmetic.


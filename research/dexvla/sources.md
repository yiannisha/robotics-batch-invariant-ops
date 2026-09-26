# DexVLA source selection

- Official PyTorch implementation: https://github.com/juruobenruo/DexVLA
- Closest complete public checkpoint: https://huggingface.co/kuromivv/DexVLA
- Official example input data: https://huggingface.co/datasets/lesjie/dexvla_example_data

The official repository publishes inference/training code and ScaleDP stage
weights, but no complete public end-to-end checkpoint containing both the VLM
and policy head. The community checkpoint is therefore used strictly as a
weight source. Its official-compatible `config2.json` instantiates the
repository's own `Qwen2VLForConditionalGenerationForVLA` and registered
`ScaleDP` classes. The investigation does not use the checkpoint repository's
custom model implementation and does not translate from JAX or another
framework.


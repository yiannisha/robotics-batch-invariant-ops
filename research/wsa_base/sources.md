# WSA Base sources

- Official PyTorch source: <https://github.com/zaleni/WSA>
- Official WSA Base LIBERO checkpoint: <https://huggingface.co/zaleni/WSA-Base-LIBERO>
- Official Qwen3-VL metadata and processor: <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>
- Official frozen Cosmos tokenizer: <https://huggingface.co/nvidia/Cosmos-Tokenizer-CI8x8>
- Public real LIBERO observations: <https://huggingface.co/datasets/lerobot/libero>

All executed model code and learned weights are native PyTorch. The source's
bundled Transformers 4.57.1 Qwen3-VL replacement was installed exactly as its
README requires. The WSA checkpoint itself contains the complete trained
Qwen3-VL vision/language backbone, generation expert, action expert, policy
projections, and 3D query modules. The separate Qwen snapshot therefore
contains metadata/tokenizer files only.

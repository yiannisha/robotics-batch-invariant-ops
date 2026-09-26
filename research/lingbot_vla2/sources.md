# LingBot-VLA 2.0 source selection

- Official PyTorch implementation: https://github.com/Robbyant/lingbot-vla-v2
- Official RoboTwin checkpoint: https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin
- Official Qwen3-VL base configuration and tokenizer: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Public real three-camera fixture: https://huggingface.co/datasets/lesjie/dexvla_example_data

The model, feature transform, Qwen3-VL integration, action expert, MoE routing,
and ten-step flow sampler all come from the authors' official PyTorch
repository. The released checkpoint loads strictly into those classes without
weight conversion. No JAX implementation or translated weights are used.

The checkpoint contains all learned parameters. Only the small Qwen3-VL
configuration, tokenizer, and image-processor assets are read from the pinned
base-model repository; no separate Qwen weight files are required.

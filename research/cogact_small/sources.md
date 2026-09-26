# CogACT-Small sources

- Official PyTorch repository: https://github.com/microsoft/CogACT
- Pinned CogACT source commit: `b174a1b86deedfab4d198d935207e7bb0527994e`
- Authors' declared OpenVLA dependency: https://github.com/arnoldland/openvla
- Pinned dependency commit: `5603207085d55148682e2a35b868ad77d7b42ece`
- Official public checkpoint: https://huggingface.co/CogACT/CogACT-Small
- Pinned checkpoint revision: `ca302a300542393a0b06e291465076a7d9105c23`
- Construction-only public Llama-2 config/tokenizer mirror:
  https://huggingface.co/NousResearch/Llama-2-7b-hf
- Primary target: the official source's `scripts/aml/test_image.png`, prompt
  `move sponge near apple`, `fractal20220817_data` action statistics, CFG 1.5,
  and ten DDIM steps
- Additional numerical targets: real public LIBERO images already used by the
  other integrations in this repository

The checkpoint contains 7,553,713,991 learned parameter elements: 6,738,939,904
in the Llama backbone, 730,911,680 in DINO+SigLIP, 71,385,600 in the multimodal
projector, and 12,476,807 in DiT-S. The Llama mirror supplies only standard
architecture metadata and tokenizer assets; all learned Llama tensors come
from CogACT-Small.

The two small patches under `scripts/model_invariance/patches` avoid importing
the TensorFlow/RLDS training stack, skip redundant placeholder vision downloads,
use the public construction metadata mirror, and memory-map the trusted official
checkpoint. They do not alter model computation or learned tensors.

# OpenHelix sources

- Official PyTorch repository: https://github.com/OpenHelix-Team/OpenHelix
- Pinned source commit: `d88867fa56977f401a7c89d62b9ebce87d8078f0`
- Official public checkpoint: https://huggingface.co/OpenHelix/openhelix
- Pinned checkpoint revision: `1a6fabbe148af045ed0710d007ed24c36fa83a53`
- Public official-format CALVIN ABC dataset:
  https://huggingface.co/datasets/KyujinL/CALVIN_ABC
- Pinned dataset revision: `0874cd64f0df8dcc3a3534360ca448705e44e954`
- Official CALVIN source: https://github.com/mees/calvin
- Pinned CALVIN commit: `fa03f01f19c65920e18cf37398a9ce859274af76`
- Pinned CALVIN environment commit:
  `1431a46bd36bde5903fb6345e68b5ccc30def666`
- Construction-only LLaVA metadata:
  https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1
- Construction-only CLIP metadata:
  https://huggingface.co/openai/clip-vit-large-patch14

The input adapter uses real RGB, depth, and robot observations from three
public CALVIN transitions. It reconstructs static and gripper point clouds
from the released CALVIN camera calibration and fixed robot transforms. The
fixed instruction is `push the sliding door to the right side`, matching the
task phrase used by the released OpenHelix evaluation code.

`scripts/model_invariance/patches/openhelix_inference_compat.patch` removes
version-pinned eager imports/downloads, constructs the complete checkpoint's
RN50 architecture without placeholder weights, and replaces DGL's unavailable
version-pinned FPS extension with an API-compatible seeded PyTorch algorithm.
It does not alter learned tensors or policy arithmetic.

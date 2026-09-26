# RDT-1B sources

- Official PyTorch repository: https://github.com/thu-ml/RoboticsDiffusionTransformer
- Pinned source commit: `cd79363a1387e8f81c7724d070ef7e45fd23150f`
- Official ManiSkill checkpoint: https://huggingface.co/robotics-diffusion-transformer/maniskill-model
- Pinned checkpoint revision: `9622afab2b7ce2312a6cf1febc526928589b77eb`
- Official vision encoder: https://huggingface.co/google/siglip-so400m-patch14-384
- Pinned vision revision: `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`
- Language conditioning: the checkpoint's official cached
  `text_embed_PickCube-v1.pt`
- Numerical image targets: real public LIBERO observations already used by
  the other integrations in this repository

The policy state dictionary contains 1,228,319,872 learned elements. The
SigLIP vision tower contains 428,225,600 learned elements. The published
ManiSkill inference bundle includes the task's cached T5 embedding, so loading
or approximating a separate text model is unnecessary.

The LIBERO frames are deliberately treated only as real-image numerical
fixtures. They are not ManiSkill observations and no task-success claim is
made from these inputs.

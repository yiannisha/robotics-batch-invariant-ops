# OpenWAM Alpha RoboTwin sources

- Official native PyTorch repository: https://github.com/OpenWAM-Official/OpenWAM
- Official checkpoint: https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full
- Public real three-camera fixture: https://huggingface.co/datasets/lesjie/dexvla_example_data
- Source commit and checkpoint revision are pinned in this directory.

The fixture is a genuine 396-frame ALOHA/RoboTwin HDF5 episode with three
480x640 RGB camera streams and the instruction `Cook rice.` The adapter uses
OpenWAM's released multiview compositor and prompt formatter. The fixture's
14-D joint-space qpos is not the checkpoint's required 20-D EEF
representation, so proprio is a deterministic in-range value derived from the
official checkpoint normalization statistics. This limitation is isolated to
proprio; images, language, weights, preprocessing, and model computation are
real. No JAX source, conversion, random substitute weights, or synthetic model
is used.

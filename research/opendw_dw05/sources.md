# OpenDW DW05 sources

- Official native PyTorch repository: https://github.com/dexmal/OpenDW
- Official RoboTwin checkpoint: https://huggingface.co/Dexmal/DW05-Robotwin
- Source commit and checkpoint revision are pinned in this directory.

The input is a genuine 396-frame ALOHA/RobotWin HDF5 episode with three
480x640 RGB camera streams, 14-D qpos/action records, and the instruction
`Cook rice.` The official `robotwin_resize` layout produces a 384x320 composite.
No JAX source, checkpoint conversion, random substitute weights, or synthetic
model input is used.

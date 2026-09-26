# X-VLA sources

- Maintained PyTorch implementation: <https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/xvla>
- Released LIBERO policy: <https://huggingface.co/lerobot/xvla-libero>
- Released LIBERO observations: <https://huggingface.co/datasets/lerobot/libero>
- X-VLA project repository: <https://github.com/2toinf/X-VLA>

The numerical investigation runs LeRobot's `XVLAPolicy`, released processor,
Florence-2 vision/language encoder, 24-layer soft-prompted action transformer,
ten flow updates, ee6d action-space postprocessing, and official LIBERO 6D
rotation conversion. All repositories and snapshots are pinned in
`model_commit.txt` and `checkpoint.txt`. No JAX implementation or converted
checkpoint is used.

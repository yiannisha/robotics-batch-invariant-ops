# Model discovery audit — 2026-09-25

This audit records public candidates; it is not a support table. A model moves
to the README support table only after end-to-end execution.

## Tier 1 updates

- **OpenPI:** the current official repository exposes π0 and π0.5 PyTorch
  implementations. Its README still lists π0-FAST as unsupported by the
  PyTorch path, so π0-FAST needs the JAX implementation or a separate PyTorch
  implementation. Public base and robot-specific checkpoints are listed in
  the repository. Next target: official π0.5-DROID or π0.5-LIBERO after JAX to
  PyTorch conversion.
- **DreamZero:** the official repository now publishes DreamZero-DROID and
  DreamZero-AgiBot checkpoints, local distributed inference, DiT caching, and
  reports approximately three-second inference on H100. This is runnable in
  scope and remains the next distinct world-action architecture.
- **InternVLA-A:** InternVLA-A1.5 supersedes A1. Its public project links code
  and weights. The inference graph uses a VLM plus lightweight continuous
  action expert; latent foresight is training-only in A1.5, so the plan's
  foresight/action-only inference ablation applies to A1 rather than A1.5.
- **MolmoAct:** MolmoAct2 supersedes the original MolmoAct release. Public base,
  DROID, BimanualYAM, SO100/101, LIBERO, and Think-LIBERO checkpoints and a
  LeRobot inference implementation are available. Prioritize MolmoAct2-LIBERO
  for a contained continuous flow-matching test.
- **GR00T:** N1.7 supersedes N1.6. The official 3B base model has a PyTorch
  inference command and an approximately 6 GB checkpoint; DROID and LIBERO
  fine-tuned checkpoints are also public. The Cosmos-Reason2-2B backbone is
  gated, so Hugging Face authorization is an external prerequisite.
- **Cosmos:** Cosmos 3 publishes Nano-Policy-DROID and Edge-Policy-DROID action
  policies. The Edge server command is public; the documentation also exposes
  WAM inference that returns both actions and future visual rollout. Nano is
  the preferred first single-H100 target, followed by Edge if memory permits.

## Tier 2 and newly discovered candidates

- **UVA:** official PyTorch code and PushT, PushT-M, LIBERO10, and UMI
  checkpoints are public. Both video and action outputs are in scope.
- **X-VLA:** official code exists and X-VLA is integrated into LeRobot with a
  public `lerobot/xvla-libero` checkpoint.
- **DeVA:** the July 2026 paper is discoverable, but no verified official public
  inference repository or weights were found in this audit. Status: BLOCKED
  pending a public release.
- **LingBot-VLA 2.0:** newly released cross-embodiment 6B VLA with public code
  and weight collections; it meets the discovery criteria and should be added
  after the named Tier 1 models.
- **UniVLA:** public unified image-grounding/video/action repository; candidate
  after UVA because it broadens the video/action coverage.

## Primary sources

- https://github.com/Physical-Intelligence/openpi
- https://github.com/dreamzero0/dreamzero
- https://github.com/InternRobotics/InternVLA-A1.5
- https://github.com/allenai/molmoact2
- https://github.com/NVIDIA/Isaac-GR00T
- https://github.com/NVIDIA/cosmos-framework
- https://github.com/ShuangLI59/unified_video_action
- https://github.com/2toinf/X-VLA
- https://github.com/Robbyant/lingbot-vla-v2
- https://github.com/baaivision/UniVLA

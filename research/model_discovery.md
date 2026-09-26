# Model discovery audit — 2026-09-26

This audit records public candidates; it is not a support table. A model moves
to the README support table only after end-to-end execution.

## Tier 1 updates

- **OpenPI:** the current official repository exposes π0 and π0.5 PyTorch
  implementations but still lists π0-FAST as unsupported by that path. The
  maintained LeRobot repository now includes a direct PyTorch `pi0_fast` port
  and public `lerobot/pi0fast-libero` PyTorch weights. That implementation has
  now been executed through B=64 and added to the support table. No JAX code or
  checkpoint conversion is needed or in scope.
- **DreamZero:** the official repository now publishes DreamZero-DROID and
  DreamZero-AgiBot checkpoints, local distributed inference, and DiT caching.
  The released 14B inference path requires at least two GPUs and its PyTorch
  checkpoint is 45.85 GB. A single-process Wan2.2 5B server exists, but no
  trained public DreamZero checkpoint is released for it. Status: BLOCKED
  under the single-H100 constraint; see `research/dreamzero/blocked.md`.
- **InternVLA-A:** InternVLA-A1.5 supersedes A1. Its public project links code
  and weights. The inference graph uses a VLM plus lightweight continuous
  action expert; latent foresight is training-only in A1.5, so the plan's
  foresight/action-only inference ablation applies to A1 rather than A1.5. The
  official A1.5 PyTorch standard and optimized action paths have now been
  executed with public LIBERO weights and real inputs through B=64; both are in
  the support table. No video-output result is claimed.
- **MolmoAct:** MolmoAct2 supersedes the original MolmoAct release. Public base,
  DROID, BimanualYAM, SO100/101, LIBERO, and Think-LIBERO checkpoints and a
  PyTorch Transformers implementation are available. The official
  MolmoAct2-LIBERO continuous flow-matching path has now been executed with
  public weights and real inputs through B=64 and added to the support table.
- **GR00T:** N1.7 supersedes N1.6. The official 3B base model has a PyTorch
  inference command and an approximately 6 GB checkpoint; DROID and LIBERO
  fine-tuned checkpoints are also public. The official N1.7 LIBERO-10 path has
  now been executed with public weights and real inputs through B=64 and added
  to the support table. Although the constructor resolves the gated
  Cosmos-Reason2-2B base first, the public GR00T shards contain the complete
  backbone. The investigation uses the exact public Qwen3-VL architecture and
  processor assets only for construction, then loads every model tensor from
  NVIDIA's checkpoint; no substitute weights or conversion are involved.
- **Cosmos:** Cosmos 3 publishes Nano-Policy-DROID and Edge-Policy-DROID action
  policies. The Edge server command is public; the documentation also exposes
  WAM inference that returns both actions and future visual rollout. Nano is
  the preferred first single-H100 target, followed by Edge if memory permits.

## Tier 2 and newly discovered candidates

- **UVA:** official PyTorch code and PushT, PushT-M, LIBERO10, and UMI
  checkpoints are public. Its action-only path, joint video/action latent path,
  and official RGB decoder have now been executed and added to the support
  table.
- **X-VLA:** official code exists and X-VLA is integrated into LeRobot with a
  public `lerobot/xvla-libero` checkpoint. The maintained PyTorch path has now
  been executed through B=64 and added to the support table.
- **SmolVLA:** the maintained LeRobot PyTorch implementation and public
  `HuggingFaceVLA/smolvla_libero` checkpoint are runnable directly. The
  ten-step LIBERO action path has now been executed through B=64 and added to
  the support table.
- **DeVA:** official PyTorch code and post-training weights are public, but the
  checkpoint depends on separately gated Cosmos-Predict2 base assets that were
  unavailable in this environment. Status: access-blocked; see
  `research/deva/constraint_audit.md`.
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
- https://github.com/huggingface/lerobot
- https://huggingface.co/HuggingFaceVLA/smolvla_libero
- https://github.com/Robbyant/lingbot-vla-v2
- https://github.com/baaivision/UniVLA

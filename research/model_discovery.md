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
- **OpenVLA-OFT:** the authors' official PyTorch implementation and complete
  public LIBERO-Spatial checkpoint are runnable on one H100. The dual-camera
  continuous-action path has now been executed through B=64 and added to the
  support table. The upstream batch-1-only output reshape is handled by a
  semantics-preserving batched adapter; no JAX code or checkpoint conversion
  is involved.
- **DexVLA:** the official PyTorch classes have now been executed with the
  closest public complete checkpoint and official three-camera example data.
  Stock fails every completed batch above one; the repaired reasoning, FiLM,
  and ten-step ScaleDP path is exact through B=33, with B=64 OOM. Because the
  complete checkpoint is community-published rather than author-published,
  that limitation is retained in the support row and evidence bundle.
- **DeVA:** official PyTorch code and post-training weights are public, but the
  checkpoint depends on separately gated Cosmos-Predict2 base assets that were
  unavailable in this environment. Status: access-blocked; see
  `research/deva/constraint_audit.md`.
- **LingBot-VLA 2.0:** the official PyTorch implementation and official 6B
  RoboTwin checkpoint have now been executed in the released FP32 inference
  mode. After replacing the released nondeterministic atomic MoE inference
  kernel with the reusable deterministic semantic fallback, stock inference
  fails every batch above one and invariant addmm/MM/BMM makes all ten flow
  steps exact through B=64. It is now in the support table.
- **UniVLA:** the official BAAI PyTorch LIBERO image-policy checkpoint, Emu3
  vision tokenizer, and FAST action tokenizer have now been executed through
  B=33. The released actions happen to remain stable in stock mode, but its
  cached-decode logits fail; invariant MM repairs every score and action. B=64
  OOMs. It is now in the support table.
- **SpatialVLA:** the official PyTorch 4B 224-pixel checkpoint and published
  example have now been executed directly through B=64. Stock changes the
  final action; reusable convolution, mean, MM, and BMM repairs make every
  tested score and action exact. The ZoeDepth path also motivated generic
  ConvTranspose2D and direct arbitrary-dimension softmax dispatch coverage.
  It is now in the support table.
- **RDT-1B:** the official PyTorch implementation, official ManiSkill policy,
  official SigLIP tower, and checkpoint-published PickCube task embedding have
  now been executed through B=64. Stock fails every non-unit policy batch; the
  first boundary is long image cross-attention SDPA, with a second addmm
  threshold at B=64. Generic SDPA and addmm repairs make all five diffusion
  steps and the complete action exact. It is now in the support table.
- **CogACT:** the official PyTorch `CogACT-Small` checkpoint, bundled example,
  and released batch inference path have now been executed through B=64.
  Stock fails every non-unit batch; invariant SDPA repairs DINO attention and
  invariant MM repairs the later Llama MLP boundary. All ten DDIM steps and
  final actions are exact after repair. It is now in the support table.
- **OpenHelix:** the official PyTorch planner and diffusion policy, complete
  public `prompt_tuning_aux` checkpoint, and public CALVIN ABC observations
  have now been executed through B=64. Existing BMM repair fixes the first
  CLIP attention boundary; the model additionally exposed direct rank-3
  `aten::linear`, now covered by a generic invariant implementation. The full
  25-step action path is in the support table.
- **WSA:** the official PyTorch WSA Base causal policy and complete public
  LIBERO checkpoint have now been executed through B=64 with real inputs. The
  checkpoint contains its full learned Qwen3-VL/action stack; no conversion is
  involved. Stock fails every B above one. Generic rank-3 linear and fixed-tree
  mean repairs make all ten flow steps and final actions exact, so it is now in
  the support table.
- **VITRA:** Microsoft's official native PyTorch implementation and complete
  public 3B checkpoint (`VITRA-VLA/VITRA-VLA-3B`, 15.07 GB) have now been
  executed through B=64 using all three released real images. Stock fails
  every B above one; the first boundary is the FP32 FOV encoder's second
  `aten::linear`. The existing fixed-schedule linear repair makes the VLM, all
  ten DDIM states, and final action exact. It is now in the support table.
- **OpenDW / DW05:** Dexmal publishes native PyTorch world-action code and a
  public RoboTwin checkpoint with both action and video components. The bundled
  policy, text encoder, and VAE total approximately 26 GB. Status: runnable
  candidate, not yet a support claim.
- **OpenWAM:** the authors publish native PyTorch code and public checkpoints
  for the 2026-09 world-action release. Status: runnable candidate, not yet a
  support claim.
- **G0.5:** official PyTorch source and a public model page exist, but the
  checkpoint requires accepting a contact-information agreement unavailable
  in this environment. Status: access-blocked until credentials are supplied.
- **PoseVLA:** the public route advertises a JAX-to-PyTorch conversion script
  rather than a native PyTorch release. It is out of scope under the explicit
  PyTorch-only rule unless a native checkpoint is published.

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
- https://github.com/moojink/openvla-oft
- https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial
- https://github.com/juruobenruo/DexVLA
- https://github.com/Robbyant/lingbot-vla-v2
- https://github.com/baaivision/UniVLA
- https://github.com/SpatialVLA/SpatialVLA
- https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt
- https://github.com/thu-ml/RoboticsDiffusionTransformer
- https://huggingface.co/robotics-diffusion-transformer/maniskill-model
- https://github.com/microsoft/CogACT
- https://github.com/OpenHelix-Team/OpenHelix
- https://github.com/zaleni/WSA
- https://github.com/microsoft/VITRA
- https://github.com/dexmal/OpenDW
- https://github.com/OpenWAM-Official/OpenWAM
- https://huggingface.co/OpenGalaxea/G05

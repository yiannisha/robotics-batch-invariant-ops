# OpenDW DW05 batch-invariance investigation

## Scope and checkpoint audit

This uses Dexmal's official native PyTorch OpenDW implementation and complete
public `DW05-Robotwin` bundle. The strict audit matches all 1,649 MoT keys
(6,020,688,078 elements) and both proprio-encoder keys. The complete runtime
also loads the official 5.681B-parameter text encoder and 704.7M-parameter VAE.
The learned action/proprio width is 14 despite a stale 32-D value in the
top-level metadata.

The released public methods enforce B=1 and allocate noise internally. The
adapters preserve the same VAE, text encoder, MoT, ActionDiT, schedulers, five
flow steps, normalization, and decoder while making the two CPU-seeded noise
tensors explicit per-sample inputs. Both action-only and joint adapters are
bitwise identical to the released B=1 methods.

## Action-only result and first divergence

Repeated stock B=1 calls are exact. Stock fails every B above one through 64
in both duplicate and unrelated compositions; B=2 changes 183 of 448 final
float32 values with maximum absolute difference 0.0080007315.

All video-conditioning inputs, projections, Conv3d patch tokens, and the full
30-layer cached video branch remain exact. The first mismatch is
`action_expert.action_encoder`, an `aten::linear` with input `[B,32,14]`,
weight `[1024,14]`, and bias `[1024]`. At B=2, 9,096 BF16 output elements
differ (maximum 0.015625). CUDA profiling shows a CUTLASS BF16 16x16 WMMA GEMM
for flattened M=32 at B=1 and a 32x32 WMMA GEMM for M=64 at B=2. The changed
tile/reduction schedule changes the unchanged target rows.

Enabling only the reusable invariant linear implementation repairs every
captured boundary, all five action-flow states, and the final action. Full
library mode is exact for all required sizes 1, 2, 3, 4, 5, 7, 8, 9, 15, 16,
17, 31, 32, 33, and 64 in both compositions. Three distinct real episode
frames additionally pass duplicate/unrelated B=2 and B=4 checks.

## Joint video/action result

The joint test generates nine 384x320 frames and a 32x14 action trajectory.
Stock B=2 changes 268 action values (maximum 0.0164108574) and 810,210 decoded
uint8 video values. Full library mode makes both the action and the complete
decoded video bitwise exact at B=2 for duplicate and unrelated compositions.

## Performance

For the actual action-encoder shape, invariant linear latency is 3.72x, 3.22x,
3.50x, and 3.60x stock at B=1, 2, 8, and 64. Absolute latency is 0.039-0.047 ms.
Measured incremental memory is lower than stock at every measured size.

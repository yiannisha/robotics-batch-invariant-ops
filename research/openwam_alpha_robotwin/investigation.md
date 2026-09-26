# OpenWAM Alpha RoboTwin batch-invariance investigation

## Scope and checkpoint audit

This investigation uses the official native PyTorch OpenWAM repository and
the complete public `OpenWAM-Alpha-Sim-RoboTwin-Full` checkpoint. OpenWAM's
strict loader accepted all 2,089 tensors and 12,406,754,092 learned elements.
The model is a dual-system joint-self-attention architecture with a Wan2.2
TI2V 5B video expert and an ActionDiT expert. It jointly denoises a
`[B,48,3,24,20]` video latent and a 32-step, 80-D unified action trajectory,
then maps the latter back to 20-D physical EEF actions.

The released deploy method constructs B=1 action noise and squeezes the action
batch dimension. The adapter leaves the released preprocessing, text encoder,
VAE, two experts, schedulers, ten synchronous flow steps, action
normalization, and decoder unchanged, while making per-sample noise and batch
dimensions explicit. At B=1 its 640 returned float32 action values are
bitwise identical to `JointInferenceEngine.generate`.

The input uses real three-camera RoboTwin frames and the real task prompt. The
public fixture stores 14-D joint qpos, while this checkpoint consumes 20-D EEF
state; deterministic checkpoint-space states derived from its normalization
statistics are therefore used. This is a numerical inference test, not a
policy-quality claim.

## Stock result and first divergence

Repeated stock B=1 calls are exact. Stock fails every B above one through 64
for both duplicate and unrelated compositions. At B=2 it changes 273 of 640
physical action values (maximum absolute difference 0.0078125) and 31,721
final video-latent values (maximum 0.078125). Decoding with the released
Wan2.2 VAE changes 617,410 uint8 pixel values.

All inputs, embeddings, Conv3d patch tokens, action projections, and all 30
Mixture-of-Transformers layer boundaries are exact on the first denoising
step. The first mismatch is the video expert's final
`video_backbone.dit.head.head` `aten::linear`. Its input `[B,360,3072]` is
identical, but at B=2 its `[B,360,192]` BF16 output changes 118 values with
maximum absolute difference 0.015625. CUDA profiling shows that flattened
M=360 at B=1 selects a cuBLASLt split-K kernel plus split-K reduction, while
M=720 at B=2 selects the corresponding non-split-K kernel. The video error is
fed back through later joint denoising steps and eventually changes actions as
well.

## Repair and coverage

Convolution-only repair still fails. Enabling only invariant linear is
sufficient to make every traced boundary and both final outputs exact;
convolution-plus-linear and full-library modes are also exact. Full-library
mode passes duplicate and unrelated composition at sizes 1, 2, 3, 4, 5, 7,
8, 9, 15, 16, 17, 31, 32, 33, and 64. Three distinct real episode frames
also pass B=2 and B=4 checks, and released-VAE decoded uint8 video is exact at
B=2.

## Performance

For the actual `[B,360,3072]` BF16 video-head input, invariant linear is
2.60x, 3.10x, 3.20x, and 2.26x stock latency at B=1, 2, 8, and 64. Absolute
latency is 0.048-0.163 ms. Measured incremental memory is lower than stock at
every measured size.

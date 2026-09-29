"""Official PyTorch UVA LIBERO-10 adapter with explicit stochastic inputs.

Required environment variables:

``UVA_LIBERO_CHECKPOINT``
    Official ``libero10.ckpt`` released by the UVA authors.
``UVA_LIBERO_SAMPLE_DIR``
    Fixture containing 64 real LIBERO episode-zero camera frames and
    ``tasks.parquet``.

UVA samples both its VAE posterior and its 100-step action diffusion process.
This adapter follows the released inference body while treating every random
draw as a per-example input, so batch-shaped RNG cannot be confused with
numerical batch dependence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import pyarrow.parquet as pq
import torch
from einops import rearrange
from omegaconf import open_dict
from PIL import Image


class UVALibero10Adapter:
    name = "uva-libero10-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("UVA_LIBERO_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("UVA_LIBERO_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.config = None
        self.num_diffusion_steps = None
        self.num_video_diffusion_steps = None
        self.task_mode = "policy_model"
        self.include_video_output = False
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError("set UVA_LIBERO_CHECKPOINT to the official libero10.ckpt")
        for filename in (
            "frames/image_001.png",
            "frames/image_064.png",
            "tasks.parquet",
        ):
            if not (self.sample_dir / filename).is_file():
                raise FileNotFoundError(
                    f"set UVA_LIBERO_SAMPLE_DIR to a directory containing {filename}"
                )

    def load_model(self):
        self._validate_paths()
        payload = torch.load(
            self.checkpoint,
            map_location="cpu",
            pickle_module=dill,
            weights_only=False,
        )
        config = payload["cfg"]
        with open_dict(config):
            # The released EMA checkpoint contains both dependencies in full.
            # Avoid loading redundant pretraining files before applying it.
            config.model.policy.vae_model_params.autoencoder_path = None
            config.model.policy.autoregressive_model_params.pretrained_model_path = None
        policy = hydra.utils.instantiate(
            config.model.policy,
            task_name=config.task.name,
            task_modes=config.task.task_modes,
            normalizer_type=config.task.dataset.normalizer_type,
            language_emb_model=config.task.dataset.language_emb_model,
        )
        incompatible = policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=False)
        allowed_unexpected = {
            "text_model.text_model.embeddings.position_ids",
            "text_model.vision_model.embeddings.position_ids",
        }
        if incompatible.missing_keys or set(incompatible.unexpected_keys) != allowed_unexpected:
            raise RuntimeError(f"unexpected UVA checkpoint incompatibility: {incompatible}")
        self.config = config
        self.num_diffusion_steps = policy.model.diffactloss.gen_diffusion.num_timesteps
        self.num_video_diffusion_steps = policy.model.diffloss.gen_diffusion.num_timesteps
        self._policy_tokenizer = policy.tokenizer
        policy.to(self.device).eval()
        return policy

    def _image_window(self, start: int) -> torch.Tensor:
        frames = []
        for offset in range(16):
            frame_number = start + offset + 1
            array = np.array(
                Image.open(self.sample_dir / "frames" / f"image_{frame_number:03d}.png").convert(
                    "RGB"
                ),
                copy=True,
            )
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(frames)

    def load_example_inputs(self, count: int):
        if self.config is None or self.num_diffusion_steps is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 contexts")

        target_frame = int(os.environ.get("UVA_TARGET_FRAME", "0"))
        if not 0 <= target_frame <= 48:
            raise ValueError("UVA_TARGET_FRAME must be between 0 and 48")
        starts = [target_frame]
        starts.extend(index for index in range(49) if index != target_frame)
        starts.extend(index for index in range(15))

        task_rows = pq.read_table(self.sample_dir / "tasks.parquet").to_pylist()
        task = task_rows[0]["__index_level_0__"]
        tokens = self._tokenize(task)

        examples = []
        for example_index, start in enumerate(starts[:count]):
            generator = torch.Generator(device="cpu").manual_seed(88_000 + start)
            examples.append(
                {
                    "images": self._image_window(start).unsqueeze(0),
                    "input_ids": tokens["input_ids"].clone(),
                    "attention_mask": tokens["attention_mask"].clone(),
                    "vae_noise": torch.randn(
                        1, 4, 16, 16, 16, dtype=torch.float32, generator=generator
                    ),
                    "action_initial_noise": torch.randn(
                        1, 16, 10, dtype=torch.float32, generator=generator
                    ),
                    "action_step_noise": torch.randn(
                        1,
                        self.num_diffusion_steps,
                        16,
                        10,
                        dtype=torch.float32,
                        generator=generator,
                    ),
                    "context_index": torch.tensor([[example_index]], dtype=torch.int64),
                }
            )
            if self.include_video_output:
                examples[-1]["video_initial_noise"] = torch.randn(
                    1, 1_024, 16, dtype=torch.float32, generator=generator
                )
                examples[-1]["video_step_noise"] = torch.randn(
                    1,
                    self.num_video_diffusion_steps,
                    1_024,
                    16,
                    dtype=torch.float32,
                    generator=generator,
                )
        return examples

    def _tokenize(self, task: str) -> dict[str, torch.Tensor]:
        policy_tokenizer = getattr(self, "_policy_tokenizer", None)
        if policy_tokenizer is None:
            raise RuntimeError("tokenizer is initialized by load_example_inputs wrapper")
        return policy_tokenizer(
            [task],
            padding="max_length",
            max_length=30,
            return_tensors="pt",
        )

    @staticmethod
    def compose_batch(samples):
        return {key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]}

    @staticmethod
    def reset_model(_model) -> None:
        return None

    def _explicit_action_sample(self, policy, z: torch.Tensor, batch: dict[str, torch.Tensor]):
        diffusion_head = policy.model.diffactloss
        if diffusion_head.act_model_type != "conv_ori":
            raise ValueError(f"unsupported UVA action head: {diffusion_head.act_model_type}")

        features = rearrange(z, "b (t s) c -> b t s c", t=diffusion_head.n_frames)
        features = rearrange(features, "b t (w h) c -> b c t w h", w=diffusion_head.w)
        self._trace("action.conv_transpose3d.input", features)
        features = diffusion_head.conv_transpose3d(features)
        self._trace("action.conv_transpose3d.output", features)
        features = diffusion_head.avg_pool(features)
        features = rearrange(features, "b c t w h -> b (t w h) c")
        self._trace("action.conditioning", features)
        batch_size, sequence_length, _ = features.shape
        conditioning = features.reshape(batch_size * sequence_length, -1)

        sample = (
            batch["action_initial_noise"]
            .to(self.device)
            .reshape(batch_size * sequence_length, diffusion_head.in_channels)
        )
        step_noises = batch["action_step_noise"].to(self.device)
        self._trace("action.diffusion.initial", sample.reshape(batch_size, sequence_length, -1))
        diffusion = diffusion_head.gen_diffusion
        temperature = float(self.config.model.policy.autoregressive_model_params.temperature)
        for step_index, timestep in enumerate(range(diffusion.num_timesteps - 1, -1, -1)):
            timesteps = torch.full(
                (sample.shape[0],), timestep, dtype=torch.long, device=self.device
            )
            result = diffusion.p_mean_variance(
                diffusion_head.net.forward,
                sample,
                timesteps,
                clip_denoised=True,
                model_kwargs={"c": conditioning},
            )
            noise = step_noises[:, step_index].reshape_as(sample)
            nonzero = (timesteps != 0).float().view(-1, 1)
            sample = result["mean"] + (
                nonzero * torch.exp(0.5 * result["log_variance"]) * noise * temperature
            )
            if step_index in (0, diffusion.num_timesteps // 2, diffusion.num_timesteps - 1):
                self._trace(
                    f"action.diffusion.step_{step_index:03d}",
                    sample.reshape(batch_size, sequence_length, -1),
                )
        return sample.reshape(batch_size, sequence_length, -1)

    def _explicit_video_sample(self, policy, z: torch.Tensor, batch: dict[str, torch.Tensor]):
        diffusion_head = policy.model.diffloss
        batch_size = batch["images"].shape[0]
        sequence_length = z.shape[0] // batch_size
        if sequence_length != 1_024:
            raise ValueError(f"expected 1024 UVA video tokens, got {sequence_length}")
        sample = (
            batch["video_initial_noise"]
            .to(self.device)
            .reshape(batch_size * sequence_length, diffusion_head.in_channels)
        )
        step_noises = batch["video_step_noise"].to(self.device)
        diffusion = diffusion_head.gen_diffusion
        temperature = float(self.config.model.policy.autoregressive_model_params.temperature)
        for step_index, timestep in enumerate(range(diffusion.num_timesteps - 1, -1, -1)):
            timesteps = torch.full(
                (sample.shape[0],), timestep, dtype=torch.long, device=self.device
            )
            result = diffusion.p_mean_variance(
                diffusion_head.net.forward,
                sample,
                timesteps,
                clip_denoised=False,
                model_kwargs={"c": z},
            )
            noise = step_noises[:, step_index].reshape_as(sample)
            nonzero = (timesteps != 0).float().view(-1, 1)
            sample = result["mean"] + (
                nonzero * torch.exp(0.5 * result["log_variance"]) * noise * temperature
            )
        return sample

    def run_model(self, policy, batch):
        from unified_video_action.utils.data_utils import (
            process_data,
            unnormalize_future_action,
        )

        images = batch["images"].to(self.device, dtype=torch.float32).div(255)
        self._trace("preprocess.images", images)
        text_inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
        }
        text_features = policy.text_model.get_text_features(**text_inputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
        self._trace("text.features", text_features)

        normalized_images, proprioception_input, _ = process_data(
            {"obs": {"image": images}},
            task_name=policy.task_name,
            eval=True,
            **policy.kwargs,
        )
        self._trace("preprocess.normalized_images", normalized_images)
        flattened = rearrange(normalized_images, "b c t h w -> (b t) c h w").float()
        posterior = policy.vae_model.encode(flattened)
        self._trace("vae.posterior.mean", posterior.mean)
        self._trace("vae.posterior.std", posterior.std)
        vae_noise = batch["vae_noise"].to(self.device).reshape_as(posterior.mean)
        latent = (posterior.mean + posterior.std * vae_noise).mul(0.2325)
        latent = rearrange(latent, "(b t) c h w -> b t c h w", b=images.shape[0])
        self._trace("vae.latent", latent)

        original_sample = policy.model.diffactloss.sample
        original_video_sample = policy.model.diffloss.sample

        def explicit_sample(z, temperature=1.0, cfg=1.0, text_latents=None):
            if cfg != 1.0:
                raise ValueError("the released LIBERO checkpoint uses action CFG 1.0")
            return self._explicit_action_sample(policy, z, batch)

        def explicit_video_sample(z, temperature=1.0, cfg=1.0, text_latents=None):
            if float(cfg) != 1.0:
                raise ValueError("the released LIBERO checkpoint uses video CFG 1.0")
            return self._explicit_video_sample(policy, z, batch)

        policy.model.diffactloss.sample = explicit_sample
        if self.include_video_output:
            policy.model.diffloss.sample = explicit_video_sample
        try:
            video_latent, normalized_action = policy.model.sample_tokens(
                bsz=images.shape[0],
                cond=latent,
                text_latents=text_features,
                num_iter=self.config.model.policy.autoregressive_model_params.num_iter,
                cfg=self.config.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=self.config.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=self.config.model.policy.autoregressive_model_params.temperature,
                history_nactions=None,
                proprioception_input=proprioception_input,
                task_mode=self.task_mode,
                vae_model=policy.vae_model,
            )
        finally:
            policy.model.diffactloss.sample = original_sample
            policy.model.diffloss.sample = original_video_sample

        action = unnormalize_future_action(
            normalizer=policy.normalizer,
            normalizer_type=policy.normalizer_type,
            actions=normalized_action[..., : policy.action_dim],
        )
        self._trace("action.normalized", normalized_action)
        self._trace("action.unnormalized", action)
        action = action[:, : policy.n_action_steps]
        if not self.include_video_output:
            return action
        video_latent = rearrange(video_latent, "(b t) c h w -> b t c h w", b=images.shape[0])
        decoded = policy.vae_model.decode(
            rearrange(video_latent, "b t c h w -> (b t) c h w") / 0.2325
        )
        decoded = decoded.clamp(-1, 1)
        video = 1 + rearrange(decoded, "(b t) c h w -> b t h w c", b=images.shape[0])
        video = (video * 127.5).to(torch.uint8)
        return {"actions": action, "video": video, "video_latent": video_latent}

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = UVALibero10Adapter()

"""Official OpenWAM-Alpha RoboTwin native-PyTorch adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image


class OpenWAMAlphaRobotwinAdapter:
    name = "openwam-alpha-robotwin-official-pytorch-joint-latent-action"

    def __init__(self) -> None:
        self.source = Path(os.environ.get("OPENWAM_SOURCE", ""))
        self.checkpoint = Path(os.environ.get("OPENWAM_CHECKPOINT", ""))
        self.sample = Path(os.environ.get("OPENWAM_SAMPLE", ""))
        self.native_deps = Path(
            os.environ.get(
                "OPENWAM_NATIVE_DEPS",
                "/workspace/batch-invariant-workdir/openwam-pydeps-native",
            )
        )
        self.python_deps = Path(os.environ.get("OPENWAM_PYDEPS", "/dev/shm/openwam/pydeps"))
        self.device = torch.device("cuda")
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _add_paths(self) -> None:
        for path in (self.source, self.python_deps, self.native_deps):
            value = str(path)
            if value in sys.path:
                sys.path.remove(value)
            sys.path.insert(0, value)

    def _validate_paths(self) -> None:
        required = (
            self.source / "openwam/deploy/model_loader.py",
            self.checkpoint / "config.yaml",
            self.checkpoint / "checkpoint_step_118655.safetensors",
            self.checkpoint / "normalization_stats.npy",
            self.sample,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing OpenWAM artifacts: {missing}")

    def _raw_state(self, frame_index: int) -> np.ndarray:
        stats = np.load(self.checkpoint / "normalization_stats.npy", allow_pickle=True).item()[
            "eef"
        ]
        # This public image episode stores joint-space qpos rather than the
        # checkpoint's 20-D EEF representation. Use deterministic, in-range
        # checkpoint-space states while retaining genuine camera observations.
        offset = (frame_index % 7 - 3) * 0.025
        state = stats["mean"] + offset * stats["std"]
        return np.clip(state, stats["q01"], stats["q99"]).astype(np.float32)

    def load_example_inputs(self, count: int):
        model = getattr(self, "_loaded_model", None)
        if model is None:
            raise RuntimeError("adapter model reference was not initialized")
        from openwam.dataloader.transforms.multiview import format_prompt_for_inference

        cfg = model._batch_invariance_cfg
        preprocessor = model._batch_invariance_preprocessor
        action_num_frames = int(cfg.inference.num_frames)
        examples = []
        with h5py.File(self.sample, "r") as episode:
            frame_count = episode["observations/qpos"].shape[0]
            task = episode["language_raw"][0].decode()
            prompt = format_prompt_for_inference(task)
            for frame_index in range(count):
                if frame_index >= frame_count:
                    raise ValueError(f"fixture contains only {frame_count} frames")
                images = {
                    "head_camera": Image.fromarray(
                        np.array(episode["observations/images/cam_high"][frame_index])
                    ),
                    "left_wrist_camera": Image.fromarray(
                        np.array(episode["observations/images/cam_left_wrist"][frame_index])
                    ),
                    "right_wrist_camera": Image.fromarray(
                        np.array(episode["observations/images/cam_right_wrist"][frame_index])
                    ),
                }
                raw_state = self._raw_state(frame_index)
                obs = preprocessor.preprocess(
                    {"images": images, "prompt": prompt, "state": raw_state}
                )
                seed = 42 + frame_index
                pipeline = model.video_backbone.preprocess_input_for_inference(
                    prompt=obs["prompt"],
                    first_frame_image=[obs["image"]],
                    num_frames=int(cfg.inference.video_num_frames),
                    height=int(cfg.inference.height),
                    width=int(cfg.inference.width),
                    seed=seed,
                    num_inference_steps=10,
                    shift=model.action_backbone.shift_action,
                    tiled=False,
                )
                reference = pipeline.get("first_frame_latents")
                if reference is not None:
                    latents = pipeline["latents"].clone()
                    latents[:, :, : reference.shape[2]] = reference
                    pipeline["latents"] = latents
                    pipeline["noise"] = latents
                action_noise = torch.randn(
                    1,
                    action_num_frames - 1,
                    model.action_dim,
                    device=self.device,
                    dtype=model.dtype,
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                )
                normalized_state = model.normalize_deploy_proprio(raw_state)
                examples.append(
                    {
                        "pipeline": {
                            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                            for key, value in pipeline.items()
                        },
                        "action_noise": action_noise[0].cpu(),
                        "proprio": normalized_state.cpu(),
                        "seed": seed,
                        "official_conditions": {
                            "prompt": obs["prompt"],
                            "first_frame_image": [obs["image"]],
                            "num_frames": action_num_frames,
                            "video_num_frames": int(cfg.inference.video_num_frames),
                            "height": int(cfg.inference.height),
                            "width": int(cfg.inference.width),
                            "seed": seed,
                            "tiled": False,
                            "denoise_steps": 10,
                            "proprio": raw_state,
                        },
                    }
                )

        if os.environ.get("OPENWAM_KEEP_PREPROCESSORS", "0") != "1":
            model.video_backbone.text_encoder.to("cpu")
            model.video_backbone.vae.to("cpu")
            torch.cuda.empty_cache()
        return examples

    def load_model(self):
        model = self._load_model()
        self._loaded_model = model
        return model

    def _load_model(self):
        self._validate_paths()
        self._add_paths()
        from omegaconf import OmegaConf
        from openwam.deploy.denoise_schedule import make_schedule
        from openwam.deploy.engine import JointInferenceEngine
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.obs_preprocess import ObsPreprocessor
        from openwam.deploy.server import merge_deploy_cfg

        training_cfg, model = load_from_checkpoint_dir(
            str(self.checkpoint), device=str(self.device)
        )
        deploy_cfg = OmegaConf.load(self.source / "configs/deploy.yaml")
        deploy_cfg.optimization.compile.enabled = False
        deploy_cfg.optimization.dit_cache.enabled = False
        deploy_cfg.optimization.decode_video = False
        deploy_cfg.inference.denoise_steps = 10
        cfg = merge_deploy_cfg(training_cfg, deploy_cfg)
        engine = JointInferenceEngine(cfg=cfg, architecture=model)
        model._batch_invariance_cfg = cfg
        model._batch_invariance_engine = engine
        model._batch_invariance_preprocessor = ObsPreprocessor.from_cfg(cfg, engine)
        model._batch_invariance_schedule = make_schedule(
            "sync",
            video_scheduler=model.video_scheduler,
            action_scheduler=model.action_scheduler,
            num_steps=10,
            shift=model.action_backbone.shift_action,
            shift_video=model.video_backbone.shift_video,
        )
        return model.eval()

    @staticmethod
    def compose_batch(samples):
        tensor_keys = {
            key
            for sample in samples
            for key, value in sample["pipeline"].items()
            if isinstance(value, torch.Tensor)
        }
        pipeline = {}
        for key in samples[0]["pipeline"]:
            values = [sample["pipeline"][key] for sample in samples]
            if key in tensor_keys:
                pipeline[key] = torch.cat(values, dim=0)
            else:
                pipeline[key] = values[0]
        pipeline["noise"] = pipeline["latents"]
        return {
            "pipeline": pipeline,
            "action_noise": torch.stack([sample["action_noise"] for sample in samples]),
            "proprio": torch.stack([sample["proprio"] for sample in samples]),
            "official_conditions": [sample["official_conditions"] for sample in samples],
        }

    def run_official_model(self, model, batch):
        conditions = batch.get("official_conditions", [])
        if len(conditions) != 1:
            raise ValueError("official OpenWAM generate() only supports batch size 1")
        result = model._batch_invariance_engine.generate(conditions[0])
        return {
            "actions": torch.from_numpy(np.asarray(result["actions"], dtype=np.float32)).unsqueeze(
                0
            )
        }

    def decode_video_latents(self, model, latents: torch.Tensor) -> torch.Tensor:
        """Decode each joint latent with the released Wan2.2 VAE and quantize."""
        vae = model.video_backbone.vae.to(self.device)
        video = vae.decode(
            latents.to(device=self.device, dtype=model.dtype),
            device=self.device,
            tiled=False,
        )
        return ((video.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu()

    def run_model(self, model, batch):
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch["pipeline"].items()
        }
        action = batch["action_noise"].to(self.device)
        proprio = batch["proprio"].to(self.device, dtype=model.dtype)
        batch_size = action.shape[0]
        self._trace("input.video_latent", inputs["latents"])
        self._trace("input.action_noise", action)
        self._trace("input.proprio", proprio)
        inactive_dims = model._resolve_inactive_action_dims(None, self.device)
        inactive_noise = None
        n_video = float(model.video_scheduler.num_train_timesteps)
        n_action = float(model.action_scheduler.num_train_timesteps)
        schedule = model._batch_invariance_schedule
        for index in range(len(schedule) - 1):
            t_video, t_action = schedule[index]
            next_video, next_action = schedule[index + 1]
            sigma_video = t_video / n_video
            sigma_action = t_action / n_action
            sigma_video_next = next_video / n_video
            sigma_action_next = next_action / n_action
            timestep_video = torch.full(
                (batch_size,), t_video, dtype=model.dtype, device=self.device
            )
            timestep_action = torch.full(
                (batch_size,), t_action, dtype=model.dtype, device=self.device
            )
            video_velocity, action_velocity = model.forward(
                action,
                timestep_action,
                **inputs,
                timestep=timestep_video,
                proprio=proprio,
            )
            self._trace(f"joint.step_{index:02d}.video_velocity", video_velocity)
            self._trace(f"joint.step_{index:02d}.action_velocity", action_velocity)
            latents = inputs["latents"] + video_velocity * (sigma_video_next - sigma_video)
            reference = inputs.get("first_frame_latents")
            if reference is not None:
                latents = latents.clone()
                latents[:, :, : reference.shape[2]] = reference
            inputs["latents"] = latents
            if inactive_dims is not None and inactive_noise is None:
                inactive_noise = action[..., inactive_dims].detach().clone() / float(sigma_action)
            action = model.action_scheduler.flow_step(
                action_velocity, sigma_action, sigma_action_next, action
            )
            if inactive_dims is not None:
                action[..., inactive_dims] = inactive_noise * float(sigma_action_next)
            self._trace(f"joint.step_{index:02d}.video_state", latents)
            self._trace(f"joint.step_{index:02d}.action_state", action)

        actions = action.float().cpu().numpy()
        if model.normalizer is not None:
            actions = model.normalizer.unnormalize(actions)
        result = {
            "actions": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            "video_latent": inputs["latents"].float().cpu(),
        }
        self._trace("output.actions", result["actions"])
        self._trace("output.video_latent", result["video_latent"])
        return result

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = OpenWAMAlphaRobotwinAdapter()

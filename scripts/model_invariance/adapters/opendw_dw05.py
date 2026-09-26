"""Official Dexmal DW05 RobotWin PyTorch action-policy adapter.

Required environment variables:

``OPENDW_SOURCE``
    Official ``dexmal/OpenDW`` checkout with the inference-only mmap patch in
    ``scripts/model_invariance/patches`` applied.
``OPENDW_CHECKPOINT``
    Complete local ``Dexmal/DW05-Robotwin`` runtime bundle.
``OPENDW_SAMPLE``
    A real RobotWin/ALOHA HDF5 episode with three RGB cameras and 14-D qpos.

The released public methods artificially require B=1 and create flow noise
internally. This adapter calls the same VAE, text encoder, MoT, ActionDiT, and
scheduler code with explicit per-sample action noise. The official runtime
supports cached context/latents, which are generated one sample at a time here
before the large text encoder and VAE are offloaded for the batch sweep.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
PROMPT_FORMAT = (
    "A video recorded from a robot's point of view executing the following " "instruction: {task}"
)


class OpenDWActionAdapter:
    name = "opendw-dw05-robotwin-official-pytorch-action"
    keep_vae = False

    def __init__(self) -> None:
        self.source = Path(os.environ.get("OPENDW_SOURCE", ""))
        self.checkpoint = Path(os.environ.get("OPENDW_CHECKPOINT", ""))
        self.sample = Path(os.environ.get("OPENDW_SAMPLE", ""))
        self.device = torch.device("cuda")
        self.trace_callback = None
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self.prompt: str | None = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        required = (
            self.source / "dexbotic/model/dw05/dw05_core.py",
            self.checkpoint / "model.pt",
            self.checkpoint / "norm_stats.json",
            self.checkpoint / "vae/model.pth",
            self.checkpoint / "text_encoder/model.pth",
            self.checkpoint / "tokenizer/tokenizer.json",
            self.sample,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing OpenDW artifacts: {missing}")

    def _add_source_path(self) -> None:
        source = str(self.source)
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)

    @staticmethod
    def _checkpoint_audit(model, checkpoint_file: Path) -> None:
        payload = torch.load(
            checkpoint_file,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        expected = model.mot.state_dict()
        saved = payload["mot"]
        missing = sorted(set(expected) - set(saved))
        unexpected = sorted(set(saved) - set(expected))
        mismatched = sorted(
            (key, tuple(expected[key].shape), tuple(saved[key].shape))
            for key in set(expected) & set(saved)
            if expected[key].shape != saved[key].shape
        )
        proprio_expected = model.proprio_encoder.state_dict()
        proprio_saved = payload["proprio_encoder"]
        proprio_mismatch = sorted(
            (key, tuple(proprio_expected[key].shape), tuple(proprio_saved[key].shape))
            for key in set(proprio_expected) & set(proprio_saved)
            if proprio_expected[key].shape != proprio_saved[key].shape
        )
        if (
            missing
            or unexpected
            or mismatched
            or set(proprio_expected) != set(proprio_saved)
            or proprio_mismatch
        ):
            raise RuntimeError(
                "official DW05 checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}, shapes={mismatched}, "
                f"proprio_shapes={proprio_mismatch}"
            )

    @staticmethod
    def _compose_image(images: list[np.ndarray]) -> np.ndarray:
        head = np.asarray(Image.fromarray(images[0]).resize((320, 256), Image.Resampling.BILINEAR))
        left = np.asarray(Image.fromarray(images[1]).resize((160, 128), Image.Resampling.BILINEAR))
        right = np.asarray(Image.fromarray(images[2]).resize((160, 128), Image.Resampling.BILINEAR))
        return np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)

    @staticmethod
    def _state_statistics(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
        statistics = json.loads((checkpoint / "norm_stats.json").read_text())
        state = statistics["state"]["default"]
        return (
            np.asarray(state["global_mean"], dtype=np.float32),
            np.asarray(state["global_std"], dtype=np.float32),
        )

    def load_example_inputs(self, count: int):
        model = getattr(self, "_loaded_model", None)
        if model is None:
            raise RuntimeError("adapter model reference was not initialized")
        state_mean, state_std = self._state_statistics(self.checkpoint)
        examples = []
        with h5py.File(self.sample, "r") as episode:
            frame_count = episode["observations/qpos"].shape[0]
            if count > frame_count:
                raise ValueError(f"fixture contains only {frame_count} frames")
            target_frame = int(os.environ.get("OPENDW_TARGET_FRAME", "0"))
            frame_indices = [target_frame]
            frame_indices.extend(index for index in range(frame_count) if index != target_frame)
            task = episode["language_raw"][0].decode()
            prompt = PROMPT_FORMAT.format(task=task)
            self.prompt = prompt
            for frame_index in frame_indices[:count]:
                images = [
                    np.array(episode[f"observations/images/{camera}"][frame_index], copy=True)
                    for camera in CAMERAS
                ]
                image = self._compose_image(images)
                image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
                image_tensor = image_tensor.to(self.device, dtype=torch.bfloat16)
                image_tensor = image_tensor * (2.0 / 255.0) - 1.0
                raw_state = np.asarray(episode["observations/qpos"][frame_index], dtype=np.float32)
                normalized_state = np.clip(
                    (raw_state - state_mean) / (state_std + 1.0e-8), -5.0, 5.0
                ).astype(np.float32)
                proprio = torch.from_numpy(normalized_state).to(self.device, dtype=torch.bfloat16)

                first_frame = model._encode_input_image_latents_tensor(
                    image_tensor.unsqueeze(0), tiled=False
                )
                context, context_mask = model._prepare_infer_context(
                    prompt=prompt,
                    context=None,
                    context_mask=None,
                    proprio=proprio.unsqueeze(0),
                )
                noise_seed = 93_000 + frame_index
                generator = torch.Generator().manual_seed(noise_seed)
                action_noise = torch.randn(32, 14, dtype=torch.float32, generator=generator)
                examples.append(
                    {
                        "first_frame": first_frame[0].detach().cpu(),
                        "context": context[0].detach().cpu(),
                        "context_mask": context_mask[0].detach().cpu(),
                        "action_noise": action_noise,
                        "raw_image": image_tensor.detach().cpu(),
                        "proprio": proprio.detach().cpu(),
                        "noise_seed": torch.tensor(noise_seed, dtype=torch.int64),
                    }
                )

        if os.environ.get("OPENDW_KEEP_PREPROCESSORS", "0") != "1":
            model.text_encoder.to("cpu")
            if not self.keep_vae:
                model.vae.to("cpu")
            torch.cuda.empty_cache()
        return examples

    def load_model(self):
        model = self._load_model_impl()
        self._loaded_model = model
        return model

    def _load_model_impl(self):
        self._validate_paths()
        self._add_source_path()
        os.environ.setdefault("DW05_MODEL_BASE_PATH", str(self.checkpoint))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from dexbotic.model.dw05 import DW05ModelConfig

        # The bundle's top-level config says 32, but the official runtime's
        # _checkpoint_dims path reads the actual learned action/proprio width:
        # both checkpoint tensors are 14-D.
        config = DW05ModelConfig(
            load_text_encoder=True,
            skip_dit_load_from_pretrain=True,
            action_dim=14,
            proprio_dim=14,
            mot_checkpoint_mixed_attn=False,
        )
        model = config.build_model(model_dtype=torch.bfloat16, device=str(self.device))
        self._checkpoint_audit(model, self.checkpoint / "model.pt")
        model.load_checkpoint(self.checkpoint / "model.pt")
        model.eval()
        statistics = json.loads((self.checkpoint / "norm_stats.json").read_text())
        action_stats = statistics["action"]["default"]
        self.action_mean = np.asarray(action_stats["global_mean"], dtype=np.float32)
        self.action_std = np.asarray(action_stats["global_std"], dtype=np.float32)
        return model

    @staticmethod
    def compose_batch(samples):
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}

    def _denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        value = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        value = value * (self.action_std + 1.0e-8) + self.action_mean
        return torch.from_numpy(value.astype(np.float32, copy=False))

    def _run_action_core(self, model, batch):
        first_frame = batch["first_frame"].to(self.device)
        context = batch["context"].to(self.device)
        context_mask = batch["context_mask"].to(self.device)
        latents_action = batch["action_noise"].to(self.device, dtype=torch.bfloat16)
        batch_size = latents_action.shape[0]

        timestep_video = torch.zeros(batch_size, dtype=first_frame.dtype, device=self.device)
        video_pre = model.video_expert.pre_dit(
            x=first_frame,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        self._trace("video.condition_tokens", video_pre["tokens"])
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache, video_seq_len = model._prefill_video_cache(video_pre, attention_mask)
        self._trace("video.cache.layer0.key", video_kv_cache[0]["k"])

        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=5,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=None,
        )
        for step, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
            timestep = step_t.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            prediction = model._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            self._trace(f"action.step_{step:02d}.velocity", prediction)
            latents_action = model.infer_action_scheduler.step(
                prediction, step_delta, latents_action
            )
            self._trace(f"action.step_{step:02d}.state", latents_action)
        return latents_action

    def run_model(self, model, batch):
        self._trace("preprocess.first_frame", batch["first_frame"])
        self._trace("preprocess.context", batch["context"])
        self._trace("input.action_noise", batch["action_noise"])
        actions = self._run_action_core(model, batch)
        normalized = actions.to(dtype=torch.float32, device="cpu")
        self._trace("actions.normalized", normalized)
        result = self._denormalize(actions)
        self._trace("actions.unnormalized", result)
        return result

    def run_official_model(self, model, batch):
        if batch["raw_image"].shape[0] != 1:
            raise ValueError("released DW05 infer_action only supports B=1")
        if self.prompt is None:
            raise RuntimeError("load_example_inputs must initialize the prompt")
        output = model.infer_action(
            prompt=self.prompt,
            input_image=batch["raw_image"].to(self.device),
            action_horizon=32,
            proprio=batch["proprio"].to(self.device),
            num_inference_steps=5,
            seed=int(batch["noise_seed"][0]),
            rand_device="cpu",
            tiled=False,
        )["action"]
        return self._denormalize(output.unsqueeze(0))

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = OpenDWActionAdapter()

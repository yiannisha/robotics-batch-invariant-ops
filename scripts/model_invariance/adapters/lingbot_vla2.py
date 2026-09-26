"""Official LingBot-VLA 2.0 PyTorch adapter for the RoboTwin checkpoint.

Required environment variables:

``LINGBOT_VLA2_CHECKPOINT``
    Local ``hf_ckpt`` directory from ``robbyant/lingbot-vla-v2-6b-robotwin``.
``LINGBOT_VLA2_SAMPLE``
    Real three-camera HDF5 episode with ALOHA-style 14-D joint state.

The released inference path mutates caller-provided flow noise, retains vision
grid metadata for the first batch shape, and uses relaxed atomic operations in
its fused MoE inference kernel.  This adapter controls those non-batch sources
by cloning explicit per-frame noise, clearing the shape cache between calls,
and substituting the reusable deterministic token-choice MoE implementation.
All remaining official FP32 model operations and ten flow-matching steps are
preserved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


class LingBotVLA2Adapter:
    name = "lingbot-vla2-6b-robotwin-official-pytorch"
    camera_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    model_keys = (
        "images",
        "img_masks",
        "lang_tokens",
        "lang_masks",
        "state",
        "image_grid_thw",
    )

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("LINGBOT_VLA2_CHECKPOINT", ""))
        self.sample = Path(os.environ.get("LINGBOT_VLA2_SAMPLE", ""))
        self.device = torch.device("cuda")
        self.server: Any | None = None
        self.trace_callback = None

    def _validate_paths(self) -> None:
        if not self.checkpoint.is_dir():
            raise FileNotFoundError("set LINGBOT_VLA2_CHECKPOINT to the official hf_ckpt directory")
        if not list(self.checkpoint.glob("*.safetensors")):
            raise FileNotFoundError("the LingBot checkpoint contains no safetensor shards")
        training_config = self.checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
        if not training_config.is_file():
            raise FileNotFoundError(
                f"official training configuration is missing at {training_config}"
            )
        if not self.sample.is_file():
            raise FileNotFoundError("set LINGBOT_VLA2_SAMPLE to a real HDF5 episode")

    @staticmethod
    def _install_deterministic_moe() -> None:
        from batch_invariant_ops import deterministic_token_choice_moe
        from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert

        qwen2_action_expert.robby_moe_forward = deterministic_token_choice_moe

    def load_model(self):
        self._validate_paths()
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        self._install_deterministic_moe()
        self.server = LingbotVLAv2Server(
            path_to_pi_model=str(self.checkpoint),
            chunk_ret=True,
            use_bf16=False,
            use_fp32=True,
            use_compile=False,
        )
        self.server.reset("robotwin")
        return self.server.vla.model.eval()

    def load_example_inputs(self, count: int):
        if self.server is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        with h5py.File(self.sample, "r") as episode:
            frame_count = episode["observations/qpos"].shape[0]
            if count > frame_count:
                raise ValueError(f"fixture contains only {frame_count} frames")
            target_frame = int(os.environ.get("LINGBOT_VLA2_TARGET_FRAME", "0"))
            if not 0 <= target_frame < frame_count:
                raise ValueError(f"LINGBOT_VLA2_TARGET_FRAME must be below {frame_count}")
            frame_indices = [target_frame]
            frame_indices.extend(index for index in range(frame_count) if index != target_frame)
            task = episode["language_raw"][0].decode()
            examples = []
            for frame_index in frame_indices[:count]:
                observation: dict[str, Any] = {
                    "task": task,
                    "observation.state": np.array(
                        episode["observations/qpos"][frame_index], copy=True
                    ),
                }
                for camera_name in self.camera_names:
                    observation[f"observation.images.{camera_name}"] = np.array(
                        episode[f"observations/images/{camera_name}"][frame_index],
                        copy=True,
                    )
                transformed = self.server._prepare_model_input(observation)
                generator = torch.Generator().manual_seed(82_000 + frame_index)
                noise = torch.randn(50, 55, dtype=torch.float32, generator=generator)
                examples.append(
                    {
                        **{key: transformed[key] for key in self.model_keys},
                        "noise": noise,
                    }
                )
        return examples

    @staticmethod
    def compose_batch(samples):
        return {
            key: torch.stack([sample[key] for sample in samples])
            for key in (*LingBotVLA2Adapter.model_keys, "noise")
        }

    @staticmethod
    def reset_model(model) -> None:
        vision = model.qwenvl_with_expert
        for attribute in (
            "pos_embeds",
            "position_embeddings",
            "cu_seqlens",
            "visual_split_sizes",
            "visual_max_seqlen",
        ):
            setattr(vision, attribute, None)

    def run_model(self, model, batch):
        inputs = {
            "images": batch["images"].to(self.device, dtype=torch.float32),
            "img_masks": batch["img_masks"].to(self.device),
            "lang_tokens": batch["lang_tokens"].to(self.device),
            "lang_masks": batch["lang_masks"].to(self.device),
            "state": batch["state"].to(self.device, dtype=torch.float32),
            "image_grid_thw": batch["image_grid_thw"].to(self.device, dtype=torch.long),
            # Official sample_actions updates x_t in place.
            "noise": batch["noise"].to(self.device).clone(),
        }
        output = model.sample_actions(**inputs)
        return output.to(dtype=torch.float32, device="cpu")

    @staticmethod
    def extract_robot_output(output):
        return output


adapter = LingBotVLA2Adapter()

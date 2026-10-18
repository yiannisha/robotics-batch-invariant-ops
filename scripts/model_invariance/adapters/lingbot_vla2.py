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

import contextlib
import json
import os
import types
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
    causal_ablation_inapplicable_operations = {
        "explicit_sdpa": {
            "reason": (
                "The released action/VLM joint path implements eager attention with "
                "torch.einsum and never dispatches aten::scaled_dot_product_attention."
            ),
            "evidence_path": (
                "/workspace/models/lingbot-vla-v2/lingbotvla/models/vla/"
                "lingbot_vla/utils.py"
            ),
        }
    }

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[3]
        self.checkpoint = Path(os.environ.get(
            "LINGBOT_VLA2_CHECKPOINT",
            "/workspace/assets/lingbot_vla2/checkpoints/global_step_50000/hf_ckpt",
        ))
        self.sample = Path(os.environ.get(
            "LINGBOT_VLA2_SAMPLE", "/workspace/assets/dexvla/sample/episode_47.hdf5"
        ))
        self.replay_manifest_path = Path(os.environ.get(
            "LINGBOT_VLA2_REPLAY_MANIFEST",
            str(root / "results/cross_model/lingbot_vla2/full/replay_manifest.json"),
        ))
        self.device = torch.device("cuda")
        self.server: Any | None = None
        self.replay_manifest: dict[str, Any] | None = None
        self.replay_noise: np.ndarray | None = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

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
        import deploy
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        self._install_deterministic_moe()
        source_root = Path(deploy.__file__).resolve().parent.parent
        # The released server resolves its robot configuration from the
        # process working directory.  Scope that assumption to construction
        # and reset so the campaign can be launched from this repository.
        with contextlib.chdir(source_root):
            self.server = LingbotVLAv2Server(
                path_to_pi_model=str(self.checkpoint),
                chunk_ret=True,
                use_bf16=False,
                use_fp32=True,
                use_compile=False,
            )
            self.server.reset("robotwin")
        return self.server.vla.model.eval()

    def execution_configuration(self) -> dict[str, Any]:
        return {
            "camera_names": list(self.camera_names),
            "image_shape_hwc": [480, 640, 3],
            "state_dim": 14,
            "flow_steps": 10,
            "action_chunk_size": 50,
            "model_action_dim": 55,
            "source_limit_observations": 396,
            "precision": "float32",
            "deterministic_moe": "deterministic_token_choice_moe",
        }

    def _load_replay_metadata(self) -> None:
        if self.replay_manifest is not None:
            return
        if not self.replay_manifest_path.is_file():
            raise FileNotFoundError(self.replay_manifest_path)
        manifest = json.loads(self.replay_manifest_path.read_text())
        if (
            manifest.get("model_id") != "lingbot_vla2"
            or len(manifest.get("observations", [])) != 396
            or manifest.get("noise_shape") != [396, 3, 50, 55]
        ):
            raise ValueError("LingBot replay manifest does not describe the frozen source limit")
        source = Path(manifest["source_root"]) / manifest["source_file"]
        if source.resolve() != self.sample.resolve():
            raise ValueError("LINGBOT_VLA2_SAMPLE differs from the frozen replay source")
        noise_path = Path(manifest["noise_file"])
        if not noise_path.is_absolute():
            noise_path = self.replay_manifest_path.parent / noise_path
        self.replay_manifest = manifest
        self.replay_noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.server is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        self._load_replay_metadata()
        assert self.replay_manifest is not None and self.replay_noise is not None
        if not 0 <= replay_index < len(self.replay_manifest["observations"]):
            raise IndexError(replay_index)
        if noise_id not in (0, 1, 2):
            raise IndexError(noise_id)
        row = self.replay_manifest["observations"][replay_index]
        if (
            row.get("replay_index") != replay_index
            or row.get("source_frame_index") != replay_index
        ):
            raise ValueError("LingBot replay index is not a one-to-one source-frame mapping")
        with h5py.File(self.sample, "r") as episode:
            raw_state = np.array(episode["observations/qpos"][replay_index], copy=True)
            observation: dict[str, Any] = {
                "task": row["task"],
                "observation.state": raw_state,
            }
            for camera_name in self.camera_names:
                observation[f"observation.images.{camera_name}"] = np.array(
                    episode[f"observations/images/{camera_name}"][replay_index], copy=True
                )
        if not np.array_equal(raw_state, np.asarray(row["state"], dtype=np.float32)):
            raise ValueError("LingBot HDF5 state differs from the frozen replay row")
        transformed = self.server._prepare_model_input(observation)
        return {
            **{key: transformed[key] for key in self.model_keys},
            "noise": torch.from_numpy(
                np.array(self.replay_noise[replay_index, noise_id], copy=True)
            ),
        }

    def load_example_inputs(self, count: int):
        if self.server is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if self.replay_manifest_path.is_file():
            return [self.load_replay_input(index, 0) for index in range(count)]
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
        original_predict_velocity = None
        if self.trace_callback is not None:
            original_predict_velocity = model.predict_velocity
            adapter = self
            step = 0

            def traced_predict_velocity(_model, *args, **kwargs):
                nonlocal step
                velocity = original_predict_velocity(*args, **kwargs)
                state = args[3]
                dt = torch.tensor(
                    -1.0 / _model.config.num_steps,
                    dtype=state.dtype,
                    device=state.device,
                )
                adapter._trace(f"flow.step_{step:02d}.prediction", velocity)
                adapter._trace(f"flow.step_{step:02d}.state", state + dt * velocity)
                step += 1
                return velocity

            model.predict_velocity = types.MethodType(traced_predict_velocity, model)
        try:
            output = model.sample_actions(**inputs)
        finally:
            if original_predict_velocity is not None:
                model.predict_velocity = original_predict_velocity
        return output.to(dtype=torch.float32, device="cpu")

    @staticmethod
    def extract_robot_output(output):
        return output


adapter = LingBotVLA2Adapter()

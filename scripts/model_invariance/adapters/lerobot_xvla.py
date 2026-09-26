"""LeRobot X-VLA LIBERO adapter using public PyTorch weights and real inputs.

Required environment variables:

``LEROBOT_XVLA_CHECKPOINT``
    Pinned local snapshot of ``lerobot/xvla-libero``.
``LEROBOT_XVLA_SAMPLE_DIR``
    Fixture containing the first 64 frames from both cameras, the first
    episode parquet, and ``tasks.parquet`` from ``lerobot/libero``.

The released policy API accepts a ``noise`` argument but its current X-VLA
implementation does not use it.  This adapter follows the official inference
body while making the initial flow noise an explicit per-example input, so RNG
shape cannot be confused with numerical batch dependence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


class LeRobotXVLAAdapter:
    name = "lerobot-xvla-libero-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("LEROBOT_XVLA_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("LEROBOT_XVLA_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.config = None
        self.preprocessor = None
        self.libero_postprocessor = None

    def _validate_paths(self) -> None:
        for filename in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set LEROBOT_XVLA_CHECKPOINT to a snapshot containing {filename}"
                )
        for filename in (
            "frames/image_001.png",
            "frames/image_064.png",
            "frames/image2_001.png",
            "frames/image2_064.png",
            "data/chunk-000/file-000.parquet",
            "tasks.parquet",
        ):
            if not (self.sample_dir / filename).is_file():
                raise FileNotFoundError(
                    f"set LEROBOT_XVLA_SAMPLE_DIR to a directory containing {filename}"
                )

    def load_model(self):
        self._validate_paths()
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import (
            XVLARotation6DToAxisAngleProcessorStep,
        )

        model = XVLAPolicy.from_pretrained(self.checkpoint)
        self.config = model.config
        self.preprocessor, _ = make_pre_post_processors(
            self.config, pretrained_path=self.checkpoint
        )
        self.libero_postprocessor = XVLARotation6DToAxisAngleProcessorStep()
        model.eval()
        return model

    def _image(self, camera: str, frame_number: int) -> torch.Tensor:
        array = np.array(
            Image.open(self.sample_dir / "frames" / f"{camera}_{frame_number:03d}.png").convert(
                "RGB"
            ),
            copy=True,
        )
        return torch.from_numpy(array).permute(2, 0, 1).float().div(255)

    def load_example_inputs(self, count: int):
        if self.config is None or self.preprocessor is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")

        target_frame = int(os.environ.get("XVLA_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("XVLA_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)

        rows = (
            pq.read_table(
                self.sample_dir / "data/chunk-000/file-000.parquet",
                columns=["observation.state", "task_index"],
            )
            .slice(0, 64)
            .to_pylist()
        )
        task_rows = pq.read_table(self.sample_dir / "tasks.parquet").to_pylist()
        tasks = {int(row["task_index"]): row["__index_level_0__"] for row in task_rows}

        examples = []
        for frame_index in frame_indices[:count]:
            row = rows[frame_index]
            frame_number = frame_index + 1
            processed = self.preprocessor(
                {
                    "observation.images.image": self._image("image", frame_number),
                    "observation.images.image2": self._image("image2", frame_number),
                    "observation.state": torch.tensor(
                        row["observation.state"], dtype=torch.float32
                    ),
                    "task": tasks[int(row["task_index"])],
                }
            )
            generator = torch.Generator(device="cpu").manual_seed(87_000 + frame_index)
            noise = torch.randn(
                self.config.chunk_size,
                model_action_dim(self.config),
                dtype=torch.float32,
                generator=generator,
            )
            examples.append(
                {
                    key: value.cpu()
                    for key, value in processed.items()
                    if isinstance(value, torch.Tensor)
                    and (key.startswith("observation.") or key == "domain_id")
                }
                | {"noise": noise.unsqueeze(0)}
            )
        return examples

    @staticmethod
    def compose_batch(samples):
        return {key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]}

    @staticmethod
    def reset_model(model) -> None:
        model.reset()

    def run_model(self, policy, batch):
        from lerobot.lerobot_types import TransitionKey

        model_batch = {key: value.to(self.device) for key, value in batch.items() if key != "noise"}
        inputs = policy._build_model_inputs(model_batch)
        model = policy.model
        target_dtype = model._get_target_dtype()
        inputs["image_input"] = inputs["image_input"].to(dtype=target_dtype)
        inputs["proprio"] = inputs["proprio"].to(dtype=target_dtype)
        encoded = model.forward_vlm(
            inputs.pop("input_ids"),
            inputs.pop("image_input"),
            inputs.pop("image_mask"),
        )

        initial_noise = batch["noise"].to(self.device, dtype=target_dtype)
        action = torch.zeros_like(initial_noise)
        steps = int(self.config.num_denoising_steps)
        for step in range(steps, 0, -1):
            timestep = torch.full(
                (initial_noise.shape[0],),
                step / steps,
                device=self.device,
                dtype=target_dtype,
            )
            noisy_action = initial_noise * timestep[:, None, None] + action * (
                1 - timestep[:, None, None]
            )
            proprio, noisy_action = model.action_space.preprocess(inputs["proprio"], noisy_action)
            action = model.transformer(
                domain_id=inputs["domain_id"],
                action_with_noise=noisy_action,
                proprio=proprio,
                t=timestep,
                **encoded,
            )
        action = model.action_space.postprocess(action)

        flattened = action.reshape(-1, action.shape[-1])
        processed = self.libero_postprocessor({TransitionKey.ACTION: flattened})[
            TransitionKey.ACTION
        ]
        return processed.reshape(action.shape[0], action.shape[1], -1).to(self.device)

    @staticmethod
    def extract_robot_output(output: Any):
        return output


def model_action_dim(config) -> int:
    # The public LIBERO checkpoint uses the ee6d action space (20 model-facing
    # channels). Keeping this helper config-derived makes fixture generation
    # fail clearly for an unsupported future checkpoint instead of silently
    # choosing the wrong noise shape.
    if config.action_mode.lower() != "ee6d":
        raise ValueError(f"unsupported X-VLA action mode: {config.action_mode}")
    return 20


adapter = LeRobotXVLAAdapter()

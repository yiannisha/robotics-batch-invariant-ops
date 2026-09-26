"""Official LeRobot SmolVLA LIBERO adapter with explicit flow noise.

Required environment variables:

``LEROBOT_SMOLVLA_CHECKPOINT``
    Local native snapshot of ``HuggingFaceVLA/smolvla_libero``.
``LEROBOT_SMOLVLA_SAMPLE_DIR``
    Real LIBERO episode-zero fixture containing two camera streams, the first
    episode parquet, and ``tasks.parquet``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


class LeRobotSmolVLAAdapter:
    name = "lerobot-smolvla-libero-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("LEROBOT_SMOLVLA_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("LEROBOT_SMOLVLA_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.config = None
        self.preprocessor = None
        self.postprocessor = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        for filename in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set LEROBOT_SMOLVLA_CHECKPOINT to a snapshot containing {filename}"
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
                    f"set LEROBOT_SMOLVLA_SAMPLE_DIR to a directory containing {filename}"
                )

    def load_model(self):
        self._validate_paths()
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = SmolVLAConfig.from_pretrained(self.checkpoint)
        config.compile_model = False
        # The policy safetensors are complete. Construct the exact architecture
        # without first loading a redundant base VLM, then strictly load them.
        config.load_vlm_weights = False
        policy = SmolVLAPolicy.from_pretrained(
            self.checkpoint, config=config, strict=True
        )
        self.config = policy.config
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        policy.to(self.device).eval()
        return policy

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

        target_frame = int(os.environ.get("SMOLVLA_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("SMOLVLA_TARGET_FRAME must be between 0 and 63")
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
            generator = torch.Generator(device="cpu").manual_seed(93_000 + frame_index)
            noise = torch.randn(
                1,
                self.config.chunk_size,
                self.config.max_action_dim,
                dtype=torch.float32,
                generator=generator,
            )
            examples.append(
                {
                    key: value.detach().cpu()
                    for key, value in processed.items()
                    if isinstance(value, torch.Tensor)
                    and key.startswith("observation.")
                }
                | {"noise": noise}
            )
        return examples

    @staticmethod
    def compose_batch(samples):
        return {key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]}

    @staticmethod
    def reset_model(policy) -> None:
        policy.reset()

    def run_model(self, policy, batch):
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        model_batch = {key: value.to(self.device) for key, value in batch.items() if key != "noise"}
        images, image_masks = policy.prepare_images(model_batch)
        state = policy.prepare_state(model_batch)
        for index, image in enumerate(images):
            self._trace(f"preprocess.image{index}", image)
        self._trace("preprocess.state", state)
        self._trace("input.noise", batch["noise"].to(self.device))

        model = policy.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images,
            image_masks,
            model_batch[OBS_LANGUAGE_TOKENS],
            model_batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )
        self._trace("prefix.embeddings", prefix_embs)
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_output, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )
        self._trace("prefix.output", prefix_output[0])

        actions = batch["noise"].to(self.device)
        step_size = -1.0 / self.config.num_steps
        for step in range(self.config.num_steps):
            timestep = torch.tensor(
                1.0 + step * step_size, dtype=torch.float32, device=self.device
            ).expand(actions.shape[0])
            velocity = model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=actions,
                timestep=timestep,
            )
            self._trace(f"flow.step_{step:02d}.velocity", velocity)
            actions = actions + step_size * velocity
            self._trace(f"flow.step_{step:02d}.state", actions)

        actions = actions[..., : self.config.action_feature.shape[0]]
        self._trace("actions.normalized", actions)
        actions = self.postprocessor(actions.detach().cpu()).to(self.device)
        self._trace("actions.unnormalized", actions)
        return actions

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = LeRobotSmolVLAAdapter()

"""InternVLA-A1.5 adapter using the official PyTorch LIBERO checkpoint.

Required environment variables:

``INTERNVLA_A15_CHECKPOINT``
    Local ``InternRobotics/InternVLA-A1.5-Libero`` snapshot.
``INTERNVLA_A15_QWEN``
    Local pinned ``Qwen/Qwen3.5-2B`` snapshot used by the official loader and
    chat/image processor.
``INTERNVLA_A15_SAMPLE_DIR``
    LIBERO fixture with ``frames/image_*.png``, ``frames/image2_*.png``, and
    ``data/chunk-000/file-000.parquet`` as described by the PI0-FAST adapter.

Set ``INTERNVLA_A15_BACKEND`` to ``standard`` (default) or ``optimized``.
Both are action-only inference modes and skip the training-time WAN branch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image


class InternVLAA15Adapter:
    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("INTERNVLA_A15_CHECKPOINT", ""))
        self.qwen = Path(os.environ.get("INTERNVLA_A15_QWEN", ""))
        self.sample_dir = Path(os.environ.get("INTERNVLA_A15_SAMPLE_DIR", ""))
        self.backend = os.environ.get("INTERNVLA_A15_BACKEND", "standard")
        self.device = torch.device("cuda")
        self.config = None
        self.processor = None
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.action_low = None
        self.action_high = None

    @property
    def name(self) -> str:
        return f"internvla-a1.5-libero-{self.backend}-public-weights"

    def _validate_paths(self) -> None:
        for root, filenames in (
            (self.checkpoint, ("config.json", "model.safetensors", "stats.json")),
            (
                self.qwen,
                (
                    "config.json",
                    "model.safetensors-00001-of-00001.safetensors",
                    "tokenizer.json",
                    "preprocessor_config.json",
                ),
            ),
            (
                self.sample_dir,
                (
                    "frames/image_001.png",
                    "frames/image_064.png",
                    "frames/image2_001.png",
                    "frames/image2_064.png",
                    "data/chunk-000/file-000.parquet",
                ),
            ),
        ):
            for filename in filenames:
                if not (root / filename).is_file():
                    raise FileNotFoundError(f"missing required fixture: {root / filename}")

    def load_model(self):
        self._validate_paths()
        if self.backend not in {"standard", "optimized"}:
            raise ValueError("INTERNVLA_A15_BACKEND must be standard or optimized")

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.internvla_a1_5 import InternVLAA15Policy
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
            InternVLAA15ChatProcessorTransformFn,
        )

        # Load through the registered base class.  The upstream subclass loader
        # currently attempts to decode the checkpoint's ``type`` discriminator
        # as a dataclass field before stripping it.
        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.device = str(self.device)
        config.vlm_model_name_or_path = str(self.qwen)
        config.action_loss_only = True
        config.inference_backend = self.backend
        config.compile_model = False
        config.gradient_checkpointing = False
        config.num_inference_steps = int(os.environ.get("INTERNVLA_A15_STEPS", "10"))
        self.config = config

        self.processor = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=str(self.qwen),
            max_length=650,
            tokenize_state=config.tokenize_state,
            max_state_dim=config.max_state_dim,
            use_fast_action_tokens=False,
            mode="eval",
            action_mode="joint",
        )

        statistics = json.loads((self.checkpoint / "stats.json").read_text())["libero_10"]
        state_statistics = statistics["observation.state"]
        action_statistics = statistics["action"]
        self.state_mean = torch.tensor(state_statistics["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(state_statistics["std"], dtype=torch.float32)
        self.action_mean = torch.tensor(action_statistics["mean"], dtype=torch.float32)
        self.action_std = torch.tensor(action_statistics["std"], dtype=torch.float32)
        self.action_low = torch.tensor(action_statistics["min"], dtype=torch.float32)
        self.action_high = torch.tensor(action_statistics["max"], dtype=torch.float32)

        model = InternVLAA15Policy.from_pretrained(
            pretrained_name_or_path=self.checkpoint,
            config=config,
            strict=False,
        )
        model.eval()
        return model

    def _load_image(self, filename: str) -> torch.Tensor:
        image = Image.open(self.sample_dir / filename).convert("RGB")
        array = np.array(image, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).float().div(255)

    @staticmethod
    def _pad_1d(tensor: torch.Tensor, length: int, value: int | bool) -> torch.Tensor:
        return F.pad(tensor, (0, length - tensor.shape[0]), value=value)

    def load_example_inputs(self, count: int):
        if self.config is None or self.processor is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")

        target_frame = int(os.environ.get("INTERNVLA_A15_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("INTERNVLA_A15_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)
        frame_indices = frame_indices[:count]

        state_rows = pq.read_table(
            self.sample_dir / "data/chunk-000/file-000.parquet",
            columns=["observation.state"],
        )["observation.state"].to_pylist()
        task = "put the white mug on the left plate and put the yellow and white mug on the right plate"
        prompt_length = 650
        examples = []
        for frame_index in frame_indices:
            image_number = frame_index + 1
            first = self._load_image(f"frames/image_{image_number:03d}.png")
            second = self._load_image(f"frames/image2_{image_number:03d}.png")
            raw_state = torch.tensor(state_rows[frame_index], dtype=torch.float32)
            state = (raw_state - self.state_mean) / self.state_std.clamp_min(1e-6)
            processed = self.processor(
                {
                    "observation.images.image0": first,
                    "observation.images.image0_mask": True,
                    "observation.images.image1": second,
                    "observation.images.image1_mask": True,
                    "observation.images.image2": torch.zeros_like(first),
                    "observation.images.image2_mask": False,
                    "observation.state": state,
                    "task": task,
                }
            )
            generator = torch.Generator(device="cpu").manual_seed(19_000 + frame_index)
            noise = torch.randn(
                self.config.chunk_size,
                self.config.max_action_dim,
                generator=generator,
                dtype=torch.float32,
            )
            examples.append(
                {
                    "pixel_values": processed["observation.pixel_values"],
                    "image_grid_thw": processed["observation.image_grid_thw"],
                    "input_ids": self._pad_1d(processed["observation.input_ids"], prompt_length, 0),
                    "attention_mask": self._pad_1d(
                        processed["observation.attention_mask"], prompt_length, False
                    ),
                    "fast_token_mask": self._pad_1d(
                        processed["observation.fast_token_mask"], prompt_length, False
                    ),
                    "state": state,
                    "noise": noise,
                }
            )
        return examples

    def reset_model(self, model) -> None:
        core = model.model
        if hasattr(core, "_graphs"):
            core._graphs.clear()
            core._static_buffers.clear()

    def run_model(self, model, batch):
        inputs = {key: value.to(self.device) for key, value in batch.items()}
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized_actions = model.model.sample_actions(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["state"],
                fast_token_mask=inputs["fast_token_mask"],
                noise=inputs["noise"],
                num_steps=self.config.num_inference_steps,
            )
        action_dim = self.action_mean.numel()
        actions = normalized_actions[..., :action_dim].float().cpu()
        actions = actions * self.action_std + self.action_mean
        actions = torch.maximum(torch.minimum(actions, self.action_high), self.action_low)
        return actions.to(self.device)

    def extract_robot_output(self, output):
        return output


adapter = InternVLAA15Adapter()

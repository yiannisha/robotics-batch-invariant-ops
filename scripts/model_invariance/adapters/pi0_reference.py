"""Adapter for the pre-fix PyTorch PiZero reference implementation.

Set ``PIZERO_SOURCE`` to the ``open-pi-zero`` checkout root and
``PIZERO_CONFIG`` to the synthetic inference YAML from the companion
batch-invariance repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


class Pi0ReferenceAdapter:
    name = "pi0-reference-random"

    def __init__(self) -> None:
        self.source = Path(os.environ.get("PIZERO_SOURCE", ""))
        self.config_path = Path(os.environ.get("PIZERO_CONFIG", ""))
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.config = None
        self.model = None

    def _validate_paths(self) -> None:
        if not (self.source / "src/model/vla/pizero.py").is_file():
            raise FileNotFoundError(
                "set PIZERO_SOURCE to an open-pi-zero checkout containing src/model/vla/pizero.py"
            )
        if not self.config_path.is_file():
            raise FileNotFoundError("set PIZERO_CONFIG to the synthetic inference YAML")

    def load_model(self):
        self._validate_paths()
        sys.path.insert(0, str(self.source))
        from omegaconf import OmegaConf
        from src.model.vla.pizero import PiZeroInference

        self.config = OmegaConf.load(self.config_path)
        self.model = PiZeroInference(self.config, use_ddp=False)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        return self.model

    def load_example_inputs(self, count: int):
        config = self.config
        model = self.model
        examples = []
        for index in range(count):
            generator = torch.Generator().manual_seed(10_000 + index)
            num_image_tokens = int(config.vision.config.num_image_tokens)
            sequence_length = int(config.max_image_text_tokens)
            input_ids = torch.full((sequence_length,), config.pad_token_id, dtype=torch.long)
            input_ids[:num_image_tokens] = config.image_token_index
            input_ids[num_image_tokens] = index + 1
            attention_mask = torch.zeros(sequence_length, dtype=torch.long)
            attention_mask[: num_image_tokens + 1] = 1
            image_size = int(config.vision.config.image_size)
            pixel_values = (
                torch.rand(
                    3,
                    image_size,
                    image_size,
                    dtype=self.dtype,
                    generator=generator,
                )
                * 2
                - 1
            )
            proprios = torch.randn(
                int(config.cond_steps),
                int(config.proprio_dim),
                dtype=self.dtype,
                generator=generator,
            )
            initial_action = torch.randn(
                int(config.horizon_steps),
                int(config.action_dim),
                dtype=self.dtype,
                generator=generator,
            )
            raw = {
                "input_ids": input_ids[None],
                "attention_mask": attention_mask[None],
                "pixel_values": pixel_values[None],
                "proprios": proprios[None],
                "initial_action": initial_action[None],
            }
            causal_mask, vlm_ids, proprio_ids, action_ids = (
                model.build_causal_mask_and_position_ids(raw["attention_mask"], dtype=self.dtype)
            )
            image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)
            examples.append(
                {
                    "input_ids": input_ids,
                    "pixel_values": pixel_values,
                    "image_text_proprio_mask": image_text_proprio_mask[0],
                    "action_mask": action_mask[0],
                    "vlm_position_ids": vlm_ids[0],
                    "proprio_position_ids": proprio_ids[0],
                    "action_position_ids": action_ids[0],
                    "proprios": proprios,
                    "initial_action": initial_action,
                }
            )
        return examples

    def run_model(self, model, batch):
        device_batch = {key: value.to(self.device) for key, value in batch.items()}
        return model(**device_batch)

    def extract_robot_output(self, output):
        return output


adapter = Pi0ReferenceAdapter()

"""Adapters for the official PyTorch OpenPI implementation.

The adapter deliberately supplies flow noise as part of each example.  This
keeps the target sample's stochastic input fixed when companion samples are
added to a batch.  By default the model uses deterministically initialized
weights; set ``OPENPI_PYTORCH_WEIGHTS`` to an official converted
``model.safetensors`` file to exercise released weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class OpenPiAdapter:
    """Run official OpenPI Pi0/Pi0.5 inference through the common harness."""

    def __init__(self, *, pi05: bool) -> None:
        self.pi05 = pi05
        self.name = f"openpi-{'pi05' if pi05 else 'pi0'}"
        self.device = torch.device("cuda")
        self.config = None
        self.model = None

    def load_model(self):
        from openpi.models import pi0_config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        # Match the public DROID configurations while avoiding torch.compile:
        # shape-specialized compilation would obscure the eager operator path
        # under investigation.
        self.config = pi0_config.Pi0Config(
            pi05=self.pi05,
            action_horizon=15 if self.pi05 else 10,
            pytorch_compile_mode=None,
        )
        self.model = PI0Pytorch(self.config)

        weights = os.environ.get("OPENPI_PYTORCH_WEIGHTS")
        if weights:
            from safetensors.torch import load_model

            weight_path = Path(weights)
            if weight_path.is_dir():
                weight_path = weight_path / "model.safetensors"
            if not weight_path.is_file():
                raise FileNotFoundError(f"OpenPI weights not found: {weight_path}")
            load_model(self.model, str(weight_path))
            self.name += "-official-weights"
        else:
            self.name += "-random-weights"

        self.model.eval().to(self.device)
        return self.model

    def load_example_inputs(self, count: int):
        if self.config is None:
            raise RuntimeError("load_model must be called before load_example_inputs")

        examples = []
        for index in range(count):
            generator = torch.Generator().manual_seed(20_000 + index)
            images = {
                key: torch.rand(3, 224, 224, generator=generator, dtype=torch.float32) * 2 - 1
                for key in IMAGE_KEYS
            }
            # Keep a short, valid token prefix and pad the remainder.  The
            # target's mask and token sequence never change across batches.
            tokens = torch.zeros(self.config.max_token_len, dtype=torch.long)
            prompt_length = 8 + index % 5
            tokens[:prompt_length] = torch.randint(
                1,
                32_000,
                (prompt_length,),
                generator=generator,
                dtype=torch.long,
            )
            token_mask = torch.zeros(self.config.max_token_len, dtype=torch.bool)
            token_mask[:prompt_length] = True
            examples.append(
                {
                    "images": images,
                    "image_masks": {
                        key: torch.tensor(True, dtype=torch.bool) for key in IMAGE_KEYS
                    },
                    "state": torch.randn(self.config.action_dim, generator=generator),
                    "tokenized_prompt": tokens,
                    "tokenized_prompt_mask": token_mask,
                    "noise": torch.randn(
                        self.config.action_horizon,
                        self.config.action_dim,
                        generator=generator,
                    ),
                }
            )
        return examples

    def run_model(self, model, batch):
        def to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            return {key: to_device(item) for key, item in value.items()}

        device_batch = to_device(batch)
        observation = SimpleNamespace(
            images=device_batch["images"],
            image_masks=device_batch["image_masks"],
            state=device_batch["state"],
            tokenized_prompt=device_batch["tokenized_prompt"],
            tokenized_prompt_mask=device_batch["tokenized_prompt_mask"],
            token_ar_mask=None,
            token_loss_mask=None,
        )
        return model.sample_actions(
            self.device,
            observation,
            noise=device_batch["noise"],
            num_steps=int(os.environ.get("OPENPI_NUM_STEPS", "10")),
        )

    def extract_robot_output(self, output):
        return output

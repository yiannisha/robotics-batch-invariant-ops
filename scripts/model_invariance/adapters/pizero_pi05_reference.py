"""Pi0.5 adapter for the PyTorch implementation vendored by batch-invariant-pizero.

Required environment variables:

``PIZERO_PYTORCH_SOURCE``
    Path to the vendored ``pi-zero-pytorch`` checkout.
``PIZERO_PI05_CHECKPOINT``
    Path to the public ``lerobot/pi05_base`` checkpoint directory containing
    ``config.json`` and ``model.safetensors``.

The upstream helper writes a second, converted ``pizero.pt`` file.  This
adapter instead copies the safetensors weights directly into the constructed
model, avoiding a redundant 14 GB on-disk copy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import torch


class PiZeroPi05ReferenceAdapter:
    name = "pi-zero-pytorch-pi05-public-weights"

    def __init__(self) -> None:
        self.source = Path(os.environ.get("PIZERO_PYTORCH_SOURCE", ""))
        self.checkpoint = Path(os.environ.get("PIZERO_PI05_CHECKPOINT", ""))
        self.device = torch.device("cuda")
        self.config: dict | None = None
        self.model = None

    def _validate_paths(self) -> None:
        if not (self.source / "pi_zero_pytorch/pi_zero.py").is_file():
            raise FileNotFoundError("set PIZERO_PYTORCH_SOURCE to the pi-zero-pytorch checkout")
        for filename in ("config.json", "model.safetensors"):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set PIZERO_PI05_CHECKPOINT to a checkpoint containing {filename}"
                )

    def load_model(self):
        self._validate_paths()
        sys.path.insert(0, str(self.source))

        from pi_zero_pytorch import PiZero, SigLIP
        from pi_zero_pytorch.load import (
            build_converted_state_dict,
            create_pizero_config_for_pi0,
        )
        from safetensors import safe_open

        self.config = json.loads((self.checkpoint / "config.json").read_text())
        if self.config.get("type") != "pi05":
            raise ValueError(f"expected a pi05 checkpoint, got {self.config.get('type')!r}")

        model_config = create_pizero_config_for_pi0(self.config)
        model_config.update(
            vit=SigLIP(norm_eps=model_config["norm_eps"]),
            vit_dim=1152,
        )
        self.model = PiZero(**model_config)
        try:
            with safe_open(
                self.checkpoint / "model.safetensors", framework="pt", device="cpu"
            ) as weights:
                # state_dict tensors alias the module parameters, so the converter's
                # copy_ calls populate the model without a second serialized copy.
                build_converted_state_dict(weights, self.model.state_dict())
        except (KeyError, RuntimeError) as error:
            raise RuntimeError(
                "the pinned pi-zero-pytorch source needs "
                "scripts/model_invariance/patches/pi_zero_pytorch_pi05_compat.patch"
            ) from error

        dtype_name = self.config.get("dtype", "float32")
        dtype = getattr(torch, dtype_name)
        self.model.eval().to(device=self.device, dtype=dtype)
        return self.model

    def load_example_inputs(self, count: int):
        if self.config is None:
            raise RuntimeError("load_model must be called before load_example_inputs")

        examples = []
        token_length = int(self.config["tokenizer_max_length"])
        trajectory_length = int(self.config["chunk_size"])
        action_dim = int(self.config["max_action_dim"])
        state_dim = int(self.config["max_state_dim"])
        for index in range(count):
            generator = torch.Generator().manual_seed(30_000 + index)
            examples.append(
                {
                    # pi-zero-pytorch accepts multiple views as [C, F, H, W].
                    "images": torch.rand(3, 3, 224, 224, generator=generator, dtype=torch.float32)
                    * 2
                    - 1,
                    "token_ids": torch.randint(
                        1, 32_000, (token_length,), generator=generator, dtype=torch.long
                    ),
                    "joint_state": torch.randn(state_dim, generator=generator),
                    "noise": torch.randn(trajectory_length, action_dim, generator=generator),
                }
            )
        return examples

    def run_model(self, model, batch):
        device_batch = {key: value.to(self.device) for key, value in batch.items()}
        explicit_noise = device_batch["noise"]
        original_randn = torch.randn

        def controlled_randn(*args, **kwargs):
            requested_shape = (
                tuple(args[0]) if len(args) == 1 and isinstance(args[0], tuple) else tuple(args)
            )
            if requested_shape == tuple(explicit_noise.shape):
                return explicit_noise
            return original_randn(*args, **kwargs)

        with mock.patch.object(torch, "randn", controlled_randn):
            output, used_noise = model.sample_actions(
                device_batch["images"],
                device_batch["token_ids"],
                device_batch["joint_state"],
                trajectory_length=explicit_noise.shape[1],
                return_original_noise=True,
                steps=int(os.environ.get("PIZERO_PI05_STEPS", "10")),
                show_pbar=False,
                cache_kv=True,
            )
        if not torch.equal(used_noise, explicit_noise):
            raise RuntimeError("Pi0.5 inference did not consume the supplied per-sample noise")
        return output

    def extract_robot_output(self, output):
        return output


adapter = PiZeroPi05ReferenceAdapter()

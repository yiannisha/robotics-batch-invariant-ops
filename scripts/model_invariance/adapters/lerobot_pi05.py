"""Official LeRobot PI0.5 adapter over the frozen real LIBERO replay.

The public ``lerobot/pi05_base`` checkpoint has three camera features.  The
LIBERO replay supplies two genuine camera streams; LeRobot's official
``_preprocess_images`` path supplies and masks the missing third camera, exactly
as it does for any deployment with fewer cameras than the checkpoint schema.
Every request carries an explicit frozen 50x32 flow-noise tensor.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class LeRobotPi05Adapter:
    name = "lerobot-pi05-base-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("LEROBOT_PI05_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("LEROBOT_PI05_SAMPLE_DIR", ""))
        self.replay_manifest_path = Path(
            os.environ.get(
                "PI05_REPLAY_MANIFEST", str(self.sample_dir / "replay_manifest.json")
            )
        )
        self.tokenizer = Path(os.environ.get("PI05_TEXT_TOKENIZER", ""))
        self.device = torch.device("cuda")
        self.config = None
        self.preprocessor = None
        self.postprocessor = None
        self.replay_manifest = None
        self.replay_noise = None
        self.fixture_root = None
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
                    f"set LEROBOT_PI05_CHECKPOINT to a snapshot containing {filename}"
                )
        if not self.replay_manifest_path.is_file():
            raise FileNotFoundError("set PI05_REPLAY_MANIFEST to the frozen replay manifest")
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            if not (self.tokenizer / filename).is_file():
                raise FileNotFoundError(
                    f"set PI05_TEXT_TOKENIZER to a local PaliGemma tokenizer containing {filename}"
                )

    def load_model(self):
        self._validate_paths()
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from safetensors.torch import load_file

        config = PI05Config.from_pretrained(self.checkpoint)
        config.device = str(self.device)
        config.compile_model = False
        config.gradient_checkpointing = False
        config.text_tokenizer_name = str(self.tokenizer)
        policy = PI05Policy(config)

        # PI05Policy.from_pretrained deliberately catches weight-loading errors
        # and may return a randomly initialized model.  Full-coverage evidence
        # must fail closed, so perform its official remapping explicitly and
        # require a strict load here.
        source = load_file(self.checkpoint / "model.safetensors", device="cpu")
        fixed = policy._fix_pytorch_state_dict_keys(source, config)
        remapped = {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in fixed.items()
        }
        remapped = policy._prepare_pretrained_state_dict(remapped)
        policy.load_state_dict(remapped, strict=True)
        del source, fixed, remapped

        self.config = policy.config
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(self.tokenizer)},
                "device_processor": {"device": "cpu"},
            },
        )
        policy.to(self.device).eval()
        return policy

    @staticmethod
    def _image(path: Path) -> torch.Tensor:
        array = np.array(Image.open(path).convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).float().div(255)

    def _load_replay_metadata(self) -> None:
        if self.replay_manifest is not None:
            return
        self.replay_manifest = json.loads(self.replay_manifest_path.read_text())
        self.fixture_root = Path(self.replay_manifest["fixture_root"])
        noise_path = Path(self.replay_manifest["noise_file"])
        if not noise_path.is_absolute():
            noise_path = self.replay_manifest_path.parent / noise_path
        self.replay_noise = np.load(noise_path, mmap_mode="r")
        if self.replay_noise.shape != (1000, 3, 50, 32):
            raise ValueError(f"unexpected PI0.5 replay noise shape {self.replay_noise.shape}")

    @lru_cache(maxsize=32)
    def _processed_replay_observation(self, replay_index: int) -> dict[str, torch.Tensor]:
        row = self.replay_manifest["observations"][replay_index]
        image_features = list(self.config.image_features)
        if len(image_features) != 3:
            raise ValueError(f"PI0.5 checkpoint exposes {len(image_features)} camera features")
        processed = self.preprocessor(
            {
                image_features[0]: self._image(self.fixture_root / row["images"]["image"]),
                image_features[1]: self._image(self.fixture_root / row["images"]["image2"]),
                "observation.state": torch.tensor(row["state"], dtype=torch.float32),
                "task": row["task"],
            }
        )
        return {
            key: value.detach().cpu()
            for key, value in processed.items()
            if isinstance(value, torch.Tensor) and key.startswith("observation.")
        }

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.config is None or self.preprocessor is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        self._load_replay_metadata()
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows) or not 0 <= noise_id < 3:
            raise IndexError((replay_index, noise_id))
        noise = np.array(self.replay_noise[replay_index, noise_id : noise_id + 1], copy=True)
        return self._processed_replay_observation(replay_index) | {
            "noise": torch.from_numpy(noise)
        }

    def load_example_inputs(self, count: int):
        if count < 1:
            raise ValueError("count must be positive")
        return [self.load_replay_input(index) for index in range(count)]

    @staticmethod
    def compose_batch(samples):
        return {
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in samples[0]
        }

    def compose_fixed_batch(self, samples):
        return self.compose_batch(samples)

    def fixed_batch_shape_metadata(self, batch):
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

        return {
            "language_padding": "max_length_right",
            "language_tokens": batch[OBS_LANGUAGE_TOKENS].shape[1],
            "canonical_length": self.config.tokenizer_max_length,
        }

    def execution_configuration(self):
        if self.config is None:
            raise RuntimeError("load_model must be called before execution_configuration")
        return {
            "language_padding": "max_length_right",
            "language_max_length": self.config.tokenizer_max_length,
            "camera_features": list(self.config.image_features),
            "missing_camera_contract": "official_zero_image_false_mask",
            "flow_steps": self.config.num_inference_steps,
            "chunk_size": self.config.chunk_size,
            "max_action_dim": self.config.max_action_dim,
        }

    @staticmethod
    def reset_model(policy) -> None:
        policy.reset()

    def run_model(self, policy, batch):
        from lerobot.policies.common.vla_utils import (
            make_att_2d_masks,
            prepare_attention_masks_4d,
        )
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        model_batch = {
            key: value.to(self.device) for key, value in batch.items() if key != "noise"
        }
        images, image_masks = policy._preprocess_images(model_batch)
        tokens = model_batch[OBS_LANGUAGE_TOKENS]
        token_masks = model_batch[OBS_LANGUAGE_ATTENTION_MASK]
        for index, image in enumerate(images):
            self._trace(f"preprocess.image{index}", image)
        self._trace("preprocess.tokens", tokens)

        model = policy.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, image_masks, tokens, token_masks
        )
        self._trace("prefix.embeddings", prefix_embs)
        prefix_attention = prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
            "eager"
        )
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        actions = batch["noise"].to(self.device)
        self._trace("input.noise", actions)
        step_size = -1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
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

        action_dimension = self.config.output_features["action"].shape[0]
        actions = actions[..., :action_dimension]
        self._trace("actions.normalized", actions)
        actions = self.postprocessor(actions.detach().cpu()).to(self.device)
        self._trace("actions.unnormalized", actions)
        return actions

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = LeRobotPi05Adapter()

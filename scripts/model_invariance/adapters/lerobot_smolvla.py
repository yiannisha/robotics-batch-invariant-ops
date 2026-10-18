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
import json
from functools import lru_cache
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
        self.replay_manifest = None
        self.replay_noise = None
        self.language_pad_token_id = None
        # The official native path retains longest-in-batch padding. The
        # repaired invariant path uses a fixed masked language extent so its
        # attention shape does not depend on unrelated task prompt lengths.
        self.canonical_prompt_padding = os.environ.get("SMOLVLA_DISABLE_CANONICAL_PROMPT_PADDING") != "1"

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        if (self.sample_dir / "replay_manifest.json").is_file():
            if not (self.sample_dir / "initial_action_noise.npy").is_file():
                raise FileNotFoundError("full replay fixture lacks stored initial-action noise")
            return
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
        policy = SmolVLAPolicy.from_pretrained(self.checkpoint, config=config, strict=True)
        self.config = policy.config
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.language_pad_token_id = policy.model.vlm_with_expert.processor.tokenizer.pad_token_id
        if self.language_pad_token_id is None:
            raise ValueError("the pinned SmolVLA tokenizer has no language pad token")
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

    @staticmethod
    def _image_path(path: Path) -> torch.Tensor:
        array = np.array(Image.open(path).convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).float().div(255)

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        """Load one frozen request lazily; never materialize all 1,000 images."""
        if self.config is None or self.preprocessor is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        if self.replay_manifest is None:
            self.replay_manifest = json.loads(
                (self.sample_dir / "replay_manifest.json").read_text()
            )
            self.replay_noise = np.load(
                self.sample_dir / self.replay_manifest["noise_file"], mmap_mode="r"
            )
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows) or not 0 <= noise_id < 3:
            raise IndexError((replay_index, noise_id))
        return self._processed_replay_observation(replay_index) | {
            "noise": torch.from_numpy(
                np.array(self.replay_noise[replay_index, noise_id : noise_id + 1], copy=True)
            )
        }

    @lru_cache(maxsize=32)
    def _processed_replay_observation(self, replay_index: int):
        row = self.replay_manifest["observations"][replay_index]
        processed = self.preprocessor(
            {
                "observation.images.image": self._image_path(
                    self.sample_dir / row["images"]["image"]
                ),
                "observation.images.image2": self._image_path(
                    self.sample_dir / row["images"]["image2"]
                ),
                "observation.state": torch.tensor(row["state"], dtype=torch.float32),
                "task": row["task"],
            }
        )
        return {
            key: value.detach().cpu()
            for key, value in processed.items()
            if isinstance(value, torch.Tensor) and key.startswith("observation.")
        }

    def load_example_inputs(self, count: int):
        if self.config is None or self.preprocessor is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if (self.sample_dir / "replay_manifest.json").is_file():
            return [self.load_replay_input(index) for index in range(count)]
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
                    if isinstance(value, torch.Tensor) and key.startswith("observation.")
                }
                | {"noise": noise}
            )
        return examples

    def compose_batch(self, samples, *, fixed_prompt: bool = False):
        from batch_invariant_ops import is_batch_invariant_mode_enabled
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        # The official processor uses right-padding to the longest prompt in
        # each batch. A singleton preprocessed request has no padding, so
        # unrelated LIBERO tasks must be padded when composed here.
        max_tokens = (
            self.config.tokenizer_max_length
            if fixed_prompt or self.canonical_prompt_padding and is_batch_invariant_mode_enabled() else
            max(sample[OBS_LANGUAGE_TOKENS].shape[1] for sample in samples)
        )
        if any(sample[OBS_LANGUAGE_TOKENS].shape[1] > max_tokens for sample in samples):
            raise ValueError("preprocessed prompt exceeds the selected pad length")
        result = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
                pad_value = self.language_pad_token_id if key == OBS_LANGUAGE_TOKENS else 0
                values = [
                    torch.nn.functional.pad(value, (0, max_tokens - value.shape[1]), value=pad_value)
                    for value in values
                ]
            result[key] = torch.cat(values, dim=0)
        return result

    def compose_fixed_batch(self, samples):
        return self.compose_batch(samples, fixed_prompt=True)

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
            "native_language_padding": self.config.pad_language_to,
            "invariant_language_padding": (
                "max_length" if self.canonical_prompt_padding else self.config.pad_language_to
            ),
            "language_max_length": self.config.tokenizer_max_length,
            "language_pad_token_id": self.language_pad_token_id,
        }

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

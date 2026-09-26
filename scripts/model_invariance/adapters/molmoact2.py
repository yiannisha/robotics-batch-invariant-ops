"""MolmoAct2 adapter using the official PyTorch LIBERO checkpoint.

Required environment variables:

``MOLMOACT2_CHECKPOINT``
    Local ``allenai/MolmoAct2-LIBERO`` snapshot.
``MOLMOACT2_SAMPLE_DIR``
    LIBERO fixture with the two camera frame directories and episode parquet
    used by the other LIBERO adapters.

The released model's native attention-bias builder allocates a leading batch
dimension of one and updates it in place with the per-example image mask.  It
therefore raises for every B > 1.  This adapter installs the minimal equivalent
implementation with that mask expanded to B before the in-place update.  The
B=1 attention bias is bitwise identical to the released implementation.

The public continuous-action API accepts a CUDA generator rather than an
explicit trajectory.  For the checkpoint's 10 x 32 trajectory, freshly seeded
CUDA generators produce an identical first-sample trajectory for all tested
batch sizes.  A fresh generator is consequently constructed for every call.
"""

from __future__ import annotations

import importlib
import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

TASK = "put the white mug on the left plate and put the yellow and white mug " "on the right plate"


class MolmoAct2Adapter:
    name = "molmoact2-libero-official-pytorch-public-weights"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("MOLMOACT2_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("MOLMOACT2_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.processor = None
        self.model = None
        self.modeling_module = None
        self.stats = None
        self.metadata = None

    def _validate_paths(self) -> None:
        for root, filenames in (
            (
                self.checkpoint,
                (
                    "config.json",
                    "model.safetensors.index.json",
                    "norm_stats.json",
                    "modeling_molmoact2.py",
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

    @staticmethod
    def _batched_native_attention_bias(
        core: Any,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Released attention-bias logic with its B>1 broadcast bug fixed."""

        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.to(device=inputs_embeds.device)

        batch_size, seq_len = inputs_embeds.shape[:2]
        # Continuous action inference calls this only for the initial VLM
        # prefill, without an existing cache.
        if past_key_values is not None:
            return core.__class__._build_native_attention_bias(
                core,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
            )

        device = inputs_embeds.device
        if attention_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        else:
            valid_mask = attention_mask.to(device=device, dtype=torch.bool)

        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))[None, None]
            .expand(batch_size, -1, -1, -1)
            .clone()
        )
        if token_type_ids is not None:
            image_mask = token_type_ids.to(device=device, dtype=torch.bool)
            can_attend_back = image_mask[:, :, None] & image_mask[:, None, :]
            causal_mask |= can_attend_back[:, None]

        allowed = valid_mask[:, None, None, :] & causal_mask
        return torch.where(
            allowed,
            torch.zeros((), device=device, dtype=inputs_embeds.dtype),
            torch.full(
                (),
                torch.finfo(inputs_embeds.dtype).min,
                device=device,
                dtype=inputs_embeds.dtype,
            ),
        )

    def load_model(self):
        self._validate_paths()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
        self.modeling_module = importlib.import_module(type(self.model).__module__)
        self.stats = self.model._get_robot_stats()
        self.metadata = self.stats.get_metadata("libero")

        core = self.model.model
        core._build_native_attention_bias = types.MethodType(
            self._batched_native_attention_bias, core
        )
        core.action_cuda_graph_manager.set_enabled(False)
        self.model.depth_decode_cuda_graph_manager.set_enabled(False)
        return self.model

    def _load_image(self, filename: str) -> Image.Image:
        return Image.open(self.sample_dir / filename).convert("RGB")

    def load_example_inputs(self, count: int):
        if self.model is None or self.processor is None or self.stats is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")

        target_frame = int(os.environ.get("MOLMOACT2_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("MOLMOACT2_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)
        frame_indices = frame_indices[:count]

        state_rows = pq.read_table(
            self.sample_dir / "data/chunk-000/file-000.parquet",
            columns=["observation.state"],
        )["observation.state"].to_pylist()
        normalized_task = self.modeling_module._normalize_question_text(TASK)
        examples = []
        for frame_index in frame_indices:
            state = np.asarray(state_rows[frame_index], dtype=np.float32)
            normalized_state = np.asarray(
                self.stats.normalize_state(state, "libero"), dtype=np.float32
            )
            discrete_state = self.modeling_module._build_discrete_state_string(
                normalized_state, int(self.model.config.num_state_tokens)
            )
            text = self.modeling_module._build_robot_text(
                task=normalized_task,
                style="robot_action",
                discrete_state_string=discrete_state,
                setup_type=str(self.metadata.get("setup_type", "")),
                control_mode=str(self.metadata.get("control_mode", "")),
                add_setup_tokens=bool(self.model.config.add_setup_tokens),
                add_control_tokens=bool(self.model.config.add_control_tokens),
                num_images=2,
            )
            image_number = frame_index + 1
            processed = self.processor(
                text=text,
                images=[
                    self._load_image(f"frames/image_{image_number:03d}.png"),
                    self._load_image(f"frames/image2_{image_number:03d}.png"),
                ],
                return_tensors="pt",
            )
            examples.append(dict(processed))
        return examples

    @staticmethod
    def compose_batch(samples):
        sequence_keys = {"input_ids", "attention_mask", "token_type_ids"}
        return {
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in samples[0]
            if key in sequence_keys
            or key
            in {
                "pixel_values",
                "image_token_pooling",
                "image_grids",
                "image_num_crops",
            }
        }

    def reset_model(self, model) -> None:
        expert = model.model.action_expert
        expert._modulation_cache_key = None
        expert._modulation_cache_value = None

    def run_model(self, model, batch):
        inputs = {key: value.to(self.device) for key, value in batch.items()}
        batch_size = int(inputs["input_ids"].shape[0])
        action_dim = int(self.stats.get_action_dim("libero"))
        action_horizon = int(self.stats.get_action_horizon("libero"))
        action_dim_is_pad = model._build_action_dim_is_pad(
            action_dim=action_dim,
            max_action_dim=int(model.config.max_action_dim),
            batch_size=batch_size,
            device=self.device,
        )
        generator = torch.Generator(device=self.device).manual_seed(271_828)
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            actions = model.model.generate_actions_from_inputs(
                **inputs,
                action_dim_is_pad=action_dim_is_pad,
                action_horizon=action_horizon,
                num_steps=int(os.environ.get("MOLMOACT2_STEPS", "10")),
                generator=generator,
            )
        actions = model._slice_action_dim(actions, action_dim)
        actions = model._slice_action_chunk(
            actions,
            int(model.config.n_obs_steps),
            int(self.stats.get_n_action_steps("libero")),
        )
        actions = self.stats.unnormalize_action(actions, "libero")
        return actions.to(device=self.device, dtype=torch.float32)

    def extract_robot_output(self, output):
        return output


adapter = MolmoAct2Adapter()

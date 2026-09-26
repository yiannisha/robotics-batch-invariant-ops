"""Official OpenVLA-OFT LIBERO-Spatial PyTorch adapter.

Required environment variables:

``OPENVLA_OFT_CHECKPOINT``
    Local snapshot of ``moojink/openvla-7b-oft-finetuned-libero-spatial``.
``OPENVLA_OFT_SAMPLE_DIR``
    Real LIBERO episode-zero fixture with two camera streams, parquet state
    data, and ``tasks.parquet``.

The released ``predict_action`` helper reshapes away its leading dimension and
therefore only supports B=1.  This adapter follows the same official PyTorch
operations while preserving that dimension for genuine batched inference.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class _Policy:
    vla: torch.nn.Module
    action_head: torch.nn.Module
    proprio_projector: torch.nn.Module

    def eval(self):
        self.vla.eval()
        self.action_head.eval()
        self.proprio_projector.eval()
        return self


class OpenVLAOFTAdapter:
    name = "openvla-oft-libero-spatial-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("OPENVLA_OFT_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("OPENVLA_OFT_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.processor = None
        self.stats: dict[str, Any] | None = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        for filename in (
            "config.json",
            "model.safetensors.index.json",
            "action_head--150000_checkpoint.pt",
            "proprio_projector--150000_checkpoint.pt",
            "dataset_statistics.json",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set OPENVLA_OFT_CHECKPOINT to a snapshot containing {filename}"
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
                    f"set OPENVLA_OFT_SAMPLE_DIR to a directory containing {filename}"
                )

    @staticmethod
    def _component_state(path: Path) -> dict[str, torch.Tensor]:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return {
            key.removeprefix("module."): value
            for key, value in state.items()
        }

    def load_model(self):
        self._validate_paths()
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector

        vla = AutoModelForVision2Seq.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        vla.vision_backbone.set_num_images_in_input(2)
        vla.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.stats = json.loads(
            (self.checkpoint / "dataset_statistics.json").read_text()
        )["libero_spatial_no_noops"]
        vla.norm_stats = {"libero_spatial_no_noops": self.stats}

        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=7
        ).to(device=self.device, dtype=torch.bfloat16)
        action_head.load_state_dict(
            self._component_state(self.checkpoint / "action_head--150000_checkpoint.pt"),
            strict=True,
        )
        proprio_projector = ProprioProjector(
            llm_dim=vla.llm_dim, proprio_dim=8
        ).to(device=self.device, dtype=torch.bfloat16)
        proprio_projector.load_state_dict(
            self._component_state(
                self.checkpoint / "proprio_projector--150000_checkpoint.pt"
            ),
            strict=True,
        )
        return _Policy(vla, action_head, proprio_projector).eval()

    @staticmethod
    def _center_crop(image: Image.Image) -> Image.Image:
        """PyTorch equivalent of the released centered 90%-area crop."""

        if image.size != (224, 224):
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
        array = np.array(image.convert("RGB"), copy=True)
        value = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div(255)
        scale = math.sqrt(0.9)
        theta = torch.tensor(
            [[[scale, 0.0, 0.0], [0.0, scale, 0.0]]], dtype=torch.float32
        )
        grid = F.affine_grid(theta, value.shape, align_corners=True)
        cropped = F.grid_sample(
            value, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        result = (
            cropped.squeeze(0)
            .mul(255)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        return Image.fromarray(result, mode="RGB")

    def _processed_image(self, camera: str, frame_number: int) -> Image.Image:
        image = Image.open(
            self.sample_dir / "frames" / f"{camera}_{frame_number:03d}.png"
        ).convert("RGB")
        return self._center_crop(image)

    def load_example_inputs(self, count: int):
        if self.processor is None or self.stats is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")

        target_frame = int(os.environ.get("OPENVLA_OFT_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("OPENVLA_OFT_TARGET_FRAME must be between 0 and 63")
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
        low = np.asarray(self.stats["proprio"]["q01"])
        high = np.asarray(self.stats["proprio"]["q99"])

        examples = []
        for frame_index in frame_indices[:count]:
            row = rows[frame_index]
            task = tasks[int(row["task_index"])]
            prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
            primary = self.processor(
                prompt, self._processed_image("image", frame_index + 1)
            )
            wrist = self.processor(
                prompt, self._processed_image("image2", frame_index + 1)
            )
            if not torch.equal(primary["input_ids"], wrist["input_ids"]):
                raise RuntimeError("primary and wrist prompt tokenization differs")
            state = np.asarray(row["observation.state"])
            state = np.clip(
                2 * (state - low) / (high - low + 1e-8) - 1,
                a_min=-1.0,
                a_max=1.0,
            )
            examples.append(
                {
                    "input_ids": primary["input_ids"],
                    "attention_mask": primary["attention_mask"],
                    "pixel_values": torch.cat(
                        [primary["pixel_values"], wrist["pixel_values"]], dim=1
                    ),
                    "proprio": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                }
            )
        return examples

    @staticmethod
    def compose_batch(samples):
        return {
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in samples[0]
        }

    def run_model(self, policy: _Policy, batch):
        from prismatic.vla.constants import IGNORE_INDEX

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device, dtype=torch.bfloat16)
        proprio = batch["proprio"].to(self.device, dtype=torch.bfloat16)
        batch_size = input_ids.shape[0]

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                [input_ids, torch.full_like(input_ids[:, :1], 29871)], dim=1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
            )
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        prompt_tokens = input_ids.shape[-1] - 1
        input_ids, attention_mask = policy.vla._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = policy.vla._prepare_labels_for_action_prediction(labels, input_ids)
        input_embeddings = policy.vla.get_input_embeddings()(input_ids)
        action_mask = policy.vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~action_mask].reshape(
            batch_size, -1, input_embeddings.shape[-1]
        )
        self._trace("input.embeddings", input_embeddings)

        projected = policy.vla._process_vision_features(
            pixel_values, language_embeddings, False
        )
        self._trace("vision.projected", projected)
        projected = policy.vla._process_proprio_features(
            projected, proprio, policy.proprio_projector
        )
        self._trace("conditioning.projected", projected)

        input_embeddings = input_embeddings * ~action_mask.unsqueeze(-1)
        multimodal, multimodal_mask = policy.vla._build_multimodal_attention(
            input_embeddings, projected, attention_mask
        )
        self._trace("language_model.input", multimodal)
        output = policy.vla.language_model(
            input_ids=None,
            attention_mask=multimodal_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        self._trace("language_model.output", hidden)
        patch_tokens = policy.vla.vision_backbone.get_num_patches() * 2 + 1
        action_hidden = hidden[
            :,
            patch_tokens + prompt_tokens : patch_tokens + prompt_tokens + 56,
            :,
        ]
        self._trace("actions.hidden", action_hidden)
        normalized = policy.action_head.predict_action(action_hidden).reshape(
            batch_size, 8, 7
        )
        self._trace("actions.normalized", normalized)

        normalized_numpy = normalized.detach().float().cpu().numpy()
        stats = self.stats["action"]
        low = np.asarray(stats["q01"])
        high = np.asarray(stats["q99"])
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)))
        actions = np.where(
            mask,
            0.5 * (normalized_numpy + 1) * (high - low + 1e-8) + low,
            normalized_numpy,
        )
        result = torch.from_numpy(actions)
        self._trace("actions.unnormalized", result)
        return result

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = OpenVLAOFTAdapter()

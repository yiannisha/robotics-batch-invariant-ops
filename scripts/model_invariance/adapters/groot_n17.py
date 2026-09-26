"""GR00T N1.7 adapter using NVIDIA's official PyTorch implementation.

Required environment variables:

``GROOT_N17_CHECKPOINT``
    Local ``nvidia/GR00T-N1.7-LIBERO/libero_10`` checkpoint directory.
``GROOT_N17_PROCESSOR``
    Local processor/config-only snapshot of ``Qwen/Qwen3-VL-2B-Instruct``.
    NVIDIA's separately licensed Cosmos-Reason2 repository cannot be fetched
    without accepting its license. Cosmos-Reason2 uses this exact public
    Qwen3-VL architecture and tokenizer. No Qwen model weights are used: the
    complete backbone and action-head weights are loaded directly from the
    NVIDIA GR00T checkpoint.
``GROOT_N17_SAMPLE_DIR``
    LIBERO fixture with ``frames/image_*.png``, ``frames/image2_*.png``, and
    ``data/chunk-000/file-000.parquet``.

The NVIDIA constructor normally resolves the separately gated Cosmos base
before overwriting it with the GR00T checkpoint. During construction only,
the adapter substitutes an uninitialized Qwen3-VL module built from the exact
public architecture config and substitutes the matching public processor.
Hugging Face then loads NVIDIA's two checkpoint shards into NVIDIA's official
``Gr00tN1d7`` classes. This is a loader workaround, not a model conversion.

Flow noise is an explicit per-example adapter input. This avoids relying on
batch-shaped RNG calls when testing the model's four-step action flow.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

TASK = "put the white mug on the left plate and put the yellow and white mug on the right plate"
STATE_SLICES = {
    "x": slice(0, 1),
    "y": slice(1, 2),
    "z": slice(2, 3),
    "roll": slice(3, 4),
    "pitch": slice(4, 5),
    "yaw": slice(5, 6),
    "gripper": slice(6, 8),
}
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _floating_to_bfloat16(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype=torch.bfloat16)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _floating_to_bfloat16(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_floating_to_bfloat16(item) for item in value]
    return value


class GrootN17Adapter:
    name = "nvidia-gr00t-n1.7-libero-official-pytorch-public-weights"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("GROOT_N17_CHECKPOINT", ""))
        self.processor_path = Path(os.environ.get("GROOT_N17_PROCESSOR", ""))
        self.sample_dir = Path(os.environ.get("GROOT_N17_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.processor = None
        self.collate_fn = None
        self.embodiment = None

    def _validate_paths(self) -> None:
        for root, filenames in (
            (
                self.checkpoint,
                (
                    "config.json",
                    "model.safetensors.index.json",
                    "processor_config.json",
                    "statistics.json",
                ),
            ),
            (
                self.processor_path,
                (
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "preprocessor_config.json",
                    "video_preprocessor_config.json",
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
                    raise FileNotFoundError(
                        f"missing required fixture: {root / filename}"
                    )

    def load_model(self):
        self._validate_paths()
        import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing_module
        import gr00t.model.modules.qwen3_backbone as backbone_module
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
        from transformers import Qwen3VLConfig, Qwen3VLProcessor

        backbone_config = Qwen3VLConfig.from_pretrained(
            self.processor_path, local_files_only=True
        )
        original_model_loader = inspect.getattr_static(
            backbone_module.Qwen3VLForConditionalGeneration, "from_pretrained"
        )
        original_processor_builder = processing_module.build_processor

        def construct_backbone_without_base_weights(_cls, _model_name, *args, **kwargs):
            del args
            allowed = {
                key: value
                for key, value in kwargs.items()
                if key in {"attn_implementation", "torch_dtype", "dtype"}
            }
            if "dtype" in allowed and "torch_dtype" not in allowed:
                allowed["torch_dtype"] = allowed.pop("dtype")
            return backbone_module.Qwen3VLForConditionalGeneration._from_config(
                backbone_config, **allowed
            ).eval()

        def build_public_processor(_model_name, _loading_kwargs):
            return Qwen3VLProcessor.from_pretrained(
                self.processor_path, local_files_only=True
            )

        backbone_module.Qwen3VLForConditionalGeneration.from_pretrained = classmethod(
            construct_backbone_without_base_weights
        )
        processing_module.build_processor = build_public_processor
        try:
            model = Gr00tN1d7.from_pretrained(
                self.checkpoint,
                local_files_only=True,
                transformers_loading_kwargs={
                    "trust_remote_code": True,
                    "local_files_only": True,
                },
            )
        finally:
            backbone_module.Qwen3VLForConditionalGeneration.from_pretrained = (
                original_model_loader
            )
            processing_module.build_processor = original_processor_builder

        # Match NVIDIA's Gr00tPolicy inference path exactly.
        model = model.to(device=self.device, dtype=self.dtype).eval()
        self.processor = Gr00tN1d7Processor.from_pretrained(
            self.checkpoint,
            model_name=str(self.processor_path),
            local_files_only=True,
            transformers_loading_kwargs={"local_files_only": True},
        )
        self.processor.eval()
        self.collate_fn = self.processor.collator
        self.embodiment = EmbodimentTag.LIBERO_PANDA
        return model

    def _load_image(self, filename: str) -> np.ndarray:
        return np.array(
            Image.open(self.sample_dir / filename).convert("RGB"), copy=True
        )

    def load_example_inputs(self, count: int):
        if self.processor is None or self.collate_fn is None or self.embodiment is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")

        from gr00t.data.types import MessageType, VLAStepData

        target_frame = int(os.environ.get("GROOT_N17_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("GROOT_N17_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)
        frame_indices = frame_indices[:count]
        state_rows = pq.read_table(
            self.sample_dir / "data/chunk-000/file-000.parquet",
            columns=["observation.state"],
        )["observation.state"].to_pylist()

        examples = []
        for frame_index in frame_indices:
            image_number = frame_index + 1
            raw_state = np.asarray(state_rows[frame_index], dtype=np.float32)
            states = {
                key: raw_state[state_slice][None]
                for key, state_slice in STATE_SLICES.items()
            }
            step = VLAStepData(
                images={
                    "image": self._load_image(f"frames/image_{image_number:03d}.png")[
                        None
                    ],
                    "wrist_image": self._load_image(
                        f"frames/image2_{image_number:03d}.png"
                    )[None],
                },
                states=states,
                actions={},
                text=TASK,
                embodiment=self.embodiment,
            )
            processed = self.processor(
                [{"type": MessageType.EPISODE_STEP.value, "content": step}]
            )
            generator = torch.Generator(device="cpu").manual_seed(71_000 + frame_index)
            noise = torch.randn(40, 132, generator=generator, dtype=self.dtype)
            examples.append({"processed": processed, "states": states, "noise": noise})
        return examples

    def compose_batch(self, samples):
        if self.collate_fn is None:
            raise RuntimeError("load_model must initialize the collator")
        collated = self.collate_fn([sample["processed"] for sample in samples])
        states = {
            key: np.stack([sample["states"][key] for sample in samples], axis=0)
            for key in STATE_SLICES
        }
        return {
            "inputs": _floating_to_bfloat16(collated["inputs"]),
            "states": states,
            "noise": torch.stack([sample["noise"] for sample in samples], dim=0),
        }

    @staticmethod
    def reset_model(model) -> None:
        del model

    def _get_action_with_noise(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Run NVIDIA's released four-step flow with explicit initial noise."""

        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_output = model.backbone(backbone_inputs)
        head = model.action_head
        features = head._encode_features(backbone_output, action_inputs)
        vl_embeds = features.backbone_features
        state_features = features.state_features
        embodiment_id = action_inputs.embodiment_id
        actions = noise.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
        dt = 1.0 / head.num_inference_timesteps

        for step in range(head.num_inference_timesteps):
            continuous_time = step / float(head.num_inference_timesteps)
            timestep = int(continuous_time * head.num_timestep_buckets)
            timesteps = torch.full(
                (actions.shape[0],), timestep, device=actions.device, dtype=torch.long
            )
            action_features = head.action_encoder(actions, timesteps, embodiment_id)
            if head.config.add_pos_embed:
                position_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=actions.device
                )
                action_features = action_features + head.position_embedding(
                    position_ids
                ).unsqueeze(0)
            state_action = torch.cat((state_features, action_features), dim=1)
            if head.config.use_alternate_vl_dit:
                model_output = head.model(
                    hidden_states=state_action,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = head.model(
                    hidden_states=state_action,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps,
                )
            velocity = head.action_decoder(model_output, embodiment_id)
            actions = actions + dt * velocity[:, -head.action_horizon :]
        return actions

    def run_model(self, model, batch):
        if self.processor is None or self.embodiment is None:
            raise RuntimeError("load_model must initialize the processor")
        normalized = self._get_action_with_noise(model, batch["inputs"], batch["noise"])
        decoded = self.processor.decode_action(
            normalized.float().cpu().numpy(), self.embodiment, batch["states"]
        )
        actions = np.concatenate([decoded[key] for key in ACTION_KEYS], axis=-1).astype(
            np.float32
        )
        # Match the official LIBERO environment wrapper: [0, 1] -> [-1, 1],
        # binarize, then invert the gripper sign.
        actions[..., -1] = -np.sign(2.0 * actions[..., -1] - 1.0)
        return torch.from_numpy(actions).to(self.device)

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = GrootN17Adapter()

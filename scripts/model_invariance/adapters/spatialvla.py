"""Official SpatialVLA 4B PyTorch policy adapter.

Required environment variables:

``SPATIALVLA_CHECKPOINT``
    Local ``IPEC-COMMUNITY/spatialvla-4b-224-pt`` snapshot.
``SPATIALVLA_SOURCE``
    Official ``SpatialVLA/SpatialVLA`` checkout.
``SPATIALVLA_SAMPLE_DIR``
    Real LIBERO fixture containing ``frames/image_*.png`` companions.

The target is the authors' published ``example.png`` and prompt.  Inference
uses the checkpoint's official Transformers remote code, BF16 weights, greedy
generation, Bridge action statistics, and unchanged 256-token generation cap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from batch_invariant_ops import is_batch_invariant_mode_enabled

PROMPT = "What action should the robot take to pick the cup?"
UNNORM_KEY = "bridge_orig/1.0.0"
CANONICAL_TEXT_LENGTH = 64


@dataclass
class _Policy:
    model: torch.nn.Module
    processor: Any

    def eval(self):
        self.model.eval()
        return self


class _SpatialBatch(list):
    def __init__(self, samples, *, fixed_prompt: bool):
        super().__init__(samples)
        self.fixed_prompt = fixed_prompt


class SpatialVLAAdapter:
    name = "spatialvla-4b-224-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("SPATIALVLA_CHECKPOINT", ""))
        self.source = Path(os.environ.get("SPATIALVLA_SOURCE", ""))
        self.sample_dir = Path(os.environ.get("SPATIALVLA_SAMPLE_DIR", ""))
        self.replay_manifest_path = Path(os.environ.get(
            "SPATIALVLA_REPLAY_MANIFEST",
            str(Path(__file__).resolve().parents[3]
                / "results/cross_model/spatialvla/full/replay_manifest.json"),
        ))
        self.device = torch.device("cuda")
        self.trace_callback = None
        self.replay_manifest = None
        self.processor = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        for filename in (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "modeling_spatialvla.py",
            "processing_spatialvla.py",
            "example.png",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    "set SPATIALVLA_CHECKPOINT to the official snapshot containing " f"{filename}"
                )
        if not (self.source / "README.md").is_file():
            raise FileNotFoundError("set SPATIALVLA_SOURCE to the official checkout")
        if not self.replay_manifest_path.is_file():
            for frame_number in (1, 63):
                filename = f"frames/image_{frame_number:03d}.png"
                if not (self.sample_dir / filename).is_file():
                    raise FileNotFoundError(
                        f"set SPATIALVLA_SAMPLE_DIR to the real LIBERO fixture containing {filename}"
                    )

    def load_model(self):
        self._validate_paths()
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = (
            AutoModel.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
        self.processor = processor
        return _Policy(model=model, processor=processor).eval()

    def load_example_inputs(self, count: int):
        if self.replay_manifest_path.is_file():
            return [self.load_replay_input(index) for index in range(count)]
        if count > 64:
            raise ValueError("the target plus public LIBERO fixture provide 64 examples")
        target = Image.open(self.checkpoint / "example.png").convert("RGB")
        examples = [target.copy()]
        for frame_number in range(1, count):
            with Image.open(self.sample_dir / "frames" / f"image_{frame_number:03d}.png") as image:
                examples.append(image.convert("RGB").copy())
        return examples

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if noise_id != 0:
            raise IndexError((replay_index, noise_id))
        if self.replay_manifest is None:
            self.replay_manifest = json.loads(self.replay_manifest_path.read_text())
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows):
            raise IndexError(replay_index)
        row = rows[replay_index]
        if row["replay_index"] != replay_index or len(row["request_ids"]) != 1:
            raise RuntimeError("SpatialVLA replay identity differs from frozen index")
        image_path = Path(self.replay_manifest["fixture_root"]) / row["images"]["image"]
        with Image.open(image_path) as image:
            frame = image.convert("RGB").copy()
        return {
            "image": frame,
            "prompt": row["task"],
            "unnorm_key": row["unnorm_key"],
            "policy_setup": row["policy_setup"],
            "replay_index": replay_index,
        }

    @staticmethod
    def compose_batch(samples):
        return _SpatialBatch(samples, fixed_prompt=False)

    @staticmethod
    def compose_fixed_batch(samples):
        return _SpatialBatch(samples, fixed_prompt=True)

    def fixed_batch_shape_metadata(self, batch):
        if self.processor is None:
            raise RuntimeError("load_model must be called before fixed-batch metadata")
        return {
            "language_padding": "max_length_right",
            "language_tokens": CANONICAL_TEXT_LENGTH + self.processor.image_seq_length,
            "text_tokens": CANONICAL_TEXT_LENGTH,
            "canonical_length": CANONICAL_TEXT_LENGTH,
        }

    @staticmethod
    def execution_configuration() -> dict[str, Any]:
        return {
            "native_language_padding": "longest_in_batch_right",
            "invariant_language_padding": "canonical_right",
            "canonical_text_length": CANONICAL_TEXT_LENGTH,
            "unnorm_key": UNNORM_KEY,
            "do_normalize": False,
            "generation": "greedy_max_new_tokens_256",
        }

    def prepare_batch(self, policy: _Policy, images):
        if images and isinstance(images[0], dict):
            prompts = [sample["prompt"] for sample in images]
            unnorm_keys = {sample["unnorm_key"] for sample in images}
            if unnorm_keys != {UNNORM_KEY}:
                raise ValueError(f"mixed or unsupported SpatialVLA unnorm keys: {unnorm_keys}")
            image_values = [sample["image"] for sample in images]
            processing = {
                "unnorm_key": UNNORM_KEY,
                "do_normalize": False,
                "padding": (
                    "max_length"
                    if is_batch_invariant_mode_enabled()
                    or getattr(images, "fixed_prompt", False) else True
                ),
            }
            if is_batch_invariant_mode_enabled() or getattr(images, "fixed_prompt", False):
                processing["max_length"] = CANONICAL_TEXT_LENGTH
        else:
            prompts = [PROMPT] * len(images)
            image_values = images
            processing = {"padding": True}
        inputs = policy.processor(
            images=image_values,
            text=prompts,
            return_tensors="pt",
            **processing,
        )
        inputs = inputs.to(torch.bfloat16).to(self.device)
        self._trace("input.input_ids", inputs["input_ids"])
        self._trace("input.pixel_values", inputs["pixel_values"])
        self._trace("input.intrinsic", inputs["intrinsic"])
        return inputs

    def generate(self, policy: _Policy, images, *, output_scores: bool = False):
        inputs = self.prepare_batch(policy, images)
        input_length = inputs["input_ids"].shape[-1]
        outputs = policy.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=output_scores,
            output_scores=output_scores,
        )
        if output_scores:
            generated = outputs.sequences[:, input_length:]
            return generated, outputs.scores
        return outputs[:, input_length:]

    def run_model(self, policy: _Policy, images):
        generated, scores = self.generate(policy, images, output_scores=True)
        unnorm_key = (
            images[0]["unnorm_key"]
            if images and isinstance(images[0], dict) else UNNORM_KEY
        )
        for step, score in enumerate(scores):
            self._trace(f"generation.step_{step:03d}.scores", score)
        self._trace("generation.tokens", generated)
        actions = []
        for row in generated:
            decoded = policy.processor.decode_actions(row.unsqueeze(0), unnorm_key=unnorm_key)[
                "actions"
            ]
            actions.append(torch.from_numpy(decoded.copy()))
        result = torch.stack(actions)
        self._trace("actions.unnormalized", result)
        return result

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = SpatialVLAAdapter()

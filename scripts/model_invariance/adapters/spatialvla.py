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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROMPT = "What action should the robot take to pick the cup?"
UNNORM_KEY = "bridge_orig/1.0.0"


@dataclass
class _Policy:
    model: torch.nn.Module
    processor: Any

    def eval(self):
        self.model.eval()
        return self


class SpatialVLAAdapter:
    name = "spatialvla-4b-224-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("SPATIALVLA_CHECKPOINT", ""))
        self.source = Path(os.environ.get("SPATIALVLA_SOURCE", ""))
        self.sample_dir = Path(os.environ.get("SPATIALVLA_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.trace_callback = None

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
        return _Policy(model=model, processor=processor).eval()

    def load_example_inputs(self, count: int):
        if count > 64:
            raise ValueError("the target plus public LIBERO fixture provide 64 examples")
        target = Image.open(self.checkpoint / "example.png").convert("RGB")
        examples = [target.copy()]
        for frame_number in range(1, count):
            with Image.open(self.sample_dir / "frames" / f"image_{frame_number:03d}.png") as image:
                examples.append(image.convert("RGB").copy())
        return examples

    @staticmethod
    def compose_batch(samples):
        return list(samples)

    def prepare_batch(self, policy: _Policy, images):
        inputs = policy.processor(
            images=images,
            text=[PROMPT] * len(images),
            return_tensors="pt",
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
        generated = self.generate(policy, images)
        self._trace("generation.tokens", generated)
        actions = []
        for row in generated:
            decoded = policy.processor.decode_actions(row.unsqueeze(0), unnorm_key=UNNORM_KEY)[
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

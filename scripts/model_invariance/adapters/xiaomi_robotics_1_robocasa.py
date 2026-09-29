"""Official Xiaomi-Robotics-1 RoboCasa native-PyTorch adapter.

The released model draws a batch-shaped flow-noise tensor inside ``forward``.
This adapter evaluates the same VLM, cached-KV DiT, and five-step Euler sampler
while making that stochastic input explicit per example.  ``run_official_model``
retains the unmodified released path for the B=1 equivalence check.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from PIL import Image


class XiaomiRobotics1RoboCasaAdapter:
    name = "xiaomi-robotics-1-robocasa-official-pytorch"
    robot_type = "robocasa_mg"
    instruction = "pick the mug from the counter and place it under the coffee machine dispenser"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("XR1_CHECKPOINT", ""))
        self.sample = Path(os.environ.get("XR1_SAMPLE", ""))
        self.device = torch.device("cuda")
        self.processor: Any | None = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        required = (
            self.checkpoint / "config.json",
            self.checkpoint / "model.safetensors.index.json",
            self.checkpoint / "model-00001-of-00003.safetensors",
            self.checkpoint / "model-00002-of-00003.safetensors",
            self.checkpoint / "model-00003-of-00003.safetensors",
            self.checkpoint / "processing_mibot.py",
            self.checkpoint / "modeling_mibot.py",
            self.sample,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing Xiaomi-Robotics-1 artifacts: {missing}")

    def load_model(self):
        self._validate_paths()
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        return model.to(self.device).eval()

    @staticmethod
    def _image(value: dict[str, Any]) -> Image.Image:
        # Xiaomi evaluates 256x256 simulator views.  The public fixture stores
        # lossless 128x128 views, so restore the evaluation resolution before
        # invoking the released processor.
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB").resize((256, 256))

    def _messages(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The following observations are captured from multiple views.\n# Base View\n",
                    },
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": images[2]},
                    {
                        "type": "text",
                        "text": (
                            "\nGenerate robot actions for the task:\n" f"{self.instruction} /no_cot"
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def load_example_inputs(self, count: int):
        if self.processor is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        table = pq.read_table(self.sample)
        if count > table.num_rows:
            raise ValueError(f"RoboCasa fixture contains only {table.num_rows} frames")

        examples = []
        for frame_index, row in enumerate(table.slice(0, count).to_pylist()):
            images = [self._image(row[key]) for key in ("image_left", "image_right", "wrist_image")]
            state = torch.zeros(1, 1, 60, dtype=torch.float32)
            # This public episode stores EEF pose rather than the seven joint
            # angles consumed by Xiaomi's evaluator.  Keep the real images and
            # use deterministic, plausible Panda joint-space states; this is a
            # numerical inference fixture, not a policy-quality evaluation.
            panda_home = torch.tensor(
                [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04],
                dtype=torch.float32,
            )
            joint_offset = (frame_index % 9 - 4) * 0.0025
            state[0, 0, :7] = panda_home[:7] + joint_offset
            state[0, 0, 7] = panda_home[7]
            data = self.processor.apply_chat_template(
                self._messages(images),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                state=state,
                robot_type=self.robot_type,
            )
            noise = torch.randn(
                10,
                60,
                dtype=torch.bfloat16,
                device=self.device,
                generator=torch.Generator(device=self.device).manual_seed(42 + frame_index),
            )
            example = {key: value.detach().cpu() for key, value in data.items()}
            example["noise"] = noise.cpu()
            example["seed"] = 42 + frame_index
            examples.append(example)
        return examples

    @staticmethod
    def compose_batch(samples):
        # Qwen packs vision patches/images across the request, while textual,
        # proprioceptive, and action tensors retain an ordinary batch axis.
        output = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key == "noise":
                output[key] = torch.stack(values)
            elif key == "seed":
                output[key] = values
            else:
                output[key] = torch.cat(values, dim=0)
        return output

    def _to_device(self, batch):
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
            if key not in {"noise", "seed"}
        }

    def run_official_model(self, model, batch):
        if len(batch["seed"]) != 1:
            raise ValueError("the released seeded path is checked only at batch size 1")
        output = model(**self._to_device(batch), seed=batch["seed"][0])
        decoded = self.processor.decode_action(output.actions, robot_type=self.robot_type)
        return {"actions": decoded[:, :, :7].cpu()}

    def run_model(self, model, batch):
        inputs = self._to_device(batch)
        noise = batch["noise"].to(self.device)
        state = inputs.pop("state")
        action_mask = inputs.pop("action_mask")

        vlm_outputs = model.vlm(**inputs, use_cache=True)
        self._trace("vlm.position_ids", vlm_outputs.position_ids)
        first_cache = vlm_outputs.past_key_values[0][0]
        self._trace("vlm.cache.layer0.key", first_cache)

        batch_size, action_length, _ = action_mask.shape
        state_length = state.shape[1]
        query_length = 1 + state_length + action_length
        position_ids = (
            torch.arange(query_length, device=self.device).view(1, 1, -1).repeat(3, batch_size, 1)
            + vlm_outputs.position_ids.max(dim=-1).values[..., None]
            + 1
        )
        position_embeds = model.rotary_emb(action_mask, position_ids)
        dit_mask = torch.tril(
            torch.ones(batch_size, query_length, query_length, device=self.device)
        )
        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, query_length, -1)
        attention_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
        state_embed = model.state_projector(state)

        self._trace("dit.state_embedding", state_embed)
        self._trace("dit.input_noise", noise)
        sample = noise.clone()
        num_steps = 5
        for step in range(num_steps):
            timestep = torch.full(
                (batch_size, 1, 1),
                step / num_steps,
                device=self.device,
                dtype=sample.dtype,
            )
            velocity = model.dit_forward(
                noisy_action=sample,
                t=timestep,
                action_mask=action_mask,
                state_embed=state_embed,
                position_embeds=position_embeds,
                past_key_values=vlm_outputs.past_key_values,
                attn_mask=attention_mask,
            )
            self._trace(f"dit.step{step}.velocity", velocity)
            sample = sample + velocity * (1.0 / num_steps)

        self._trace("dit.normalized_actions", sample)
        decoded = self.processor.decode_action(sample, robot_type=self.robot_type)
        return {"actions": decoded[:, :, :7].cpu()}

    @staticmethod
    def extract_robot_output(output):
        return output


adapter = XiaomiRobotics1RoboCasaAdapter()

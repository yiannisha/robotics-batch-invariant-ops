"""Official Xiaomi-Robotics-1 RoboCasa native-PyTorch adapter.

The released model draws a batch-shaped flow-noise tensor inside ``forward``.
This adapter evaluates the same VLM, cached-KV DiT, and five-step Euler sampler
while making that stochastic input explicit per example.  ``run_official_model``
retains the unmodified released path for the B=1 equivalence check.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


BASE_SEED = 20261015


class XiaomiRobotics1RoboCasaAdapter:
    name = "xiaomi-robotics-1-robocasa-official-pytorch"
    robot_type = "robocasa_mg"
    instruction = "pick the mug from the counter and place it under the coffee machine dispenser"
    causal_ablation_inapplicable_operations = {
        "explicit_sdpa": {
            "reason": (
                "The pinned eager policy trace executes explicit matrix attention and does "
                "not dispatch aten::scaled_dot_product_attention."
            ),
            "evidence_path": (
                "/workspace/robotics-batch-invariant-ops/results/cross_model/"
                "xiaomi_robotics_1/full/operator/aten_trace_B8.json"
            ),
        }
    }

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[3]
        self.checkpoint = Path(os.environ.get(
            "XR1_CHECKPOINT", "/workspace/assets/xiaomi_robotics_1/checkpoint"
        ))
        self.sample = Path(os.environ.get(
            "XR1_SAMPLE",
            "/workspace/assets/xiaomi_robotics_1/sample/data/chunk-000/episode_000000.parquet",
        ))
        self.replay_manifest_path = Path(os.environ.get(
            "XR1_REPLAY_MANIFEST",
            str(root / "results/cross_model/xiaomi_robotics_1/full/replay_manifest.json"),
        ))
        self.device = torch.device("cuda")
        self.processor: Any | None = None
        self.replay_manifest: dict[str, Any] | None = None
        self.replay_noise: np.ndarray | None = None
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

    def _messages(
        self, images: list[Image.Image], instruction: str | None = None
    ) -> list[dict[str, Any]]:
        instruction = instruction or self.instruction
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
                            "\nGenerate robot actions for the task:\n" f"{instruction} /no_cot"
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    @staticmethod
    def _path_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize((256, 256))

    def execution_configuration(self) -> dict[str, Any]:
        return {
            "camera_features": ["image_left", "image_right", "wrist_image"],
            "source_image_shape_hwc": [128, 128, 3],
            "policy_image_shape_hwc": [256, 256, 3],
            "real_state_dim": 25,
            "padded_state_dim": 60,
            "canonical_language_tokens": 260,
            "flow_steps": 5,
            "action_chunk_size": 10,
            "model_action_dim": 60,
            "robot_output_dim": 7,
            "robot_type": self.robot_type,
        }

    def _load_replay_metadata(self) -> None:
        if self.replay_manifest is not None:
            return
        if not self.replay_manifest_path.is_file():
            raise FileNotFoundError(self.replay_manifest_path)
        manifest = json.loads(self.replay_manifest_path.read_text())
        if (
            manifest.get("model_id") != "xiaomi_robotics_1"
            or len(manifest.get("observations", [])) != 1000
            or manifest.get("noise_shape") != [1000, 3, 10, 60]
        ):
            raise ValueError("Xiaomi replay manifest does not describe the frozen campaign")
        noise_path = Path(manifest["noise_file"])
        if not noise_path.is_absolute():
            noise_path = self.replay_manifest_path.parent / noise_path
        self.replay_manifest = manifest
        self.replay_noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.processor is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        self._load_replay_metadata()
        assert self.replay_manifest is not None and self.replay_noise is not None
        if not 0 <= replay_index < len(self.replay_manifest["observations"]):
            raise IndexError(replay_index)
        if noise_id not in (0, 1, 2):
            raise IndexError(noise_id)
        row = self.replay_manifest["observations"][replay_index]
        if row.get("replay_index") != replay_index:
            raise ValueError("Xiaomi replay indices are not one-to-one")
        fixture = Path(self.replay_manifest["fixture_root"])
        images = [self._path_image(fixture / row["images"][key]) for key in (
            "image_left", "image_right", "wrist_image"
        )]
        real_state = torch.tensor(row["state"], dtype=torch.float32)
        if tuple(real_state.shape) != (25,):
            raise ValueError("Xiaomi frozen real state is not 25-D")
        state = torch.zeros(1, 1, 60, dtype=torch.float32)
        state[0, 0, :25] = real_state
        data = self.processor.apply_chat_template(
            self._messages(images, row["task"]),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state,
            robot_type=self.robot_type,
        )
        example = {key: value.detach().cpu() for key, value in data.items()}
        example["noise"] = torch.from_numpy(
            np.array(self.replay_noise[replay_index, noise_id], copy=True)
        ).to(torch.bfloat16)
        example["seed"] = BASE_SEED + replay_index * 3 + noise_id
        return example

    def load_example_inputs(self, count: int):
        if self.processor is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if self.replay_manifest_path.is_file():
            return [self.load_replay_input(index, 0) for index in range(count)]
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

    def _compose(self, samples, canonical_language: bool):
        # Qwen packs vision patches/images across the request, while textual,
        # proprioceptive, and action tensors retain an ordinary batch axis.
        output = {}
        language_length = 260 if canonical_language else max(
            sample["input_ids"].shape[1] for sample in samples
        )
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key == "noise":
                output[key] = torch.stack(values)
            elif key == "seed":
                output[key] = values
            elif key in {"input_ids", "attention_mask"}:
                pad_value = (
                    int(self.processor.tokenizer.pad_token_id or 0)
                    if key == "input_ids" else 0
                )
                padded = [
                    F.pad(value, (0, language_length - value.shape[1]), value=pad_value)
                    for value in values
                ]
                output[key] = torch.cat(padded, dim=0)
            else:
                output[key] = torch.cat(values, dim=0)
        return output

    def compose_batch(self, samples):
        from batch_invariant_ops import is_batch_invariant_mode_enabled

        return self._compose(samples, canonical_language=is_batch_invariant_mode_enabled())

    def compose_fixed_batch(self, samples):
        return self._compose(samples, canonical_language=True)

    @staticmethod
    def fixed_batch_shape_metadata(batch):
        return {
            "language_padding": "canonical_right",
            "language_tokens": int(batch["input_ids"].shape[1]),
            "canonical_length": 260,
        }

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
            self._trace(f"flow.step_{step:02d}.prediction", velocity)
            self._trace(f"flow.step_{step:02d}.state", sample)

        self._trace("dit.normalized_actions", sample)
        decoded = self.processor.decode_action(sample, robot_type=self.robot_type)
        return {"actions": decoded[:, :, :7].cpu()}

    @staticmethod
    def extract_robot_output(output):
        return output


adapter = XiaomiRobotics1RoboCasaAdapter()

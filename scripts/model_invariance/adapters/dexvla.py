"""Official DexVLA PyTorch code with the closest public full checkpoint.

Required environment variables:

``DEXVLA_CHECKPOINT``
    Local ``checkpoint-60000`` snapshot from ``kuromivv/DexVLA``.
``DEXVLA_SAMPLE``
    One official ``lesjie/dexvla_example_data`` HDF5 episode.

DexVLA's authors publish official PyTorch inference code and pretrained
ScaleDP heads, but no complete end-to-end robot checkpoint.  The selected
community checkpoint contains all Qwen2-VLA and ScaleDP weights and ships an
official-code-compatible ``config2.json``.  The adapter preserves the official
greedy reasoning plus ten-step ScaleDP path while generalizing two upstream
B=1 constants and making the initial diffusion noise an explicit input.
"""

from __future__ import annotations

import json
import os
import pickle
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


@dataclass
class _Policy:
    model: torch.nn.Module
    processor: Any

    def eval(self):
        self.model.eval()
        return self


class DexVLAAdapter:
    name = "dexvla-community-full-checkpoint-official-pytorch"
    camera_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("DEXVLA_CHECKPOINT", ""))
        self.sample = Path(os.environ.get("DEXVLA_SAMPLE", ""))
        self.device = torch.device("cuda")
        self.stats: dict[str, np.ndarray] | None = None
        self.processor: Any | None = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        for filename in (
            "config2.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "dataset_stats.pkl",
            "preprocessor_config.json",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set DEXVLA_CHECKPOINT to a checkpoint containing {filename}"
                )
        if not self.sample.is_file():
            raise FileNotFoundError("set DEXVLA_SAMPLE to an official HDF5 episode")

    @staticmethod
    def _install_packed_vision_attention_bridge(model: torch.nn.Module) -> None:
        """Preserve Qwen's packed-image boundaries in invariant mode.

        Official Qwen2-VL SDPA concatenates every image along the sequence
        dimension and supplies a block-diagonal mask.  The generic dispatcher
        cannot infer that those blocks are independent batch items, so its
        reduction width still changes with the request batch.  DexVLA already
        passes the exact ``cu_seqlens`` metadata; use the library's varlen
        operator when invariant mode is active and retain the official method
        byte-for-byte otherwise.
        """
        from batch_invariant_ops import (
            is_batch_invariant_mode_enabled,
            varlen_scaled_dot_product_attention_batch_invariant,
        )
        from qwen2_vla.models.modeling_qwen2_vla import apply_rotary_pos_emb_vision

        for attention in (block.attn for block in model.visual.blocks):
            official_forward = attention.forward

            def forward(
                self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=None,
                *,
                _official_forward=official_forward,
            ):
                if not is_batch_invariant_mode_enabled():
                    return _official_forward(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        rotary_pos_emb=rotary_pos_emb,
                    )
                sequence_length = hidden_states.shape[0]
                query, key, value = (
                    self.qkv(hidden_states)
                    .reshape(sequence_length, 3, self.num_heads, -1)
                    .permute(1, 0, 2, 3)
                    .unbind(0)
                )
                query = apply_rotary_pos_emb_vision(query.unsqueeze(0), rotary_pos_emb).squeeze(0)
                key = apply_rotary_pos_emb_vision(key.unsqueeze(0), rotary_pos_emb).squeeze(0)
                attended = varlen_scaled_dot_product_attention_batch_invariant(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    cu_seqlens,
                    cu_seqlens,
                )
                return self.proj(attended.squeeze(0).reshape(sequence_length, -1))

            attention.forward = types.MethodType(forward, attention)

    def load_model(self):
        self._validate_paths()
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        import policy_heads  # noqa: F401 - registers the ScaleDP config/model
        from qwen2_vla.models.configuration_qwen2_vla import Qwen2VLAConfig
        from qwen2_vla.models.modeling_qwen2_vla import (
            Qwen2VLForConditionalGenerationForVLA,
        )
        from qwen2_vla.utils.processing_qwen2_vla import Qwen2VLProcessor

        config = Qwen2VLAConfig.from_dict(
            json.loads((self.checkpoint / "config2.json").read_text())
        )
        # The community checkpoint uses its older DiT_H alias; the official
        # repository calls the identical 32-layer/1280-wide shape ScaleDP_H.
        config.policy_head_size = "ScaleDP_H"
        model = Qwen2VLForConditionalGenerationForVLA.from_pretrained(
            self.checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        processor = Qwen2VLProcessor.from_pretrained(self.checkpoint, use_fast=False)
        self.processor = processor

        # The official README documents bfloat16 construction of the old DDIM
        # schedule as an inference bug. Reconstructing it after model loading
        # is the released float32 fix and prevents NaNs at the final step.
        model.policy_head.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
        self._install_packed_vision_attention_bridge(model)
        with (self.checkpoint / "dataset_stats.pkl").open("rb") as handle:
            self.stats = pickle.load(handle)
        return _Policy(model.eval(), processor)

    def load_example_inputs(self, count: int):
        if self.stats is None or self.processor is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        with h5py.File(self.sample, "r") as episode:
            frame_count = episode["observations/qpos"].shape[0]
            if count > frame_count:
                raise ValueError(f"fixture contains only {frame_count} frames")
            target_frame = int(os.environ.get("DEXVLA_TARGET_FRAME", "0"))
            if not 0 <= target_frame < frame_count:
                raise ValueError(f"DEXVLA_TARGET_FRAME must be below {frame_count}")
            frame_indices = [target_frame]
            frame_indices.extend(index for index in range(frame_count) if index != target_frame)
            task = episode["language_raw"][0].decode()
            examples = []
            for frame_index in frame_indices[:count]:
                images = [
                    torch.from_numpy(
                        np.array(
                            episode[f"observations/images/{camera}"][frame_index],
                            copy=True,
                        )
                    )
                    for camera in self.camera_names
                ]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None},
                            {"type": "image", "image": None},
                            {"type": "image", "image": None},
                            {"type": "text", "text": task},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processed = self.processor(
                    text=text,
                    images=images,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                state = np.array(episode["observations/qpos"][frame_index], copy=True)
                state = (state - self.stats["qpos_mean"]) / self.stats["qpos_std"]
                generator = torch.Generator(device="cuda").manual_seed(96_000 + frame_index)
                noise = torch.randn(
                    1,
                    50,
                    14,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                ).cpu()
                examples.append(
                    {
                        "input_ids": processed["input_ids"],
                        "attention_mask": processed["attention_mask"],
                        "pixel_values": processed["pixel_values"],
                        "image_grid_thw": processed["image_grid_thw"],
                        "states": torch.tensor(state).unsqueeze(0),
                        "noise": noise,
                    }
                )
        return examples

    @staticmethod
    def compose_batch(samples):
        return {key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]}

    def run_model(self, policy: _Policy, batch):
        model = policy.model
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        image_grid_thw = batch["image_grid_thw"].to(self.device)
        states = batch["states"].to(self.device)
        batch_size, prompt_length = input_ids.shape
        self._trace("input.pixel_values", pixel_values)
        self._trace("input.states", states)
        self._trace("input.noise", batch["noise"])

        generated = model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            is_eval=True,
            num_beams=1,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=256,
            eos_token_id=policy.processor.tokenizer.eos_token_id,
            pad_token_id=policy.processor.tokenizer.eos_token_id,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        sequences = generated.sequences
        self._trace("reasoning.tokens", sequences)
        hidden_states = torch.cat([step[-1] for step in generated.hidden_states], dim=1)
        self._trace("reasoning.hidden", hidden_states)

        # Generation continues finished rows with EOS until every companion is
        # done.  The official B=1 FiLM path never sees those extra hidden
        # states, so trim each row at its own first EOS before conditioning.
        # This also avoids requiring every batch member to reason for the same
        # number of tokens while keeping the learned ScaleDP call batched.
        eos_token_id = policy.processor.tokenizer.eos_token_id
        conditioning_items = []
        for batch_index in range(batch_size):
            generated_ids = sequences[batch_index, prompt_length:]
            eos_locations = torch.nonzero(generated_ids == eos_token_id, as_tuple=False)
            generated_length = (
                int(eos_locations[0].item()) + 1
                if eos_locations.numel()
                else generated_ids.shape[0]
            )
            sequence_end = prompt_length + generated_length
            # Generation hidden states contain the prompt prefill followed by
            # one state for every generated token except the final token.
            hidden_end = sequence_end - 1
            labels = torch.cat(
                [
                    torch.full(
                        (1, prompt_length),
                        -100,
                        device=self.device,
                        dtype=torch.long,
                    ),
                    torch.ones(
                        (1, generated_length),
                        device=self.device,
                        dtype=torch.long,
                    ),
                ],
                dim=1,
            )
            conditioning_items.append(
                model.film_forward(
                    labels=labels,
                    input_ids=sequences[batch_index : batch_index + 1, :sequence_end],
                    hidden_states=hidden_states[batch_index : batch_index + 1, :hidden_end],
                )
            )
        conditioning = torch.cat(conditioning_items, dim=0)
        self._trace("policy.conditioning", conditioning)

        head = model.policy_head
        scheduler = head.noise_scheduler
        scheduler.set_timesteps(head.num_inference_timesteps)
        actions = batch["noise"].to(self.device, dtype=conditioning.dtype)
        states = states.to(conditioning.dtype)
        for index, timestep in enumerate(scheduler.timesteps):
            noise_prediction = head.model_forward(
                actions,
                timestep,
                global_cond=conditioning,
                states=states,
            )
            self._trace(f"diffusion.step_{index:02d}.prediction", noise_prediction)
            actions = scheduler.step(
                model_output=noise_prediction,
                timestep=timestep,
                sample=actions,
            ).prev_sample
            self._trace(f"diffusion.step_{index:02d}.state", actions)
        self._trace("actions.normalized", actions)

        normalized = actions.detach().float().cpu().numpy()
        result = ((normalized + 1) / 2) * (
            self.stats["action_max"] - self.stats["action_min"]
        ) + self.stats["action_min"]
        output = torch.from_numpy(result)
        self._trace("actions.unnormalized", output)
        return output

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = DexVLAAdapter()

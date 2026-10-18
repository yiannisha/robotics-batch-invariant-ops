"""Official CogACT-Small PyTorch adapter using the released batch path.

Required environment variables:

``COGACT_CHECKPOINT``
    Local ``CogACT/CogACT-Small`` snapshot.
``COGACT_SOURCE``
    Official ``microsoft/CogACT`` checkout.
``COGACT_OPENVLA_SOURCE``
    The authors' ``arnoldland/openvla`` dependency with the repository's
    inference-only construction patch applied.
``COGACT_SAMPLE_DIR``
    Public LIBERO frame fixture used for unrelated real-image companions.
``COGACT_LLAMA_METADATA``
    Pinned local Llama-2 config/tokenizer mirror used only to construct the
    empty language backbone before loading every learned tensor from CogACT.

The target follows the official Azure inference example: its bundled
``scripts/aml/test_image.png``, prompt ``move sponge near apple``, RT-1 action
statistics, CFG 1.5, and ten deterministic DDIM steps.  Initial diffusion
noise is represented as an explicit per-sample input.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from batch_invariant_ops import is_batch_invariant_mode_enabled

PROMPT = "move sponge near apple"
UNNORM_KEY = "fractal20220817_data"
CFG_SCALE = 1.5
DDIM_STEPS = 10
CANONICAL_LANGUAGE_TOKENS = 27


class _CogACTBatch(dict):
    def __init__(self, values, *, fixed_language: bool):
        super().__init__(values)
        self.fixed_language = fixed_language


@dataclass
class _Policy:
    model: torch.nn.Module
    prismatic_vlm_class: type

    def eval(self):
        self.model.eval()
        return self


class CogACTSmallAdapter:
    name = "cogact-small-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("COGACT_CHECKPOINT", ""))
        self.source = Path(os.environ.get("COGACT_SOURCE", ""))
        self.openvla_source = Path(os.environ.get("COGACT_OPENVLA_SOURCE", ""))
        self.sample_dir = Path(os.environ.get("COGACT_SAMPLE_DIR", ""))
        self.llama_metadata = Path(os.environ.get("COGACT_LLAMA_METADATA", ""))
        self.replay_manifest_path = Path(os.environ.get(
            "COGACT_REPLAY_MANIFEST",
            str(Path(__file__).resolve().parents[3]
                / "results/cross_model/cogact_small/full/replay_manifest.json"),
        ))
        self.device = torch.device("cuda")
        self.trace_callback = None
        self.replay_manifest = None
        self.replay_noise = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> Path:
        checkpoint_file = self.checkpoint / "checkpoints" / "CogACT-Small.pt"
        for path in (
            checkpoint_file,
            self.checkpoint / "config.json",
            self.checkpoint / "dataset_statistics.json",
            self.source / "vla" / "cogactvla.py",
            self.source / "scripts" / "aml" / "test_image.png",
            self.openvla_source / "prismatic" / "models" / "vlms" / "prismatic.py",
            self.llama_metadata / "config.json",
            self.llama_metadata / "tokenizer.model",
            self.llama_metadata / "tokenizer_config.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(f"required CogACT artifact is missing: {path}")
        for frame_number in (1, 63):
            frame = self.sample_dir / "frames" / f"image_{frame_number:03d}.png"
            if not frame.is_file():
                raise FileNotFoundError(
                    "set COGACT_SAMPLE_DIR to the public LIBERO fixture containing " f"{frame.name}"
                )
        return checkpoint_file

    def load_model(self):
        checkpoint_file = self._validate_paths()
        # CogACT's own ``vla`` package must precede the OpenVLA dependency's
        # ``prismatic`` package when both source trees are used directly.
        for path in (str(self.openvla_source), str(self.source)):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)

        from prismatic.models.backbones.llm.llama2 import LLAMA2_MODELS
        from prismatic.models.vlms.prismatic import PrismaticVLM
        from transformers.modeling_utils import no_init_weights
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from vla import load_vla

        # Keep construction fully pinned and offline.  The CogACT checkpoint
        # supplies all learned Llama tensors; this directory contains only the
        # architecture and tokenizer metadata needed by AutoConfig/Tokenizer.
        LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"] = str(self.llama_metadata)

        # Construct the empty Llama directly in the deployment BF16 dtype.
        # Every learned tensor is replaced by the strict CogACT checkpoint;
        # this has the same rounding result as load-FP32-then-convert while
        # avoiding a transient ~14 GB allocation under the shared cgroup.
        had_local_descriptor = "_from_config" in LlamaForCausalLM.__dict__
        local_descriptor = LlamaForCausalLM.__dict__.get("_from_config")
        original_from_config = LlamaForCausalLM._from_config

        @classmethod
        def bf16_from_config(cls, config, **kwargs):
            kwargs["torch_dtype"] = torch.bfloat16
            with no_init_weights():
                return original_from_config(config, **kwargs)

        LlamaForCausalLM._from_config = bf16_from_config
        try:
            model = load_vla(
                checkpoint_file,
                load_for_training=False,
                action_model_type="DiT-S",
                future_action_window_size=15,
            )
        finally:
            if had_local_descriptor:
                LlamaForCausalLM._from_config = local_descriptor
            else:
                delattr(LlamaForCausalLM, "_from_config")
        # This is the authors' documented deployment configuration: the VLM
        # runs in BF16 while the small diffusion action head remains FP32.
        model.vlm = model.vlm.to(torch.bfloat16)
        model = model.to(self.device).eval()
        model.action_model.create_ddim(ddim_step=DDIM_STEPS)
        return _Policy(model=model, prismatic_vlm_class=PrismaticVLM).eval()

    def load_example_inputs(self, count: int):
        if count > 64:
            raise ValueError("the official target plus public fixture provide 64 examples")
        with Image.open(self.source / "scripts" / "aml" / "test_image.png") as handle:
            target = handle.convert("RGB").copy()
        images = [target]
        for frame_number in range(1, count):
            with Image.open(self.sample_dir / "frames" / f"image_{frame_number:03d}.png") as handle:
                images.append(handle.convert("RGB").copy())

        examples = []
        for index, image in enumerate(images):
            generator = torch.Generator(device="cuda").manual_seed(74_000 + index)
            noise = torch.randn(
                1,
                16,
                7,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ).cpu()
            examples.append({"image": image, "instruction": PROMPT, "noise": noise})
        return examples

    @staticmethod
    def _sha256_bytes(value: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.replay_manifest is None:
            self.replay_manifest = json.loads(self.replay_manifest_path.read_text())
            self.replay_noise = np.load(self.replay_manifest["noise_file"], mmap_mode="r")
            if self.replay_noise.shape != (1000, 3, 16, 7):
                raise ValueError("CogACT replay noise shape differs")
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows) or noise_id not in (0, 1, 2):
            raise IndexError((replay_index, noise_id))
        row = rows[replay_index]
        if row["replay_index"] != replay_index or len(row["request_ids"]) != 3:
            raise ValueError("CogACT replay identity differs")
        raw_noise = np.array(self.replay_noise[replay_index, noise_id], copy=True)
        if self._sha256_bytes(raw_noise) != row["noise"][noise_id]["sha256"]:
            raise ValueError("CogACT request-local noise changed")
        relative = row["images"]["observation"]
        image_path = Path(self.replay_manifest["fixture_root"]) / relative
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").copy()
        return {
            "image": image,
            "instruction": row["instruction"],
            "noise": torch.from_numpy(raw_noise).unsqueeze(0),
        }

    @staticmethod
    def compose_batch(samples):
        return _CogACTBatch({
            "images": [sample["image"] for sample in samples],
            "instructions": [sample["instruction"] for sample in samples],
            "noise": torch.cat([sample["noise"] for sample in samples], dim=0),
        }, fixed_language=is_batch_invariant_mode_enabled())

    @staticmethod
    def compose_fixed_batch(samples):
        return _CogACTBatch({
            "images": [sample["image"] for sample in samples],
            "instructions": [sample["instruction"] for sample in samples],
            "noise": torch.cat([sample["noise"] for sample in samples], dim=0),
        }, fixed_language=True)

    @staticmethod
    def execution_configuration() -> dict[str, Any]:
        return {
            "native_language_padding": "longest_in_batch_right",
            "invariant_language_padding": "canonical_right",
            "canonical_language_tokens": CANONICAL_LANGUAGE_TOKENS,
            "prompt_builder": "PurePromptBuilder",
            "cfg_scale": CFG_SCALE,
            "ddim_steps": DDIM_STEPS,
        }

    @staticmethod
    def fixed_batch_shape_metadata(batch):
        return {
            "language_padding": "canonical_right",
            "language_tokens": CANONICAL_LANGUAGE_TOKENS,
            "canonical_length": CANONICAL_LANGUAGE_TOKENS,
        }

    def _cognition_features(
        self, policy: _Policy, images, instructions, *, fixed_language: bool,
    ):
        model = policy.model
        image_transform = model.vlm.vision_backbone.image_transform
        tokenizer = model.vlm.llm_backbone.tokenizer
        input_ids = []
        pixel_values = []
        for image, instruction in zip(images, instructions, strict=True):
            prompt_builder = model.vlm.get_prompt_builder()
            prompt_builder.add_turn(
                role="human",
                message=f"What action should the robot take to {instruction.lower()}?",
            )
            token_ids = tokenizer(
                prompt_builder.get_prompt(), truncation=True, return_tensors="pt"
            ).input_ids.squeeze(0)
            token_ids = torch.cat([token_ids, torch.tensor([29_871, 2])])
            input_ids.append(token_ids)
            pixel_values.append(image_transform(image))

        maximum = max(value.shape[0] for value in input_ids)
        if maximum > CANONICAL_LANGUAGE_TOKENS:
            raise ValueError(
                f"CogACT real prompt has {maximum} tokens, exceeding canonical "
                f"length {CANONICAL_LANGUAGE_TOKENS}"
            )
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )[:, : tokenizer.model_max_length]
        if fixed_language and input_ids.shape[1] < CANONICAL_LANGUAGE_TOKENS:
            padded = torch.full(
                (input_ids.shape[0], CANONICAL_LANGUAGE_TOKENS),
                tokenizer.pad_token_id, dtype=input_ids.dtype,
            )
            padded[:, : input_ids.shape[1]] = input_ids
            input_ids = padded
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        if isinstance(pixel_values[0], torch.Tensor):
            pixels = torch.stack(pixel_values).to(self.device)
        else:
            pixels = {
                key: torch.stack([item[key] for item in pixel_values]).to(self.device)
                for key in pixel_values[0]
            }
        self._trace("input.input_ids", input_ids)
        self._trace("input.attention_mask", attention_mask)
        if isinstance(pixels, dict):
            for key, value in pixels.items():
                self._trace(f"input.pixel_values.{key}", value)
        else:
            self._trace("input.pixel_values", pixels)

        autocast_dtype = model.vlm.llm_backbone.half_precision_dtype
        with torch.autocast(
            "cuda",
            dtype=autocast_dtype,
            enabled=model.vlm.enable_mixed_precision_training,
        ):
            output = super(policy.prismatic_vlm_class, model.vlm).generate(
                input_ids=input_ids,
                pixel_values=pixels,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
            )

        vision = model.vlm.vision_backbone
        if vision.featurizer is not None:
            patch_count = vision.featurizer.patch_embed.num_patches
        else:
            patch_count = vision.siglip_featurizer.patch_embed.num_patches
        hidden = output.hidden_states[0][-1][:, patch_count:]
        cumulative = attention_mask.cumsum(dim=1)
        final_indices = (cumulative == cumulative.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded = final_indices.unsqueeze(-1).expand(-1, hidden.size(-1))
        cognition = hidden.gather(1, expanded.unsqueeze(1)).squeeze(1)
        self._trace("vlm.cognition", cognition)
        return cognition

    def run_model(self, policy: _Policy, batch):
        model = policy.model
        cognition = self._cognition_features(
            policy, batch["images"], batch["instructions"],
            fixed_language=batch.fixed_language,
        )
        action_dtype = next(model.action_model.net.parameters()).dtype
        cognition = cognition.unsqueeze(1).to(action_dtype)
        batch_size = cognition.shape[0]
        noise = batch["noise"].to(self.device, dtype=action_dtype)
        self._trace("diffusion.noise", noise)

        doubled_noise = torch.cat([noise, noise], dim=0)
        unconditional = model.action_model.net.z_embedder.uncondition.unsqueeze(0)
        unconditional = unconditional.expand(batch_size, 1, -1)
        conditioning = torch.cat([cognition, unconditional], dim=0)
        model_kwargs = {"z": conditioning, "cfg_scale": CFG_SCALE}
        sample_function = model.action_model.net.forward_with_cfg

        final = None
        progressive = model.action_model.ddim_diffusion.ddim_sample_loop_progressive(
            sample_function,
            doubled_noise.shape,
            noise=doubled_noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self.device,
            eta=0.0,
        )
        for step_index, sample in enumerate(progressive):
            final = sample["sample"]
            self._trace(f"diffusion.{step_index}.sample", final)
        assert final is not None
        normalized = final.chunk(2, dim=0)[0]
        self._trace("actions.normalized", normalized)

        normalized_numpy = normalized.cpu().numpy()
        statistics = model.get_action_stats(UNNORM_KEY)
        mask = statistics.get("mask", np.ones_like(statistics["q01"], dtype=bool))
        high = np.asarray(statistics["q99"])
        low = np.asarray(statistics["q01"])
        normalized_numpy = np.clip(normalized_numpy, -1, 1)
        normalized_numpy[:, :, 6] = np.where(normalized_numpy[:, :, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_numpy + 1) * (high - low) + low,
            normalized_numpy,
        )
        result = torch.from_numpy(actions.copy())
        self._trace("actions.unnormalized", result)
        return result

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = CogACTSmallAdapter()

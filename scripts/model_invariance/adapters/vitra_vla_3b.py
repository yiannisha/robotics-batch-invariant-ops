"""Official Microsoft VITRA-VLA-3B PyTorch adapter.

Required environment variables:

``VITRA_SOURCE``
    Official ``microsoft/VITRA`` checkout with the inference-only metadata
    construction patch in ``scripts/model_invariance/patches`` applied.
``VITRA_CHECKPOINT``
    Local native snapshot of ``VITRA-VLA/VITRA-VLA-3B``.
``VITRA_PALIGEMMA_METADATA``
    Local PaliGemma2 config/processor/tokenizer metadata.  The complete VITRA
    checkpoint supplies all learned weights; no base-model weights are loaded.

The released convenience method only accepts batch size one and samples noise
internally.  This adapter calls the same PyTorch preprocessing, VLM, DiT, and
DDIM code directly, with the initial DDIM noise made an explicit per-sample
input so batch comparisons hold stochastic state fixed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

INSTRUCTION = "Left hand: None. Right hand: Pick up the picture of Michael Jackson."


class VitraVLA3BAdapter:
    name = "vitra-vla-3b-official-pytorch"

    def __init__(self) -> None:
        self.source = Path(os.environ.get("VITRA_SOURCE", ""))
        self.checkpoint = Path(os.environ.get("VITRA_CHECKPOINT", ""))
        self.paligemma_metadata = Path(os.environ.get("VITRA_PALIGEMMA_METADATA", ""))
        self.device = torch.device("cuda")
        self.processor = None
        self.normalizer = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        required = (
            self.source / "vitra/models/vla_builder.py",
            self.source / "vitra/models/vla/vitra_paligemma.py",
            self.source / "examples/0001.jpg",
            self.source / "examples/0002.jpg",
            self.source / "examples/0003.png",
            self.checkpoint / "configs/config.json",
            self.checkpoint / "checkpoints/vitra-vla-3b.pt",
            self.checkpoint / "statistics/dataset_statistics.json",
            self.paligemma_metadata / "config.json",
            self.paligemma_metadata / "tokenizer.json",
            self.paligemma_metadata / "preprocessor_config.json",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing VITRA artifacts: {missing}")

    def _add_source_path(self) -> None:
        source = str(self.source)
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)

    def load_model(self):
        self._validate_paths()
        self._add_source_path()

        from vitra.models import load_model
        from vitra.utils.data_utils import load_normalizer

        config = json.loads((self.checkpoint / "configs/config.json").read_text())
        config["vlm"]["pretrained_model_name_or_path"] = str(self.paligemma_metadata)
        config["model_load_path"] = str(self.checkpoint / "checkpoints/vitra-vla-3b.pt")
        config["statistics_path"] = str(self.checkpoint / "statistics/dataset_statistics.json")
        model = load_model(config).to(self.device).eval()
        # This is the precision used by the released inference builder: the
        # config records BF16 training, but build_vla does not pass use_bf16.
        if model.use_bf16 or next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("expected the released VITRA FP32 inference path")
        self.processor = model.processor
        self.normalizer = load_normalizer(config)
        return model

    @staticmethod
    def _pad_state(normalized_state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        state_mask = torch.tensor([False, True], dtype=torch.bool)
        current_state = torch.tensor(normalized_state, dtype=torch.float32)
        expanded = state_mask.repeat_interleave(current_state.numel() // 2)
        current_state = current_state * expanded.to(current_state.dtype)
        padded = torch.zeros(212, dtype=torch.float32)
        padded_mask = torch.zeros(212, dtype=torch.bool)
        padded[:51] = current_state[:51]
        padded_mask[:51] = expanded[:51]
        padded[51:102] = current_state[61:112]
        padded_mask[51:102] = expanded[61:112]
        return padded, padded_mask

    @staticmethod
    def _action_mask() -> torch.Tensor:
        mask = torch.zeros(16, 192, dtype=torch.bool)
        mask[:, 51:102] = True
        return mask

    def load_example_inputs(self, count: int):
        if self.processor is None or self.normalizer is None:
            raise RuntimeError("load_model must be called before load_example_inputs")

        from vitra.utils.data_utils import resize_short_side_to_target

        image_paths = (
            self.source / "examples/0002.jpg",
            self.source / "examples/0001.jpg",
            self.source / "examples/0003.png",
        )
        normalized_state = self.normalizer.normalize_state(
            np.zeros_like(self.normalizer.state_mean, dtype=np.float32)
        )
        state, state_mask = self._pad_state(normalized_state)
        action_mask = self._action_mask()
        examples = []
        for index in range(count):
            image = Image.open(image_paths[index % len(image_paths)]).convert("RGB")
            image = resize_short_side_to_target(image, target=224)
            processed = self.processor(
                text="<image>" + INSTRUCTION,
                images=image,
                return_tensors="pt",
            )
            generator = torch.Generator(device=self.device).manual_seed(71_000 + index)
            noise = torch.randn(
                16,
                192,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            ).cpu()
            examples.append(
                {
                    "pixel_values": processed["pixel_values"][0],
                    "input_ids": processed["input_ids"][0],
                    "state": state.clone(),
                    "state_mask": state_mask.clone(),
                    "action_mask": action_mask.clone(),
                    "fov": torch.tensor([np.deg2rad(60.0), np.deg2rad(60.0)], dtype=torch.float32),
                    "noise": noise,
                }
            )
        return examples

    @staticmethod
    def compose_batch(samples):
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}

    def _sample_actions(self, model, action_features, batch):
        policy = model.act_model
        batch_size = action_features.shape[0]
        noise = batch["noise"]
        x_mask = batch["action_mask"]
        state = batch["state"][:, None]
        state_mask = batch["state_mask"][:, None]
        model_dtype = next(policy.net.parameters()).dtype
        action_features = action_features.to(model_dtype)

        duplicated_noise = torch.cat([noise, noise], dim=0)
        uncondition = policy.net.z_embedder.uncondition[None].expand(batch_size, 1, -1)
        conditioning = torch.cat([action_features, uncondition], dim=0)
        model_kwargs = {
            "z": conditioning,
            "x_mask": x_mask,
            "cfg_scale": 5.0,
            "state": state,
            "state_mask": state_mask,
        }
        if policy.ddim_diffusion is None:
            policy.create_ddim(ddim_step=10)

        sample = None
        progressive = policy.ddim_diffusion.ddim_sample_loop_progressive(
            policy.net.forward_with_cfg,
            duplicated_noise.shape,
            noise=duplicated_noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self.device,
            eta=0.0,
        )
        for step, output in enumerate(progressive):
            sample = output["sample"]
            self._trace(f"ddim.step_{step:02d}.state", sample[:batch_size])
        if sample is None:
            raise RuntimeError("VITRA DDIM sampler produced no steps")
        return sample[:batch_size]

    def _postprocess(self, actions, action_mask):
        actions = actions * action_mask.to(actions.dtype)
        actions = actions[..., :102]
        self._trace("actions.normalized", actions)

        # Match the released NumPy postprocessing exactly, including its FP64
        # promotion from JSON statistics, then return a tensor for the harness.
        unnormalized = self.normalizer.unnormalize_action(actions.detach().cpu().numpy())
        result = torch.from_numpy(unnormalized)
        self._trace("actions.unnormalized", result)
        return result

    def run_model(self, model, batch):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        attention_mask = torch.ones_like(batch["input_ids"], dtype=torch.bool)
        self._trace("preprocess.pixel_values", batch["pixel_values"])
        self._trace("preprocess.input_ids", batch["input_ids"])
        self._trace("preprocess.state", batch["state"])
        self._trace("input.noise", batch["noise"])

        hidden, hidden_mask = model.prepare_vlm_features(
            batch["pixel_values"],
            batch["input_ids"],
            attention_mask,
            batch["state_mask"],
            batch["state"],
            batch["fov"],
            use_cache=False,
        )
        self._trace("vlm.hidden", hidden)
        action_features = model.extract_cognition_token(hidden, hidden_mask)
        self._trace("vlm.cognition", action_features)
        actions = self._sample_actions(model, action_features, batch)
        return self._postprocess(actions, batch["action_mask"])

    def run_official_model(self, model, batch):
        """Run the released private inference path with controlled DDIM noise."""

        batch = {key: value.to(self.device) for key, value in batch.items()}
        attention_mask = torch.ones_like(batch["input_ids"], dtype=torch.bool)
        hidden, hidden_mask = model.prepare_vlm_features(
            batch["pixel_values"],
            batch["input_ids"],
            attention_mask,
            batch["state_mask"],
            batch["state"],
            batch["fov"],
            use_cache=False,
        )
        expected_shape = tuple(batch["noise"].shape)
        original_randn = torch.randn

        def controlled_randn(*size, **kwargs):
            shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], tuple) else size
            if tuple(shape) == expected_shape:
                return batch["noise"].clone()
            return original_randn(*size, **kwargs)

        torch.randn = controlled_randn
        try:
            actions, _ = model._forward_act_model(
                vlm_features=hidden,
                attention_mask=hidden_mask,
                action_masks=batch["action_mask"],
                current_state=batch["state"],
                current_state_mask=batch["state_mask"],
                mode="eval",
                repeated_diffusion_steps=1,
                cfg_scale=5.0,
                use_ddim=True,
                num_ddim_steps=10,
            )
        finally:
            torch.randn = original_randn
        return self._postprocess(actions, batch["action_mask"])

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = VitraVLA3BAdapter()

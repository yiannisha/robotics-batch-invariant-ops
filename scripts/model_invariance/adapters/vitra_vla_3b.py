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
import torch.nn.functional as F
from PIL import Image

INSTRUCTION = "Left hand: None. Right hand: Pick up the picture of Michael Jackson."
CANONICAL_LANGUAGE_TOKENS = 276
EXPECTED_XHAND_HUMAN_MAPPING = (
    (10, 8, 1),
    (11, 14, 1),
    (12, 17, 1),
    (13, 23, 1),
    (16, 26, 1),
    (17, 32, 1),
    (14, 35, 1),
    (15, 41, 1),
    (7, 43, -1),
    (6, 44, 1),
    (8, 46, -1),
    (9, 7, 1),
)


class VitraVLA3BAdapter:
    name = "vitra-vla-3b-official-pytorch"

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[3]
        self.source = Path(os.environ.get("VITRA_SOURCE", "/workspace/models/vitra"))
        self.checkpoint = Path(os.environ.get(
            "VITRA_CHECKPOINT", "/workspace/assets/vitra/checkpoint"
        ))
        self.paligemma_metadata = Path(os.environ.get(
            "VITRA_PALIGEMMA_METADATA", "/workspace/assets/vitra/paligemma_metadata"
        ))
        self.replay_manifest_path = Path(os.environ.get(
            "VITRA_REPLAY_MANIFEST",
            str(root / "results/cross_model/vitra_vla_3b/full/replay_manifest.json"),
        ))
        self.device = torch.device("cuda")
        self.processor = None
        self.normalizer = None
        self.replay_manifest: dict[str, Any] | None = None
        self.replay_noise: np.ndarray | None = None
        self.xhand_human_mapping = EXPECTED_XHAND_HUMAN_MAPPING
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
        from vitra.datasets.robot_dataset import XHAND_HUMAN_MAPPING
        from vitra.utils.data_utils import load_normalizer

        official_mapping = tuple(tuple(value) for value in XHAND_HUMAN_MAPPING)
        if official_mapping != EXPECTED_XHAND_HUMAN_MAPPING:
            raise RuntimeError("official VITRA XHand-to-human mapping changed")
        self.xhand_human_mapping = official_mapping

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

    def execution_configuration(self) -> dict[str, Any]:
        return {
            "source": "microsoft/VITRA-TeleData",
            "source_episodes": 50,
            "source_image_shape_hwc": [720, 1280, 3],
            "model_image_shape_hwc": [224, 398, 3],
            "real_xhand_state_dim": 18,
            "human_statistics_state_dim": 122,
            "padded_model_state_dim": 212,
            "canonical_language_tokens": CANONICAL_LANGUAGE_TOKENS,
            "ddim_steps": 10,
            "action_chunk_size": 16,
            "model_action_dim": 192,
            "robot_output_dim": 102,
            "precision": "float32",
        }

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

    def _load_replay_metadata(self) -> None:
        if self.replay_manifest is not None:
            return
        if not self.replay_manifest_path.is_file():
            raise FileNotFoundError(self.replay_manifest_path)
        manifest = json.loads(self.replay_manifest_path.read_text())
        if (
            manifest.get("model_id") != "vitra_vla_3b"
            or len(manifest.get("observations", [])) != 1000
            or manifest.get("noise_shape") != [1000, 3, 16, 192]
            or manifest.get("unique_physical_observations") != 1000
        ):
            raise ValueError("VITRA replay manifest does not describe the frozen campaign")
        noise_path = Path(manifest["noise_file"])
        if not noise_path.is_absolute():
            noise_path = self.replay_manifest_path.parent / noise_path
        self.replay_manifest = manifest
        self.replay_noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)

    def _raw_human_state(self, xhand_state: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            raise RuntimeError("VITRA normalizer is not loaded")
        if xhand_state.shape != (18,):
            raise ValueError("VITRA replay XHand state is not 18-D")
        if self.normalizer.state_mean.shape != (122,):
            raise ValueError("VITRA base checkpoint statistics are not 122-D")
        right_hand = np.zeros(51, dtype=np.float32)
        right_hand[:6] = xhand_state[:6]
        for source, destination, sign in self.xhand_human_mapping:
            right_hand[destination] = sign * xhand_state[source]
        # Human checkpoint statistics use [left 51, left beta 10,
        # right 51, right beta 10].  Shape parameters are unavailable and are
        # ignored by the model-side hand mask/padding operation.
        raw = np.zeros(122, dtype=np.float32)
        raw[61:112] = right_hand
        return raw

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.processor is None or self.normalizer is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        self._load_replay_metadata()
        assert self.replay_manifest is not None and self.replay_noise is not None
        if not 0 <= replay_index < len(self.replay_manifest["observations"]):
            raise IndexError(replay_index)
        if noise_id not in (0, 1, 2):
            raise IndexError(noise_id)
        row = self.replay_manifest["observations"][replay_index]
        if row.get("replay_index") != replay_index:
            raise ValueError("VITRA replay indices are not one-to-one")

        fixture_root = Path(self.replay_manifest["fixture_root"])
        image = Image.open(fixture_root / row["image"]).convert("RGB")
        processed = self.processor(
            text="<image>" + row["prompt"],
            images=image,
            return_tensors="pt",
        )
        xhand_state = np.asarray(row["xhand_state"], dtype=np.float32)
        raw_human_state = self._raw_human_state(xhand_state)
        normalized_state = self.normalizer.normalize_state(raw_human_state)
        state, state_mask = self._pad_state(normalized_state)
        intrinsics = np.asarray(row["camera_intrinsics"], dtype=np.float64)
        height, width = row["source_image_shape_hwc"][:2]
        fov = np.asarray([
            2 * np.arctan(width / (2 * intrinsics[0, 0])),
            2 * np.arctan(height / (2 * intrinsics[1, 1])),
        ], dtype=np.float32)
        return {
            "pixel_values": processed["pixel_values"][0],
            "input_ids": processed["input_ids"][0],
            "attention_mask": processed["attention_mask"][0].bool(),
            "state": state,
            "state_mask": state_mask,
            "action_mask": self._action_mask(),
            "fov": torch.from_numpy(fov),
            "noise": torch.from_numpy(
                np.array(self.replay_noise[replay_index, noise_id], copy=True)
            ),
        }

    def load_example_inputs(self, count: int):
        if self.processor is None or self.normalizer is None:
            raise RuntimeError("load_model must be called before load_example_inputs")

        if self.replay_manifest_path.is_file():
            return [self.load_replay_input(index, 0) for index in range(count)]

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
                    "attention_mask": processed["attention_mask"][0].bool(),
                    "state": state.clone(),
                    "state_mask": state_mask.clone(),
                    "action_mask": action_mask.clone(),
                    "fov": torch.tensor([np.deg2rad(60.0), np.deg2rad(60.0)], dtype=torch.float32),
                    "noise": noise,
                }
            )
        return examples

    def _compose(self, samples, canonical_language: bool):
        language_length = (
            CANONICAL_LANGUAGE_TOKENS
            if canonical_language
            else max(sample["input_ids"].shape[0] for sample in samples)
        )
        output = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key in {"input_ids", "attention_mask"}:
                if any(value.shape[0] > language_length for value in values):
                    raise ValueError("VITRA prompt exceeds canonical language length")
                pad_value = (
                    int(self.processor.tokenizer.pad_token_id or 0)
                    if key == "input_ids"
                    else 0
                )
                values = [
                    F.pad(value, (0, language_length - value.shape[0]), value=pad_value)
                    for value in values
                ]
            output[key] = torch.stack(values, dim=0)
        return output

    def compose_batch(self, samples):
        from batch_invariant_ops import is_batch_invariant_mode_enabled

        return self._compose(
            samples,
            canonical_language=is_batch_invariant_mode_enabled(),
        )

    def compose_fixed_batch(self, samples):
        return self._compose(samples, canonical_language=True)

    @staticmethod
    def fixed_batch_shape_metadata(batch):
        return {
            "language_padding": "canonical_right",
            "language_tokens": int(batch["input_ids"].shape[1]),
            "canonical_length": CANONICAL_LANGUAGE_TOKENS,
        }

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
        attention_mask = batch["attention_mask"]
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
        attention_mask = batch["attention_mask"]
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

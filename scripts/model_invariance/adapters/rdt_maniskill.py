"""Official RDT-1B PyTorch ManiSkill policy adapter.

Required environment variables:

``RDT_CHECKPOINT``
    Official ``robotics-diffusion-transformer/maniskill-model`` snapshot.
``RDT_SIGLIP``
    Local ``google/siglip-so400m-patch14-384`` snapshot.
``RDT_SOURCE``
    Official ``thu-ml/RoboticsDiffusionTransformer`` checkout.
``RDT_SAMPLE_DIR``
    Real public LIBERO frame fixture used for numerical image companions.

The official ManiSkill release includes a cached T5 embedding for PickCube-v1,
so this adapter consumes that learned embedding directly instead of loading a
second copy of T5.  Each request's six history/camera slots are encoded with
the official SigLIP tower before request batching.  Diffusion noise is an
explicit per-sample input; all five released DPM-Solver steps remain batched.
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
import yaml
from PIL import Image

from batch_invariant_ops import is_batch_invariant_mode_enabled

# Pinned official ``configs.state_vec.STATE_VEC_IDX_MAPPING`` values for
# ``right_arm_joint_[0-6]_pos`` and ``right_gripper_open``.
MANISKILL_INDICES = (0, 1, 2, 3, 4, 5, 6, 10)
ACTION_MIN = (
    -0.7472005486488342,
    -0.08631071448326111,
    -0.4995281398296356,
    -2.658363103866577,
    -0.5751323103904724,
    1.8290787935256958,
    -2.245187997817993,
    -1.0,
)
ACTION_MAX = (
    0.7654682397842407,
    1.4984270334243774,
    0.46786263585090637,
    -0.38181185722351074,
    0.5517147779464722,
    3.291581630706787,
    2.575840711593628,
    1.0,
)
STATE_MIN = (
    -0.7463043928146362,
    -0.0801204964518547,
    -0.4976441562175751,
    -2.657780647277832,
    -0.5742632150650024,
    1.8309762477874756,
    -2.2423808574676514,
    0.0,
)
STATE_MAX = (
    0.7645499110221863,
    1.4967026710510254,
    0.4650936424732208,
    -0.3866899907588959,
    0.5505855679512024,
    3.2900545597076416,
    2.5737812519073486,
    0.03999999910593033,
)
CANONICAL_LANGUAGE_TOKENS = 30


class _RDTBatch(dict):
    def __init__(self, values, *, fixed_language: bool):
        super().__init__(values)
        self.fixed_language = fixed_language


@dataclass
class _Policy:
    model: torch.nn.Module
    vision: torch.nn.Module
    processor: Any

    def eval(self):
        self.model.eval()
        self.vision.eval()
        return self


class RDTManiSkillAdapter:
    name = "rdt-1b-maniskill-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("RDT_CHECKPOINT", ""))
        self.siglip = Path(os.environ.get("RDT_SIGLIP", ""))
        self.source = Path(os.environ.get("RDT_SOURCE", ""))
        self.sample_dir = Path(os.environ.get("RDT_SAMPLE_DIR", ""))
        self.replay_manifest_path = Path(os.environ.get(
            "RDT_REPLAY_MANIFEST",
            str(Path(__file__).resolve().parents[3]
                / "results/cross_model/rdt_1b/full/replay_manifest.json"),
        ))
        self.device = torch.device("cuda")
        self.policy: _Policy | None = None
        self.trace_callback = None
        self.replay_manifest = None
        self.replay_noise = None
        self.language_embeddings: dict[str, torch.Tensor] = {}
        self.background_tokens: torch.Tensor | None = None
        self.image_token_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # SigLIP preprocessing is deliberately outside the batched RDT policy
        # request.  The pinned B=8 ATen trace proves that this policy boundary
        # has no convolution to repair in the patch-projection ablation stage.
        self.causal_ablation_inapplicable_operations = {
            "convolution": {
                "reason": (
                    "official SigLIP image encoding completes before request batching; "
                    "the traced RDT policy call consumes image tokens and exercises no convolution"
                ),
                "evidence_path": str(
                    Path(__file__).resolve().parents[3]
                    / "results/cross_model/rdt_1b/full/operator/aten_trace_B8.json"
                ),
            }
        }

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        required = (
            self.checkpoint / "rdt" / "mp_rank_00_model_states.pt",
            self.checkpoint / "lang_embeds" / "text_embed_PickCube-v1.pt",
            self.siglip / "config.json",
            self.siglip / "model.safetensors",
            self.siglip / "preprocessor_config.json",
            self.source / "configs" / "base.yaml",
            self.source / "models" / "rdt_runner.py",
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"required RDT artifact is missing: {path}")
        if self.replay_manifest_path.is_file():
            return
        for frame_number in (1, 64):
            path = self.sample_dir / "frames" / f"image_{frame_number:03d}.png"
            if not path.is_file():
                raise FileNotFoundError(
                    "set RDT_SAMPLE_DIR to the public LIBERO fixture containing " f"{path.name}"
                )

    def load_model(self):
        self._validate_paths()
        source_text = str(self.source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        from models.rdt_runner import RDTRunner

        config = yaml.safe_load((self.source / "configs" / "base.yaml").read_text())
        vision = SiglipVisionTower(str(self.siglip), args=None).to(
            self.device, dtype=torch.bfloat16
        )
        image_condition_length = (
            config["common"]["img_history_size"]
            * config["common"]["num_cameras"]
            * vision.num_patches
        )
        model = RDTRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
            img_cond_len=image_condition_length,
            img_pos_embed_config=[
                (
                    "image",
                    (
                        config["common"]["img_history_size"],
                        config["common"]["num_cameras"],
                        -vision.num_patches,
                    ),
                )
            ],
            lang_pos_embed_config=[("lang", -config["dataset"]["tokenizer_max_length"])],
            dtype=torch.bfloat16,
        )
        checkpoint = torch.load(
            self.checkpoint / "rdt" / "mp_rank_00_model_states.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["module"], strict=True)
        del checkpoint
        model = model.to(self.device, dtype=torch.bfloat16).eval()
        self.policy = _Policy(model=model, vision=vision.eval(), processor=vision.image_processor)
        self.language_embeddings = {
            task: torch.load(
                self.checkpoint / "lang_embeds" / f"text_embed_{task}.pt",
                map_location="cpu", weights_only=True,
            ).to(torch.bfloat16)
            for task in (
                "PegInsertionSide-v1", "PickCube-v1", "StackCube-v1",
                "PlugCharger-v1", "PushCube-v1",
            )
        }
        if max(value.shape[1] for value in self.language_embeddings.values()) != CANONICAL_LANGUAGE_TOKENS:
            raise ValueError("RDT language artifacts no longer have canonical width 30")
        return self.policy.eval()

    @staticmethod
    def _expand_to_square(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        edge = max(width, height)
        result = Image.new(image.mode, (edge, edge), color)
        result.paste(image, ((edge - width) // 2, (edge - height) // 2))
        return result

    def _encode_six_slots(self, policy: _Policy, image: Image.Image) -> torch.Tensor:
        color = tuple(int(value * 255) for value in policy.processor.image_mean)
        size = policy.processor.size
        background = Image.fromarray(
            np.ones((size["height"], size["width"], 3), dtype=np.uint8)
            * np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
        )
        # The released ManiSkill wrapper uses three cameras at two history
        # steps.  Its evaluation environment supplies only the current
        # exterior camera, leaving the other five slots as mean backgrounds.
        slots = [background, background, background, image, background, background]
        tensors = []
        for slot in slots:
            square = self._expand_to_square(slot, color)
            tensors.append(
                policy.processor.preprocess(square, return_tensors="pt")["pixel_values"][0]
            )
        pixels = torch.stack(tensors).to(self.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            tokens = policy.vision(pixels).detach()
        return tokens.reshape(1, -1, policy.vision.hidden_size).cpu()

    @staticmethod
    def _sha256_bytes(value: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    def _background_image(self, policy: _Policy) -> Image.Image:
        color = tuple(int(value * 255) for value in policy.processor.image_mean)
        size = policy.processor.size
        return Image.fromarray(
            np.ones((size["height"], size["width"], 3), dtype=np.uint8)
            * np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
        )

    def _encode_images(self, policy: _Policy, images: list[Image.Image]) -> torch.Tensor:
        color = tuple(int(value * 255) for value in policy.processor.image_mean)
        pixels = torch.stack([
            policy.processor.preprocess(
                self._expand_to_square(image, color), return_tensors="pt"
            )["pixel_values"][0]
            for image in images
        ]).to(self.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            return policy.vision(pixels).detach().cpu()

    def _replay_image_tokens(self, replay_index: int, row: dict) -> torch.Tensor:
        if self.policy is None:
            raise RuntimeError("load_model must precede replay loading")
        cached = self.image_token_cache.get(replay_index)
        if cached is None:
            root = Path(self.replay_manifest["fixture_root"])
            valid = row["history_valid"]
            if not isinstance(valid, list) or len(valid) != 2 or valid[1] is not True:
                raise ValueError("RDT replay history validity differs from evaluator contract")
            loaded = []
            for index in range(2):
                relative = row["images"][f"history_{index}"]
                if valid[index]:
                    with Image.open(root / relative) as handle:
                        loaded.append(handle.convert("RGB").copy())
                else:
                    loaded.append(self._background_image(self.policy))
            encoded = self._encode_images(self.policy, loaded)
            cached = (encoded[0], encoded[1])
            self.image_token_cache[replay_index] = cached
        if self.background_tokens is None:
            self.background_tokens = self._encode_images(
                self.policy, [self._background_image(self.policy)]
            )[0]
        previous, current = cached
        background = self.background_tokens
        slots = (previous, background, background, current, background, background)
        return torch.cat(slots, dim=0).unsqueeze(0)

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.policy is None:
            raise RuntimeError("load_model must precede replay loading")
        if self.replay_manifest is None:
            self.replay_manifest = json.loads(self.replay_manifest_path.read_text())
            noise_path = Path(self.replay_manifest["noise_file"])
            self.replay_noise = np.load(noise_path, mmap_mode="r")
            if self.replay_noise.shape != (1000, 3, 64, 128):
                raise ValueError("RDT replay noise shape differs")
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows) or noise_id not in (0, 1, 2):
            raise IndexError((replay_index, noise_id))
        row = rows[replay_index]
        if row["replay_index"] != replay_index or len(row["request_ids"]) != 3:
            raise ValueError("RDT replay identity differs")
        raw_noise = np.array(self.replay_noise[replay_index, noise_id], copy=True)
        if self._sha256_bytes(raw_noise) != row["noise"][noise_id]["sha256"]:
            raise ValueError("RDT request-local noise changed")
        joints = np.asarray(row["proprio"], dtype=np.float32)
        if joints.shape != (8,):
            raise ValueError("RDT replay proprio differs")
        minimum = np.asarray(STATE_MIN, dtype=np.float32)
        maximum = np.asarray(STATE_MAX, dtype=np.float32)
        normalized = (joints - minimum) / (maximum - minimum) * 2 - 1
        state = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
        state[0, 0, list(MANISKILL_INDICES)] = torch.from_numpy(normalized).to(torch.bfloat16)
        action_mask = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
        action_mask[0, 0, list(MANISKILL_INDICES)] = 1
        language = self.language_embeddings[row["task"]].clone()
        return {
            "language": language,
            "language_mask": torch.ones(language.shape[:2], dtype=torch.bool),
            "image_tokens": self._replay_image_tokens(replay_index, row),
            "state": state,
            "action_mask": action_mask,
            "control_frequency": torch.tensor([25], dtype=torch.long),
            "noise": torch.from_numpy(raw_noise).to(torch.bfloat16).unsqueeze(0),
        }

    def load_example_inputs(self, count: int):
        if self.policy is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the public image fixture provides 64 examples")
        language = torch.load(
            self.checkpoint / "lang_embeds" / "text_embed_PickCube-v1.pt",
            map_location="cpu",
            weights_only=True,
        ).to(torch.bfloat16)
        language_mask = torch.ones(language.shape[:2], dtype=torch.bool)
        action_mask = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
        action_mask[:, :, list(MANISKILL_INDICES)] = 1

        examples = []
        for frame_number in range(1, count + 1):
            with Image.open(self.sample_dir / "frames" / f"image_{frame_number:03d}.png") as handle:
                image = handle.convert("RGB").copy()
            image_tokens = self._encode_six_slots(self.policy, image)
            state = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
            if frame_number > 1:
                # Deterministic, valid normalized proprioceptive companions.
                value = ((frame_number - 1) % 17 - 8) / 10
                state[:, :, list(MANISKILL_INDICES)] = value
            generator = torch.Generator(device="cuda").manual_seed(73_000 + frame_number)
            noise = torch.randn(
                1,
                64,
                128,
                device=self.device,
                dtype=torch.bfloat16,
                generator=generator,
            ).cpu()
            examples.append(
                {
                    "language": language.clone(),
                    "language_mask": language_mask.clone(),
                    "image_tokens": image_tokens,
                    "state": state,
                    "action_mask": action_mask.clone(),
                    "control_frequency": torch.tensor([25], dtype=torch.long),
                    "noise": noise,
                }
            )
        # Vision preprocessing is complete and the 3.5 GB tower is not part of
        # the batched policy call.  Offload it so B=64 exercises the policy on
        # a single H100 without artificial memory pressure from an idle model.
        self.policy.vision.to("cpu")
        torch.cuda.empty_cache()
        return examples

    def _compose(self, samples, *, fixed_language: bool):
        canonical = fixed_language or is_batch_invariant_mode_enabled()
        language_tokens = (
            CANONICAL_LANGUAGE_TOKENS if canonical
            else max(sample["language"].shape[1] for sample in samples)
        )
        language = torch.zeros(
            len(samples), language_tokens, 4096, dtype=torch.bfloat16
        )
        language_mask = torch.zeros(len(samples), language_tokens, dtype=torch.bool)
        for index, sample in enumerate(samples):
            tokens = sample["language"].shape[1]
            language[index, :tokens] = sample["language"][0]
            language_mask[index, :tokens] = sample["language_mask"][0]
        values = {
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in samples[0] if key not in ("language", "language_mask")
        }
        values.update({"language": language, "language_mask": language_mask})
        return _RDTBatch(values, fixed_language=canonical)

    def compose_batch(self, samples):
        return self._compose(samples, fixed_language=False)

    def compose_fixed_batch(self, samples):
        return self._compose(samples, fixed_language=True)

    @staticmethod
    def execution_configuration() -> dict[str, Any]:
        return {
            "native_language_padding": "longest_in_batch_right",
            "invariant_language_padding": "canonical_right",
            "canonical_language_tokens": CANONICAL_LANGUAGE_TOKENS,
            "image_history": 2,
            "num_cameras": 3,
            "diffusion_steps": 5,
        }

    @staticmethod
    def fixed_batch_shape_metadata(batch):
        return {
            "language_padding": "canonical_right",
            "language_tokens": batch["language"].shape[1],
            "canonical_length": CANONICAL_LANGUAGE_TOKENS,
        }

    @staticmethod
    def reset_model(policy: _Policy) -> None:
        # set_timesteps also resets the multistep history, but clearing it here
        # makes repeated inference independent even if a future diffusers
        # version changes that implementation detail.
        policy.model.noise_scheduler_sample.model_outputs = [
            None
        ] * policy.model.noise_scheduler_sample.config.solver_order

    def run_model(self, policy: _Policy, batch):
        model = policy.model
        language = batch["language"].to(self.device)
        language_mask = batch["language_mask"].to(self.device)
        image_tokens = batch["image_tokens"].to(self.device)
        state = batch["state"].to(self.device)
        action_mask = batch["action_mask"].to(self.device)
        control_frequency = batch["control_frequency"].to(self.device)
        noisy_action = batch["noise"].to(self.device).clone()
        self._trace("input.language", language)
        self._trace("input.image_tokens", image_tokens)
        self._trace("input.state", state)
        self._trace("input.noise", noisy_action)

        state_with_mask = torch.cat([state, action_mask], dim=2)
        language_condition, image_condition, state_trajectory = model.adapt_conditions(
            language, image_tokens, state_with_mask
        )
        self._trace("condition.language", language_condition)
        self._trace("condition.image", image_condition)
        self._trace("condition.state", state_trajectory)

        expanded_mask = action_mask.expand(-1, model.pred_horizon, -1)
        model.noise_scheduler_sample.set_timesteps(model.num_inference_timesteps)
        for step_index, timestep in enumerate(model.noise_scheduler_sample.timesteps):
            action_trajectory = torch.cat([noisy_action, expanded_mask], dim=2)
            action_trajectory = model.state_adaptor(action_trajectory)
            state_action_trajectory = torch.cat([state_trajectory, action_trajectory], dim=1)
            prediction = model.model(
                state_action_trajectory,
                control_frequency,
                timestep.unsqueeze(-1).to(self.device),
                language_condition,
                image_condition,
                lang_mask=language_mask,
            )
            self._trace(f"diffusion.{step_index}.prediction", prediction)
            noisy_action = model.noise_scheduler_sample.step(
                prediction, timestep, noisy_action
            ).prev_sample.to(state_trajectory.dtype)
            self._trace(f"diffusion.{step_index}.sample", noisy_action)

        normalized = noisy_action * expanded_mask
        selected = normalized[:, :, list(MANISKILL_INDICES)]
        minimum = torch.tensor(ACTION_MIN, device=self.device, dtype=torch.float32)
        maximum = torch.tensor(ACTION_MAX, device=self.device, dtype=torch.float32)
        actions = (selected + 1) / 2 * (maximum - minimum) + minimum
        self._trace("actions.unnormalized", actions)
        return actions.cpu()

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = RDTManiSkillAdapter()

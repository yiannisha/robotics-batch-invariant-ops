"""LeRobot Pi0-FAST adapter using public PyTorch weights and a LIBERO sample.

Required environment variables:

``LEROBOT_PI0FAST_CHECKPOINT``
    Local ``lerobot/pi0fast-libero`` snapshot containing ``config.json``,
    ``model.safetensors``, and the normalization state safetensors.
``LEROBOT_PI0FAST_SAMPLE_DIR``
    Directory containing the first 64 frames from both camera videos under
    ``frames/`` and ``data/chunk-000/file-000.parquet`` from
    ``lerobot/libero`` episode zero.

The Google PaliGemma tokenizer repository is gated independently of the public
policy weights.  The adapter therefore defaults to a public byte-identical
tokenizer mirror while keeping LeRobot's model implementation and checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pyarrow.parquet as pq
from safetensors import safe_open
from scipy.fftpack import idct
from transformers import AutoProcessor, AutoTokenizer

TEXT_TOKENIZER = "leo009/paligemma-3b-pt-224"
TEXT_TOKENIZER_REVISION = "39996beb6fb17c5d16a50d3ef8f7a96ad9d03986"
ACTION_TOKENIZER_REVISION = "79ae83e3cbd8786dcb84b628569f8d076ca8151e"


class LeRobotPi0FastAdapter:
    name = "lerobot-pi0fast-libero-public-weights"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("LEROBOT_PI0FAST_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("LEROBOT_PI0FAST_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.config = None
        self.text_tokenizer = None
        self.action_tokenizer = None
        self.action_mean = None
        self.action_std = None

    def _validate_paths(self) -> None:
        for filename in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor_step_2_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set LEROBOT_PI0FAST_CHECKPOINT to a snapshot containing {filename}"
                )
        for filename in (
            "frames/image_001.png",
            "frames/image_064.png",
            "frames/image2_001.png",
            "frames/image2_064.png",
            "data/chunk-000/file-000.parquet",
        ):
            if not (self.sample_dir / filename).is_file():
                raise FileNotFoundError(
                    f"set LEROBOT_PI0FAST_SAMPLE_DIR to a directory containing {filename}"
                )

    def load_model(self):
        self._validate_paths()
        from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPytorch

        self.config = PI0FastConfig.from_pretrained(self.checkpoint)
        self.config.compile_model = False
        self.config.gradient_checkpointing = False
        self.config.use_kv_cache = True
        self.config.temperature = 0.0

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            os.environ.get("PI0FAST_TEXT_TOKENIZER", TEXT_TOKENIZER),
            revision=os.environ.get("PI0FAST_TEXT_TOKENIZER_REVISION", TEXT_TOKENIZER_REVISION),
            add_eos_token=True,
            add_bos_token=False,
        )
        self.action_tokenizer = AutoProcessor.from_pretrained(
            self.config.action_tokenizer_name,
            revision=os.environ.get("PI0FAST_ACTION_TOKENIZER_REVISION", ACTION_TOKENIZER_REVISION),
            trust_remote_code=True,
        )

        model = PI0FastPytorch(
            self.config,
            paligemma_tokenizer=self.text_tokenizer,
        )
        state = model.state_dict()
        loaded = set()
        with safe_open(
            self.checkpoint / "model.safetensors", framework="pt", device="cpu"
        ) as weights:
            for source_key in weights.keys():
                destination_key = source_key.removeprefix("model.")
                if destination_key in state:
                    state[destination_key].copy_(weights.get_tensor(source_key))
                    loaded.add(destination_key)

            # LeRobot's standard loader ties the language embedding to the LM
            # head by cloning this checkpoint tensor.
            embedding_key = (
                "paligemma_with_expert.paligemma.model.language_model." "embed_tokens.weight"
            )
            lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
            state[embedding_key].copy_(weights.get_tensor(f"model.{lm_head_key}"))
            loaded.add(embedding_key)

        missing = sorted(set(state) - loaded)
        if missing:
            raise RuntimeError(f"checkpoint did not populate {len(missing)} tensors: {missing[:5]}")

        with safe_open(
            self.checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            framework="pt",
            device="cpu",
        ) as statistics:
            self.action_mean = statistics.get_tensor("action.mean")
            self.action_std = statistics.get_tensor("action.std")

        model.eval().to(self.device)
        return model

    def _load_image(self, filename: str) -> torch.Tensor:
        image = Image.open(self.sample_dir / filename).convert("RGB").resize((224, 224))
        array = np.array(image, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).float().div(255)

    def _tokenize_prompt(self, task: str, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.text_tokenizer is not None
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        prompt = (
            f"Task: {task.strip().replace('_', ' ')}, State: "
            + " ".join(map(str, discretized))
            + ";\n"
        )
        encoded = self.text_tokenizer(
            prompt,
            padding="max_length",
            max_length=int(self.config.tokenizer_max_length),
            truncation=True,
            return_tensors="pt",
        )
        return encoded.input_ids[0], encoded.attention_mask[0].bool()

    def load_example_inputs(self, count: int):
        if self.config is None:
            raise RuntimeError("load_model must be called before load_example_inputs")

        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")
        target_frame = int(os.environ.get("PI0FAST_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("PI0FAST_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)
        frame_indices = frame_indices[:count]

        state_rows = pq.read_table(
            self.sample_dir / "data/chunk-000/file-000.parquet",
            columns=["observation.state"],
        )["observation.state"].to_pylist()
        with safe_open(
            self.checkpoint / "policy_preprocessor_step_2_normalizer_processor.safetensors",
            framework="pt",
            device="cpu",
        ) as statistics:
            state_mean = statistics.get_tensor("observation.state.mean")
            state_std = statistics.get_tensor("observation.state.std")

        task = "put the white mug on the left plate and put the yellow and white mug on the right plate"
        examples = []
        for frame_index in frame_indices:
            # The ffmpeg extraction pattern is one-indexed while the dataset
            # rows and PI0FAST_TARGET_FRAME are zero-indexed.
            image_number = frame_index + 1
            first = self._load_image(f"frames/image_{image_number:03d}.png")
            second = self._load_image(f"frames/image2_{image_number:03d}.png")
            raw_state = torch.tensor(state_rows[frame_index], dtype=torch.float32)
            state = (raw_state - state_mean) / (state_std + 1e-8)
            tokens, token_mask = self._tokenize_prompt(task, state.numpy())
            examples.append(
                {
                    "images": torch.stack((first, second, torch.zeros_like(first))),
                    "image_masks": torch.tensor((True, True, False)),
                    "tokens": tokens,
                    "token_mask": token_mask,
                }
            )
        return examples

    def _decode_actions(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        assert self.text_tokenizer is not None and self.action_tokenizer is not None
        prefix_ids = self.text_tokenizer.encode("Action: ", add_special_tokens=False)
        delimiter_id = self.text_tokenizer.convert_tokens_to_ids("|")
        decoded = []
        for sequence in generated_tokens.detach().cpu().tolist():
            if (
                self.config.validate_action_token_prefix
                and sequence[: len(prefix_ids)] != prefix_ids
            ):
                raise RuntimeError("generated sequence does not start with the Action prefix")
            sequence = sequence[len(prefix_ids) :]
            if delimiter_id in sequence:
                sequence = sequence[: sequence.index(delimiter_id)]
            action_ids = [
                self.text_tokenizer.vocab_size - 1 - self.config.fast_skip_tokens - token
                for token in sequence
            ]
            token_text = self.action_tokenizer.bpe_tokenizer.decode(action_ids)
            coefficients = np.array(list(map(ord, token_text))) + self.action_tokenizer.min_token
            expected = self.config.n_action_steps * self.config.output_features["action"].shape[0]
            coefficients = coefficients[:expected]
            if coefficients.size < expected:
                coefficients = np.pad(coefficients, (0, expected - coefficients.size))
            coefficients = coefficients.reshape(
                self.config.n_action_steps,
                self.config.output_features["action"].shape[0],
            )
            decoded.append(idct(coefficients / self.action_tokenizer.scale, axis=0, norm="ortho"))
        actions = torch.tensor(np.stack(decoded), dtype=torch.float32)
        return actions * self.action_std + self.action_mean

    def run_model(self, model, batch):
        images = batch["images"].to(self.device)
        image_masks = batch["image_masks"].to(self.device)
        tokens = batch["tokens"].to(self.device)
        token_mask = batch["token_mask"].to(self.device)
        image_list = [images[:, index] * 2 - 1 for index in range(images.shape[1])]
        mask_list = [image_masks[:, index] for index in range(image_masks.shape[1])]
        generated_tokens = model.sample_actions_fast_kv_cache(
            image_list,
            mask_list,
            tokens,
            token_mask,
            max_decoding_steps=int(os.environ.get("PI0FAST_MAX_DECODING_STEPS", "256")),
            temperature=0.0,
        )
        return {
            "action_tokens": generated_tokens,
            "actions": self._decode_actions(generated_tokens).to(self.device),
        }

    def extract_robot_output(self, output):
        # Tokens after the first end marker are padding/ignored by the FAST
        # decoder. A companion sample can keep the batch-wide decode loop alive
        # after x0 has finished, so those raw storage slots are not robot output.
        return output["actions"]


adapter = LeRobotPi0FastAdapter()

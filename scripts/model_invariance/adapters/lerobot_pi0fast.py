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

import json
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
        self.replay_manifest_path = Path(
            os.environ.get("PI0FAST_REPLAY_MANIFEST", str(self.sample_dir / "replay_manifest.json"))
        )
        self.device = torch.device("cuda")
        self.config = None
        self.text_tokenizer = None
        self.action_tokenizer = None
        self.action_mean = None
        self.action_std = None
        self.replay_manifest = None
        self.replay_state_stats = None
        self.trace_callback = None

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
        if self.replay_manifest_path.is_file():
            return
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
            os.environ.get("PI0FAST_ACTION_TOKENIZER", self.config.action_tokenizer_name),
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
        if self.replay_manifest_path.is_file():
            return [self.load_replay_input(index) for index in range(count)]

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

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        """Load one real LIBERO replay request for deterministic decoding."""
        if self.config is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        if noise_id != 0:
            raise ValueError("Pi0-FAST greedy action decoding has no flow-noise index")
        if self.replay_manifest is None:
            self.replay_manifest = json.loads(
                self.replay_manifest_path.read_text()
            )
            with safe_open(
                self.checkpoint / "policy_preprocessor_step_2_normalizer_processor.safetensors",
                framework="pt", device="cpu",
            ) as statistics:
                self.replay_state_stats = (
                    statistics.get_tensor("observation.state.mean"),
                    statistics.get_tensor("observation.state.std"),
                )
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows):
            raise IndexError(replay_index)
        row = rows[replay_index]
        first = self._load_image(row["images"]["image"])
        second = self._load_image(row["images"]["image2"])
        state_mean, state_std = self.replay_state_stats
        raw_state = torch.tensor(row["state"], dtype=torch.float32)
        state = (raw_state - state_mean) / (state_std + 1e-8)
        tokens, token_mask = self._tokenize_prompt(row["task"], state.numpy())
        return {
            "images": torch.stack((first, second, torch.zeros_like(first))),
            "image_masks": torch.tensor((True, True, False)),
            "tokens": tokens,
            "token_mask": token_mask,
        }

    def _decode_actions(
        self, generated_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.text_tokenizer is not None and self.action_tokenizer is not None
        prefix_ids = self.text_tokenizer.encode("Action: ", add_special_tokens=False)
        prefix_tokens = self.text_tokenizer.convert_ids_to_tokens(prefix_ids)
        decoded = []
        valid_prefix = []
        for sequence in generated_tokens.detach().cpu().tolist():
            # Match PI0FastPolicy.decode_actions_with_fast exactly.  The
            # official validity check is on the first two token *strings*
            # ("Action", ":"), while removal uses the normally generated
            # three-token sequence ("Action", ":", whitespace).  Some valid
            # greedy batches omit the whitespace token; rejecting those here
            # made the replay adapter stricter than the official evaluator.
            token_sequence = self.text_tokenizer.convert_ids_to_tokens(sequence)
            valid = (
                len(token_sequence) >= 2
                and token_sequence[0] == "Action"
                and token_sequence[1] == ":"
            )
            valid_prefix.append(valid)
            if self.config.validate_action_token_prefix and not valid:
                # The official evaluator asserts here and therefore produces
                # no robot action.  Retain that categorical behavioral failure
                # as a Boolean output and use NaNs only as a shape-preserving
                # unavailable-action sentinel for numerical aggregation.
                decoded.append(np.full(
                    (
                        self.config.n_action_steps,
                        self.config.output_features["action"].shape[0],
                    ),
                    np.nan,
                    dtype=np.float32,
                ))
                continue
            if "|" in token_sequence:
                token_sequence = token_sequence[: token_sequence.index("|")]
            index = 0
            while index <= len(token_sequence) - len(prefix_tokens):
                if token_sequence[index : index + len(prefix_tokens)] == prefix_tokens:
                    token_sequence = (
                        token_sequence[:index]
                        + token_sequence[index + len(prefix_tokens) :]
                    )
                else:
                    index += 1
            sequence = self.text_tokenizer.convert_tokens_to_ids(token_sequence)
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
        actions = actions * self.action_std + self.action_mean
        return actions, torch.tensor(valid_prefix, dtype=torch.bool)

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
        if self.trace_callback is not None:
            # Greedy FAST decoding returns fixed-width storage with zero-filled
            # columns after the batch-wide end marker.  Record every token step
            # that was actually generated; this preserves the complete
            # autoregressive chronology without pretending the trailing storage
            # slots were model iterations.
            nonzero_columns = torch.any(generated_tokens != 0, dim=0)
            generated_steps = int(torch.nonzero(nonzero_columns, as_tuple=False)[-1].item()) + 1
            for step in range(generated_steps):
                self.trace_callback(
                    f"flow.step_{step:03d}.prediction",
                    generated_tokens[:, step : step + 1],
                )
        actions, decode_valid = self._decode_actions(generated_tokens)
        return {
            "action_tokens": generated_tokens,
            "actions": actions.to(self.device),
            "official_decode_valid": decode_valid.to(self.device),
        }

    def extract_robot_output(self, output):
        # Tokens after the first end marker are padding/ignored by the FAST
        # decoder. A companion sample can keep the batch-wide decode loop alive
        # after x0 has finished, so those raw storage slots are not robot output.
        return {
            "actions": output["actions"],
            "official_decode_valid": output["official_decode_valid"],
        }


adapter = LeRobotPi0FastAdapter()

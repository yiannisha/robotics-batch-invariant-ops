"""Official BAAI UniVLA PyTorch LIBERO image-policy adapter.

Required environment variables:

``UNIVLA_CHECKPOINT``
    Local ``Yuqi1997/UniVLA`` ``UNIVLA_LIBERO_IMG_BS192_8K`` directory.
``UNIVLA_VISION_TOKENIZER``
    Local ``BAAI/Emu3-VisionTokenizer`` snapshot.
``UNIVLA_SOURCE``
    Official ``baaivision/UniVLA`` checkout.
``UNIVLA_SAMPLE_DIR``
    Real LIBERO fixture containing the two camera frame directories.

This follows the released PyTorch LIBERO wrapper: both camera images are
resized to 200 x 200, encoded by Emu3-VisionTokenizer, formatted in ``VLA``
mode, greedily decoded under the FAST action-vocabulary constraint, and
unnormalized with the released LIBERO constants.  Image VQ codes are produced
one observation at a time, matching online policy use and keeping preprocessing
independent of the inference request batch.
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

TASK = "put the white mug on the left plate and put the yellow and white mug " "on the right plate"
EOA_TOKEN_ID = 151845
GENERATION_COMPATIBILITY = (
    "transformers_post_4_50_explicit_generation_mixin_and_legacy_dynamic_cache_api"
)
ACTION_HIGH = np.array(
    [
        0.93712500009996,
        0.86775000009256,
        0.93712500009996,
        0.13175314309916836,
        0.19275000005139997,
        0.3353504997073735,
        0.9996000000999599,
    ]
)
ACTION_LOW = np.array(
    [
        -0.7046250000751599,
        -0.80100000008544,
        -0.9375000001,
        -0.11467779149968735,
        -0.16395000004372,
        -0.2240490058320433,
        -1.0000000001,
    ]
)


@dataclass
class _Policy:
    model: torch.nn.Module
    tokenizer: Any
    processor: Any
    action_tokenizer: Any
    generation_config: Any
    logits_processor: Any
    allowed_token_ids: torch.Tensor
    last_action_token_id: int

    def eval(self):
        self.model.eval()
        return self


class _UniVLABatch(dict):
    """Batch payload carrying whether language padding must use the frozen shape."""

    def __init__(self, values, *, fixed_language: bool):
        super().__init__(values)
        self.fixed_language = fixed_language


class UniVLAAdapter:
    name = "univla-baai-libero-image-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("UNIVLA_CHECKPOINT", ""))
        self.vision_checkpoint = Path(os.environ.get("UNIVLA_VISION_TOKENIZER", ""))
        self.source = Path(os.environ.get("UNIVLA_SOURCE", ""))
        self.sample_dir = Path(os.environ.get("UNIVLA_SAMPLE_DIR", ""))
        self.replay_manifest_path = Path(os.environ.get(
            "UNIVLA_REPLAY_MANIFEST",
            str(Path(__file__).resolve().parents[3]
                / "results/cross_model/univla_baai/full/replay_manifest.json"),
        ))
        self.device = torch.device("cuda")
        self.image_processor = None
        self.image_tokenizer = None
        self.replay_manifest: dict[str, Any] | None = None
        self.replay_cache: dict[int, dict[str, Any]] = {}
        self.canonical_prompt_length: int | None = None
        self.language_pad_token_id: int | None = None
        self.trace_callback = None
        direct_flash_evidence = str(
            self.source / "reference/Emu3/emu3/mllm/modeling_emu3.py"
        )
        self.causal_ablation_inapplicable_operations = {
            "explicit_sdpa": {
                "reason": (
                    "The released Emu3 policy calls FlashAttention directly instead of "
                    "torch.scaled_dot_product_attention; the adapter's full-invariant mode "
                    "bridges that direct call separately."
                ),
                "evidence_path": direct_flash_evidence,
            },
            "convolution": {
                "reason": (
                    "Image VQ encoding is request-independent preprocessing completed before "
                    "the traced policy batch; the exercised autoregressive policy contains no "
                    "convolution boundary."
                ),
                "evidence_path": str(Path(__file__).resolve()),
            },
        }

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        for filename in (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "emu3.tiktoken",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set UNIVLA_CHECKPOINT to the released checkpoint containing {filename}"
                )
        for filename in (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "modeling_emu3visionvq.py",
        ):
            if not (self.vision_checkpoint / filename).is_file():
                raise FileNotFoundError(
                    "set UNIVLA_VISION_TOKENIZER to the official snapshot " f"containing {filename}"
                )
        for filename in (
            "reference/Emu3/emu3/mllm/modeling_emu3.py",
            "pretrain/fast/processing_action_tokenizer.py",
            "pretrain/fast/tokenizer.json",
        ):
            if not (self.source / filename).is_file():
                raise FileNotFoundError(
                    f"set UNIVLA_SOURCE to the official checkout containing {filename}"
                )
        if self.replay_manifest_path.is_file():
            manifest = json.loads(self.replay_manifest_path.read_text())
            rows = manifest.get("observations", [])
            if (len(rows) != 1000
                    or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
                    or any(len(row.get("request_ids", [])) != 1 for row in rows)):
                raise ValueError("UniVLA replay must contain 1,000 deterministic LIBERO requests")
            root = Path(manifest.get("fixture_root", ""))
            for relative in rows[0].get("images", {}).values():
                if not (root / relative).is_file():
                    raise FileNotFoundError(root / relative)
        else:
            for filename in (
                "frames/image_001.png",
                "frames/image_064.png",
                "frames/image2_001.png",
                "frames/image2_064.png",
            ):
                if not (self.sample_dir / filename).is_file():
                    raise FileNotFoundError(
                        f"set UNIVLA_SAMPLE_DIR to the real LIBERO fixture containing {filename}"
                    )

    @staticmethod
    def _install_attention_bridge(model: torch.nn.Module) -> None:
        """Use the repository attention kernels only in invariant mode.

        Emu3 calls FlashAttention directly, outside PyTorch's SDPA dispatcher.
        Its decode kernel changes arithmetic when the request grows from 16 to
        32 rows.  Preserve the released FlashAttention path in native mode and
        bridge both its padded-prefill and cached-decode forms to the tested
        batch-invariant attention operators when invariant mode is enabled.
        """
        from batch_invariant_ops import (
            is_batch_invariant_mode_enabled,
            scaled_dot_product_attention_batch_invariant,
            varlen_scaled_dot_product_attention_batch_invariant,
        )
        from flash_attn.bert_padding import pad_input

        for attention in (layer.self_attn for layer in model.model.layers):
            official_forward = attention._flash_attention_forward

            def invariant_flash_attention(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                dropout=0.0,
                softmax_scale=None,
                *,
                _official_forward=official_forward,
            ):
                if not is_batch_invariant_mode_enabled():
                    return _official_forward(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        query_length,
                        dropout=dropout,
                        softmax_scale=softmax_scale,
                    )
                if dropout != 0.0:
                    raise ValueError("UniVLA invariant inference requires attention dropout 0")

                causal = self.is_causal
                if attention_mask is not None:
                    batch_size = query_states.shape[0]
                    (
                        query_unpadded,
                        key_unpadded,
                        value_unpadded,
                        indices_q,
                        cumulative_lengths,
                        _maximum_lengths,
                    ) = self._upad_input(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        query_length,
                    )
                    output = varlen_scaled_dot_product_attention_batch_invariant(
                        query_unpadded.unsqueeze(0),
                        key_unpadded.unsqueeze(0),
                        value_unpadded.unsqueeze(0),
                        cumulative_lengths[0],
                        cumulative_lengths[1],
                        is_causal=causal,
                        causal_type="BottomRight",
                        scale=softmax_scale,
                    ).squeeze(0)
                    return pad_input(output, indices_q, batch_size, query_length)

                # FlashAttention uses bottom-right causal alignment for cached
                # decoding.  Express that explicitly because ordinary SDPA's
                # rectangular causal mask is top-left aligned.
                key_length = key_states.shape[1]
                invariant_mask = None
                if causal and query_length != key_length:
                    query_indexes = torch.arange(query_length, device=query_states.device)[:, None]
                    key_indexes = torch.arange(key_length, device=query_states.device)[None, :]
                    invariant_mask = key_indexes <= query_indexes + (key_length - query_length)
                    causal = False
                return scaled_dot_product_attention_batch_invariant(
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2),
                    value_states.transpose(1, 2),
                    attn_mask=invariant_mask,
                    is_causal=causal,
                    scale=softmax_scale,
                    enable_gqa=query_states.shape[2] != key_states.shape[2],
                ).transpose(1, 2)

            attention._flash_attention_forward = types.MethodType(
                invariant_flash_attention, attention
            )

    def load_model(self):
        self._validate_paths()
        emu_source = str(self.source / "reference" / "Emu3")
        source_root = str(self.source)
        for path in (source_root, emu_source):
            if path not in sys.path:
                sys.path.insert(0, path)

        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoProcessor,
            GenerationConfig,
            LogitsProcessor,
        )
        from transformers.cache_utils import DynamicCache
        from transformers.generation import GenerationMixin

        from emu3.mllm import Emu3MoE, Emu3Processor, Emu3Tokenizer

        # The pinned Emu3 source predates removal of Cache.seen_tokens in
        # Transformers 4.57.  DynamicCache remains non-sliding here, so the
        # former running token count is exactly its current sequence length.
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(
                lambda cache: cache.get_seq_length()
            )
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda _cache: None
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = (
                lambda cache, _new_seq_length, layer_idx=0:
                cache.get_seq_length(layer_idx)
            )

        # UniVLA's pinned Emu3MoE implements prepare_inputs_for_generation but
        # predates Transformers 4.50, when PreTrainedModel stopped inheriting
        # GenerationMixin.  Preserve the released model implementation and add
        # only the generation API mixin that its official wrapper expects.
        if issubclass(Emu3MoE, GenerationMixin):
            generation_model_class = Emu3MoE
        else:
            class Emu3MoEWithGeneration(Emu3MoE, GenerationMixin):
                pass

            generation_model_class = Emu3MoEWithGeneration

        class ActionIDConstraintLogitsProcessor(LogitsProcessor):
            def __init__(self, allowed_token_ids):
                self.allowed_token_ids = allowed_token_ids

            def __call__(self, input_ids, scores):
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[:, self.allowed_token_ids] = True
                scores[~mask] = -float("inf")
                return scores

        model = (
            generation_model_class.from_pretrained(
                self.checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
        self._install_attention_bridge(model)
        tokenizer = Emu3Tokenizer.from_pretrained(
            self.checkpoint,
            model_max_length=model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            raise RuntimeError("released UniVLA tokenizer has no pad token")
        self.language_pad_token_id = int(tokenizer.pad_token_id)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vision_checkpoint, trust_remote_code=True
        )
        # Required by the released inference wrapper.  Without this assignment
        # the processor expands a 200px input to 512px and exceeds the policy's
        # 1,600-token context after encoding the two camera streams.
        self.image_processor.min_pixels = 80 * 80
        self.image_tokenizer = (
            AutoModel.from_pretrained(self.vision_checkpoint, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        processor = Emu3Processor(self.image_processor, self.image_tokenizer, tokenizer)
        if self.replay_manifest_path.is_file():
            self.replay_manifest = json.loads(self.replay_manifest_path.read_text())
            example = self._processed_replay_observation(0)
            tasks = sorted({row["task"] for row in self.replay_manifest["observations"]})
            lengths = []
            for task in tasks:
                inputs = processor.video_process(
                    text=[task],
                    video_tokens=example["video_tokens"].unsqueeze(1),
                    gripper_tokens=example["gripper_tokens"].unsqueeze(1),
                    context_frames=1,
                    frames=1,
                    mode="VLA",
                    padding=False,
                    return_tensors="pt",
                )
                lengths.append(int(inputs.input_ids.shape[1]))
            self.canonical_prompt_length = max(lengths)
            if self.canonical_prompt_length + 80 > model.config.max_position_embeddings:
                raise ValueError(
                    "frozen UniVLA prompt plus 80 action tokens exceeds the model context"
                )
        action_tokenizer = AutoProcessor.from_pretrained(
            self.source / "pretrain" / "fast", trust_remote_code=True
        )
        last_action_token_id = tokenizer.pad_token_id - 1
        allowed_token_ids = list(
            range(
                last_action_token_id - action_tokenizer.vocab_size,
                last_action_token_id + 1,
            )
        ) + [EOA_TOKEN_ID]
        generation_config = GenerationConfig(
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=EOA_TOKEN_ID,
            do_sample=False,
        )
        return _Policy(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            action_tokenizer=action_tokenizer,
            generation_config=generation_config,
            logits_processor=ActionIDConstraintLogitsProcessor(allowed_token_ids),
            allowed_token_ids=torch.tensor(allowed_token_ids, device=self.device),
            last_action_token_id=last_action_token_id,
        ).eval()

    def _encode_path(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((200, 200))
        pixels = self.image_processor(image, return_tensors="pt")["pixel_values"].to(self.device)
        return self.image_tokenizer.encode(pixels).cpu()

    @staticmethod
    def _runtime_image_to_pil(value: Any) -> Image.Image:
        """Recover the exact uint8 camera frame from LeRobot's env pipeline."""

        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to("cpu")
            if tensor.ndim != 3:
                raise ValueError(f"UniVLA runtime image must be rank 3, got {tensor.shape}")
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            if tensor.shape[-1] != 3:
                raise ValueError(f"UniVLA runtime image must have three channels: {tensor.shape}")
            if tensor.dtype.is_floating_point:
                tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
            else:
                tensor = tensor.to(torch.uint8)
            array = tensor.contiguous().numpy()
        else:
            array = np.asarray(value)
            if array.ndim != 3:
                raise ValueError(f"UniVLA runtime image must be rank 3, got {array.shape}")
            if array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.moveaxis(array, 0, -1)
            if array.shape[-1] != 3:
                raise ValueError(f"UniVLA runtime image must have three channels: {array.shape}")
            if np.issubdtype(array.dtype, np.floating):
                array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8, copy=False)
        return Image.fromarray(array, mode="RGB")

    def _encode_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((200, 200))
        pixels = self.image_processor(image, return_tensors="pt")["pixel_values"].to(
            self.device
        )
        return self.image_tokenizer.encode(pixels).cpu()

    def prepare_runtime_sample(
        self, *, image: Any, wrist_image: Any, task: str
    ) -> dict[str, Any]:
        """Build one released UniVLA request from a live LIBERO observation."""

        if self.image_processor is None or self.image_tokenizer is None:
            raise RuntimeError("load_model must be called before prepare_runtime_sample")
        with torch.inference_mode():
            return {
                "video_tokens": self._encode_pil(self._runtime_image_to_pil(image)),
                "gripper_tokens": self._encode_pil(
                    self._runtime_image_to_pil(wrist_image)
                ),
                "task": str(task),
            }

    def _encode_image(self, camera: str, frame_number: int) -> torch.Tensor:
        return self._encode_path(
            self.sample_dir / "frames" / f"{camera}_{frame_number:03d}.png"
        )

    def _processed_replay_observation(self, replay_index: int) -> dict[str, Any]:
        cached = self.replay_cache.get(replay_index)
        if cached is not None:
            return cached
        if self.replay_manifest is None:
            self.replay_manifest = json.loads(self.replay_manifest_path.read_text())
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows):
            raise IndexError(replay_index)
        row = rows[replay_index]
        root = Path(self.replay_manifest["fixture_root"])
        sample = {
            "video_tokens": self._encode_path(root / row["images"]["image"]),
            "gripper_tokens": self._encode_path(root / row["images"]["image2"]),
            "task": row["task"],
        }
        self.replay_cache[replay_index] = sample
        return sample

    def load_example_inputs(self, count: int):
        if self.image_processor is None or self.image_tokenizer is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if self.replay_manifest_path.is_file():
            return [self.load_replay_input(index) for index in range(count)]
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")
        target_frame = int(os.environ.get("UNIVLA_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("UNIVLA_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)

        examples = []
        self.image_tokenizer.to(self.device)
        with torch.inference_mode():
            for frame_index in frame_indices[:count]:
                frame_number = frame_index + 1
                examples.append(
                    {
                        "video_tokens": self._encode_image("image", frame_number),
                        "gripper_tokens": self._encode_image("image2", frame_number),
                        "task": TASK,
                    }
                )
        # VQ encoding is preprocessing, not part of the policy request batch.
        # Offload it after the fixed fixture is encoded to leave more memory for
        # large inference batches.
        self.image_tokenizer.cpu()
        torch.cuda.empty_cache()
        return examples

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.image_processor is None or self.image_tokenizer is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        if noise_id != 0:
            raise IndexError("UniVLA greedy action decoding has no request-local noise")
        return self._processed_replay_observation(replay_index)

    @staticmethod
    def compose_batch(samples, *, fixed_language: bool = False):
        return _UniVLABatch(
            {
                "video_tokens": torch.cat([sample["video_tokens"] for sample in samples], dim=0),
                "gripper_tokens": torch.cat(
                    [sample["gripper_tokens"] for sample in samples], dim=0
                ),
                "tasks": [sample.get("task", TASK) for sample in samples],
            },
            fixed_language=fixed_language,
        )

    def compose_fixed_batch(self, samples):
        return self.compose_batch(samples, fixed_language=True)

    def fixed_batch_shape_metadata(self, batch):
        return {
            "language_padding": "canonical_right",
            "language_tokens": self.canonical_prompt_length,
            "canonical_length": self.canonical_prompt_length,
        }

    def execution_configuration(self):
        if self.canonical_prompt_length is None:
            raise RuntimeError("load_model must establish the frozen UniVLA prompt length")
        return {
            "native_language_padding": "longest_in_batch_right",
            "invariant_language_padding": "canonical_right",
            "language_max_length": self.canonical_prompt_length,
            "language_pad_token_id": self.language_pad_token_id,
            "maximum_generated_action_tokens": 80,
            "decoding": "greedy_constrained_fast",
            "camera_resize_hw": [200, 200],
        }

    def prepare_batch(self, policy: _Policy, batch):
        batch_size = batch["video_tokens"].shape[0]
        video_tokens = batch["video_tokens"].unsqueeze(1)
        gripper_tokens = batch["gripper_tokens"].unsqueeze(1)
        self._trace("input.video_tokens", video_tokens)
        self._trace("input.gripper_tokens", gripper_tokens)

        from batch_invariant_ops import is_batch_invariant_mode_enabled

        fixed = batch.fixed_language or is_batch_invariant_mode_enabled()
        padding = "max_length" if fixed else "longest"
        extra = {"max_length": self.canonical_prompt_length} if fixed else {}
        inputs = policy.processor.video_process(
            text=batch["tasks"],
            video_tokens=video_tokens,
            gripper_tokens=gripper_tokens,
            context_frames=1,
            frames=1,
            mode="VLA",
            padding=padding,
            return_tensors="pt",
            **extra,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        self._trace("input.input_ids", input_ids)
        return input_ids, attention_mask

    def run_model(self, policy: _Policy, batch):
        batch_size = batch["video_tokens"].shape[0]
        input_ids, attention_mask = self.prepare_batch(policy, batch)

        generation = policy.model.generate(
            input_ids,
            policy.generation_config,
            max_new_tokens=80,
            logits_processor=[policy.logits_processor],
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
        )
        for step, scores in enumerate(generation.scores):
            self._trace(
                f"flow.step_{step:03d}.scores",
                scores.index_select(1, policy.allowed_token_ids),
            )
        generated = generation.sequences[:, input_ids.shape[1] :]
        self._trace("generation.tokens", generated)

        actions = []
        for batch_index in range(batch_size):
            row = generated[batch_index]
            eoa_locations = torch.nonzero(row == EOA_TOKEN_ID, as_tuple=False)
            if eoa_locations.numel():
                row = row[: int(eoa_locations[0].item())]
            processed = policy.last_action_token_id - row
            decoded = policy.action_tokenizer.decode(
                processed.unsqueeze(0).cpu(), time_horizon=10, action_dim=7
            )[0]
            action = 0.5 * (decoded + 1) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
            action[..., -1] = np.where(action[..., -1] > 0, 1, -1)
            actions.append(torch.from_numpy(action.copy()))

        result = torch.stack(actions)
        self._trace("actions.unnormalized", result)
        return result

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = UniVLAAdapter()

"""MolmoAct2 adapter using the official PyTorch LIBERO checkpoint.

Required environment variables:

``MOLMOACT2_CHECKPOINT``
    Local ``allenai/MolmoAct2-LIBERO`` snapshot.
``MOLMOACT2_SAMPLE_DIR``
    LIBERO fixture with the two camera frame directories and episode parquet
    used by the other LIBERO adapters.

The released model's native attention-bias builder allocates a leading batch
dimension of one and updates it in place with the per-example image mask.  It
therefore raises for every B > 1.  This adapter installs the minimal equivalent
implementation with that mask expanded to B before the in-place update.  The
B=1 attention bias is bitwise identical to the released implementation.

The public continuous-action API accepts a CUDA generator rather than an
explicit trajectory.  For the checkpoint's 10 x 32 trajectory, freshly seeded
CUDA generators produce an identical first-sample trajectory for all tested
batch sizes.  A fresh generator is consequently constructed for every call.
"""

from __future__ import annotations

import importlib
import hashlib
import io
import json
import os
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

TASK = "put the white mug on the left plate and put the yellow and white mug " "on the right plate"
CHECKPOINT_REPOSITORY = "allenai/MolmoAct2-LIBERO"
CHECKPOINT_REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
CHECKPOINT_SHARDS = (
    ("model-00001-of-00005.safetensors", 4_919_324_120, "5aa56113a3c1c95f0c03b90bada5b7ac2babf59c752a3d6201b8362a530c6809"),
    ("model-00002-of-00005.safetensors", 4_844_690_992, "1fd95aec9ad144d3f4eac42db88a000321e016beeae606ecfcb84fcbdb13ecf4"),
    ("model-00003-of-00005.safetensors", 4_844_691_024, "e6c455434915e74a8d750bedd0fc4c23b5a8ce37954aaff9c3e3f22085a26890"),
    ("model-00004-of-00005.safetensors", 4_877_619_536, "4e5bcf5bad81573153d6deaa3d25c3e53dc697658488b1cdbab4c8a3dffba1b6"),
    ("model-00005-of-00005.safetensors", 2_282_630_240, "76140ad00c225f95b80eb6b6a5fea2f985e4fdde3356a3c67c93d87d083d4ee2"),
)


class MolmoAct2Adapter:
    name = "molmoact2-libero-official-pytorch-public-weights"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("MOLMOACT2_CHECKPOINT", ""))
        self.sample_dir = Path(os.environ.get("MOLMOACT2_SAMPLE_DIR", ""))
        self.replay_manifest_path = Path(os.environ.get(
            "MOLMOACT2_REPLAY_MANIFEST",
            "/workspace/robotics-batch-invariant-ops/results/cross_model/molmoact2/full/replay_manifest.json",
        ))
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.processor = None
        self.model = None
        self.modeling_module = None
        self.stats = None
        self.metadata = None
        self.replay_manifest = None
        self.replay_noise = None
        self.stream_checkpoint = os.environ.get("MOLMOACT2_STREAM_CHECKPOINT", "0") == "1"
        self.checkpoint_receipt = Path(os.environ.get(
            "MOLMOACT2_CHECKPOINT_RECEIPT",
            "/workspace/robotics-batch-invariant-ops/results/cross_model/molmoact2/full/checkpoint_stream_receipt.json",
        ))
        self.canonical_prompt_padding = (
            os.environ.get("MOLMOACT2_DISABLE_CANONICAL_PROMPT_PADDING") != "1"
        )
        self.canonical_prompt_length = 497
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        required = [(
            self.checkpoint,
            (
                "config.json", "model.safetensors.index.json", "norm_stats.json",
                "modeling_molmoact2.py",
            ),
        )]
        if self.replay_manifest_path.is_file():
            manifest = json.loads(self.replay_manifest_path.read_text())
            rows = manifest.get("observations") or []
            if len(rows) != 1000:
                raise ValueError("MolmoAct2 replay manifest cardinality differs")
            fixture = Path(manifest["fixture_root"])
            required.append((fixture, tuple(
                path for row in rows[:1] for path in row["images"].values()
            )))
        else:
            required.append((
                self.sample_dir,
                (
                    "frames/image_001.png", "frames/image_064.png",
                    "frames/image2_001.png", "frames/image2_064.png",
                    "data/chunk-000/file-000.parquet",
                ),
            ))
        for root, filenames in required:
            for filename in filenames:
                if not (root / filename).is_file():
                    raise FileNotFoundError(f"missing required fixture: {root / filename}")

    @staticmethod
    def _batched_native_attention_bias(
        core: Any,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Released attention-bias logic with its B>1 broadcast bug fixed."""

        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.to(device=inputs_embeds.device)

        batch_size, seq_len = inputs_embeds.shape[:2]
        # Continuous action inference calls this only for the initial VLM
        # prefill, without an existing cache.
        if past_key_values is not None:
            return core.__class__._build_native_attention_bias(
                core,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
            )

        device = inputs_embeds.device
        if attention_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        else:
            valid_mask = attention_mask.to(device=device, dtype=torch.bool)

        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))[None, None]
            .expand(batch_size, -1, -1, -1)
            .clone()
        )
        if token_type_ids is not None:
            image_mask = token_type_ids.to(device=device, dtype=torch.bool)
            can_attend_back = image_mask[:, :, None] & image_mask[:, None, :]
            causal_mask |= can_attend_back[:, None]

        allowed = valid_mask[:, None, None, :] & causal_mask
        return torch.where(
            allowed,
            torch.zeros((), device=device, dtype=inputs_embeds.dtype),
            torch.full(
                (),
                torch.finfo(inputs_embeds.dtype).min,
                device=device,
                dtype=inputs_embeds.dtype,
            ),
        )

    def load_model(self):
        self._validate_paths()
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )
        if self.stream_checkpoint:
            config = AutoConfig.from_pretrained(self.checkpoint, trust_remote_code=True)
            with torch.device("meta"):
                self.model = AutoModelForImageTextToText.from_config(
                    config, trust_remote_code=True, dtype=self.dtype
                )
            self._load_streamed_checkpoint(self.model)
            self.model.eval()
        else:
            self.model = (
                AutoModelForImageTextToText.from_pretrained(
                    self.checkpoint,
                    trust_remote_code=True,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
                .to(self.device)
                .eval()
            )
        self.modeling_module = importlib.import_module(type(self.model).__module__)
        self.stats = self.model._get_robot_stats()
        self.metadata = self.stats.get_metadata("libero")

        core = self.model.model
        core._build_native_attention_bias = types.MethodType(
            self._batched_native_attention_bias, core
        )
        core.action_cuda_graph_manager.set_enabled(False)
        self.model.depth_decode_cuda_graph_manager.set_enabled(False)
        return self.model

    def _record_checkpoint(self, name: str, size: int, digest: str, source: str) -> None:
        if self.checkpoint_receipt.is_file():
            receipt = json.loads(self.checkpoint_receipt.read_text())
            if (receipt.get("repository") != CHECKPOINT_REPOSITORY
                    or receipt.get("revision") != CHECKPOINT_REVISION):
                raise RuntimeError("MolmoAct2 checkpoint receipt belongs to another snapshot")
        else:
            receipt = {
                "schema_version": 1, "repository": CHECKPOINT_REPOSITORY,
                "revision": CHECKPOINT_REVISION, "files": {},
            }
        receipt["files"][name] = {
            "bytes": size, "sha256": digest, "source": source,
            "verification": "full_file_sha256_before_tensor_load",
        }
        receipt["all_pinned_files_verified"] = set(receipt["files"]) == {
            item[0] for item in CHECKPOINT_SHARDS
        }
        self.checkpoint_receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_receipt.with_name(
            self.checkpoint_receipt.name + f".tmp.{os.getpid()}"
        )
        temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, self.checkpoint_receipt)

    def _load_streamed_checkpoint(self, model: torch.nn.Module) -> None:
        from accelerate.utils import set_module_tensor_to_device
        from safetensors.torch import load as load_safetensors

        expected_state = set(model.state_dict())
        index = json.loads((self.checkpoint / "model.safetensors.index.json").read_text())
        weight_map = index.get("weight_map") or {}
        if (len(weight_map) != 1295 or set(weight_map) != expected_state
                or set(weight_map.values()) != {item[0] for item in CHECKPOINT_SHARDS}):
            raise RuntimeError("MolmoAct2 checkpoint index/model state mapping differs")
        parameter_names = set(dict(model.named_parameters()))
        loaded = set()
        for name, size, expected_hash in CHECKPOINT_SHARDS:
            local = self.checkpoint / name
            if local.is_file():
                data = local.read_bytes()
                source = "resident_local_file"
            else:
                url = (
                    f"https://huggingface.co/{CHECKPOINT_REPOSITORY}/resolve/"
                    f"{CHECKPOINT_REVISION}/{name}?download=true"
                )
                request = urllib.request.Request(
                    url, headers={"User-Agent": "batch-invariance-molmoact2/1"}
                )
                buffer = io.BytesIO()
                digest = hashlib.sha256()
                received = 0
                with urllib.request.urlopen(request, timeout=600) as response:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        buffer.write(block)
                        digest.update(block)
                        received += len(block)
                if received != size or digest.hexdigest() != expected_hash:
                    raise RuntimeError(f"streamed MolmoAct2 shard failed hash/size: {name}")
                data = buffer.getvalue()
                source = "pinned_remote_memory_stream"
            if len(data) != size or hashlib.sha256(data).hexdigest() != expected_hash:
                raise RuntimeError(f"MolmoAct2 checkpoint shard differs: {name}")
            state = load_safetensors(data)
            expected_names = {key for key, filename in weight_map.items() if filename == name}
            if set(state) != expected_names:
                raise RuntimeError(f"MolmoAct2 shard tensor inventory differs: {name}")
            for tensor_name, value in state.items():
                set_module_tensor_to_device(
                    model, tensor_name, self.device, value=value,
                    dtype=self.dtype if tensor_name in parameter_names else None,
                )
                loaded.add(tensor_name)
            self._record_checkpoint(name, size, expected_hash, source)
            del state, data
            if "buffer" in locals():
                del buffer
        missing = expected_state - loaded
        meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
        meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
        # These two caches are explicitly non-persistent in the released model
        # and therefore are not checkpoint tensors.  Meta construction leaves
        # their initial ``torch.empty(0)`` values on meta; restore that exact
        # runtime-empty state on the target device.  The official forward path
        # deterministically builds the RoPE values on first use.
        expected_runtime_buffers = {
            name for name in meta_buffers
            if name.endswith("._pos_sin_cache") or name.endswith("._pos_cos_cache")
        }
        if set(meta_buffers) == expected_runtime_buffers and len(meta_buffers) == 2:
            for name in meta_buffers:
                module_name, buffer_name = name.rsplit(".", 1)
                module = model.get_submodule(module_name)
                module._buffers[buffer_name] = torch.empty(0, device=self.device)
            meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
        if missing or meta_parameters or meta_buffers:
            raise RuntimeError(
                f"MolmoAct2 strict streamed load incomplete: missing={len(missing)} "
                f"meta_parameters={len(meta_parameters)} meta_buffers={len(meta_buffers)}"
            )
        receipt = json.loads(self.checkpoint_receipt.read_text())
        receipt["strict_model_load"] = {
            "expected_state_tensors": len(expected_state),
            "loaded_state_tensors": len(loaded),
            "missing_state_tensors": [],
            "meta_parameters_remaining": [],
            "meta_buffers_remaining": [],
            "target_parameter_dtype": str(self.dtype),
        }
        temporary = self.checkpoint_receipt.with_name(
            self.checkpoint_receipt.name + f".tmp.{os.getpid()}"
        )
        temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, self.checkpoint_receipt)

    def _load_image(self, filename: str) -> Image.Image:
        return Image.open(self.sample_dir / filename).convert("RGB")

    def load_example_inputs(self, count: int):
        if self.model is None or self.processor is None or self.stats is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")

        target_frame = int(os.environ.get("MOLMOACT2_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("MOLMOACT2_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)
        frame_indices = frame_indices[:count]

        state_rows = pq.read_table(
            self.sample_dir / "data/chunk-000/file-000.parquet",
            columns=["observation.state"],
        )["observation.state"].to_pylist()
        normalized_task = self.modeling_module._normalize_question_text(TASK)
        examples = []
        for frame_index in frame_indices:
            state = np.asarray(state_rows[frame_index], dtype=np.float32)
            normalized_state = np.asarray(
                self.stats.normalize_state(state, "libero"), dtype=np.float32
            )
            discrete_state = self.modeling_module._build_discrete_state_string(
                normalized_state, int(self.model.config.num_state_tokens)
            )
            text = self.modeling_module._build_robot_text(
                task=normalized_task,
                style="robot_action",
                discrete_state_string=discrete_state,
                setup_type=str(self.metadata.get("setup_type", "")),
                control_mode=str(self.metadata.get("control_mode", "")),
                add_setup_tokens=bool(self.model.config.add_setup_tokens),
                add_control_tokens=bool(self.model.config.add_control_tokens),
                num_images=2,
            )
            image_number = frame_index + 1
            processed = self.processor(
                text=text,
                images=[
                    self._load_image(f"frames/image_{image_number:03d}.png"),
                    self._load_image(f"frames/image2_{image_number:03d}.png"),
                ],
                return_tensors="pt",
            )
            examples.append(
                dict(processed)
                | {"noise_seed": torch.tensor([271_828 + frame_index], dtype=torch.int64)}
            )
        return examples

    def _load_replay_metadata(self) -> None:
        if self.replay_manifest is not None:
            return
        manifest = json.loads(self.replay_manifest_path.read_text())
        if (manifest.get("model_id") != "molmoact2"
                or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
                or len(manifest.get("observations", [])) != 1000
                or manifest.get("noise_shape") != [1000, 3, 10, 32]):
            raise ValueError("MolmoAct2 full replay manifest differs")
        noise = np.load(manifest["noise_file"], mmap_mode="r", allow_pickle=False)
        if noise.shape != (1000, 3, 10, 32) or noise.dtype != np.float32:
            raise ValueError("MolmoAct2 explicit request noise differs")
        self.replay_manifest = manifest
        self.replay_noise = noise

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.model is None or self.processor is None or self.stats is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        self._load_replay_metadata()
        if not 0 <= replay_index < 1000 or not 0 <= noise_id < 3:
            raise IndexError((replay_index, noise_id))
        row = self.replay_manifest["observations"][replay_index]
        normalized_task = self.modeling_module._normalize_question_text(row["task"])
        state = np.asarray(row["state"], dtype=np.float32)
        normalized_state = np.asarray(
            self.stats.normalize_state(state, "libero"), dtype=np.float32
        )
        discrete_state = self.modeling_module._build_discrete_state_string(
            normalized_state, int(self.model.config.num_state_tokens)
        )
        text = self.modeling_module._build_robot_text(
            task=normalized_task,
            style="robot_action",
            discrete_state_string=discrete_state,
            setup_type=str(self.metadata.get("setup_type", "")),
            control_mode=str(self.metadata.get("control_mode", "")),
            add_setup_tokens=bool(self.model.config.add_setup_tokens),
            add_control_tokens=bool(self.model.config.add_control_tokens),
            num_images=2,
        )
        fixture = Path(self.replay_manifest["fixture_root"])
        processed = self.processor(
            text=text,
            images=[
                Image.open(fixture / row["images"]["image"]).convert("RGB"),
                Image.open(fixture / row["images"]["image2"]).convert("RGB"),
            ],
            return_tensors="pt",
        )
        return dict(processed) | {
            "trajectory": torch.from_numpy(
                np.asarray(self.replay_noise[replay_index, noise_id]).copy()
            ),
        }

    @staticmethod
    def _runtime_image_to_pil(value: Any) -> Image.Image:
        """Recover the exact uint8 image supplied by LeRobot's env pipeline."""

        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to("cpu")
            if tensor.ndim != 3:
                raise ValueError(f"MolmoAct2 runtime image must be rank 3, got {tensor.shape}")
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            if tensor.shape[-1] != 3:
                raise ValueError(f"MolmoAct2 runtime image must have three channels: {tensor.shape}")
            if tensor.dtype.is_floating_point:
                tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
            else:
                tensor = tensor.to(torch.uint8)
            array = tensor.contiguous().numpy()
        else:
            array = np.asarray(value)
            if array.ndim != 3:
                raise ValueError(f"MolmoAct2 runtime image must be rank 3, got {array.shape}")
            if array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.moveaxis(array, 0, -1)
            if array.shape[-1] != 3:
                raise ValueError(f"MolmoAct2 runtime image must have three channels: {array.shape}")
            if np.issubdtype(array.dtype, np.floating):
                array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8, copy=False)
        return Image.fromarray(array, mode="RGB")

    def prepare_runtime_sample(
        self,
        *,
        image: Any,
        wrist_image: Any,
        state: Any,
        task: str,
        trajectory: Any,
    ) -> dict[str, Any]:
        """Create one official checkpoint request from a live LIBERO observation."""

        if self.model is None or self.processor is None or self.stats is None:
            raise RuntimeError("load_model must be called before prepare_runtime_sample")
        state_array = np.asarray(
            state.detach().to("cpu") if isinstance(state, torch.Tensor) else state,
            dtype=np.float32,
        )
        if state_array.shape != (8,):
            raise ValueError(f"MolmoAct2 LIBERO state must be 8-D, got {state_array.shape}")
        normalized_state = np.asarray(
            self.stats.normalize_state(state_array, "libero"), dtype=np.float32
        )
        normalized_task = self.modeling_module._normalize_question_text(str(task))
        discrete_state = self.modeling_module._build_discrete_state_string(
            normalized_state, int(self.model.config.num_state_tokens)
        )
        text = self.modeling_module._build_robot_text(
            task=normalized_task,
            style="robot_action",
            discrete_state_string=discrete_state,
            setup_type=str(self.metadata.get("setup_type", "")),
            control_mode=str(self.metadata.get("control_mode", "")),
            add_setup_tokens=bool(self.model.config.add_setup_tokens),
            add_control_tokens=bool(self.model.config.add_control_tokens),
            num_images=2,
        )
        processed = self.processor(
            text=text,
            images=[
                self._runtime_image_to_pil(image),
                self._runtime_image_to_pil(wrist_image),
            ],
            return_tensors="pt",
        )
        trajectory_tensor = torch.as_tensor(trajectory).detach().to("cpu", dtype=torch.float32)
        if tuple(trajectory_tensor.shape) != (10, 32):
            raise ValueError(
                f"MolmoAct2 runtime trajectory must be 10x32, got {trajectory_tensor.shape}"
            )
        return dict(processed) | {"trajectory": trajectory_tensor}

    def compose_batch(self, samples, *, fixed_prompt: bool = False):
        from batch_invariant_ops import is_batch_invariant_mode_enabled

        sequence_keys = {"input_ids", "attention_mask", "token_type_ids"}
        result = {
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in samples[0]
            if key not in sequence_keys and key in {
                "pixel_values",
                "image_token_pooling",
                "image_grids",
                "image_num_crops",
                "noise_seed",
            }
        }
        lengths = [int(sample["input_ids"].shape[1]) for sample in samples]
        max_length = (
            self.canonical_prompt_length
            if fixed_prompt or (
                self.canonical_prompt_padding and is_batch_invariant_mode_enabled()
            ) else max(lengths)
        )
        if any(length > max_length for length in lengths):
            raise ValueError("MolmoAct2 prompt exceeds canonical frozen LIBERO length")
        padding_side = self.processor.tokenizer.padding_side
        if padding_side not in ("left", "right"):
            raise ValueError(f"unsupported MolmoAct2 padding side: {padding_side}")
        pad_values = {
            "input_ids": int(self.processor.tokenizer.pad_token_id),
            "attention_mask": 0,
            "token_type_ids": 0,
        }
        for key in sequence_keys:
            if key not in samples[0]:
                continue
            rows = []
            for sample, length in zip(samples, lengths):
                value = sample[key][0]
                padded = torch.full(
                    (max_length,), pad_values[key], dtype=value.dtype
                )
                if padding_side == "left":
                    padded[max_length - length:] = value
                else:
                    padded[:length] = value
                rows.append(padded)
            result[key] = torch.stack(rows)
        if "trajectory" in samples[0]:
            result["trajectory"] = torch.stack([sample["trajectory"] for sample in samples])
        return result

    def compose_fixed_batch(self, samples):
        return self.compose_batch(samples, fixed_prompt=True)

    def fixed_batch_shape_metadata(self, batch):
        return {
            "language_padding": "max_length_right",
            "language_tokens": int(batch["input_ids"].shape[1]),
            "canonical_length": self.canonical_prompt_length,
        }

    def execution_configuration(self):
        if self.processor is None:
            raise RuntimeError("load_model must be called before execution_configuration")
        return {
            "camera_features": [
                "observation.images.image", "observation.images.wrist_image",
            ],
            "source_image_shape_hwc": [256, 256, 3],
            "processed_pixel_values_shape_per_request": [2, 729, 588],
            "processed_image_token_pooling_shape_per_request": [392, 4],
            "state_dim": 8,
            "native_language_padding": "longest_in_batch_right",
            "invariant_language_padding": "max_length_right",
            "canonical_language_tokens": self.canonical_prompt_length,
            "language_pad_token_id": int(self.processor.tokenizer.pad_token_id),
            "flow_steps": 10,
            "max_action_horizon": 10,
            "max_action_dim": 32,
            "robot_output_dim": 7,
            "precision": "bfloat16",
        }

    def reset_model(self, model) -> None:
        expert = model.model.action_expert
        expert._modulation_cache_key = None
        expert._modulation_cache_value = None

    def _generate_actions_with_trajectory(
        self,
        core,
        inputs,
        *,
        trajectory,
        action_dim_is_pad,
        action_horizon,
        num_steps,
    ):
        """Released continuous-action body with explicit initial trajectory."""

        action_expert = core._require_action_expert()
        outputs = core(**inputs, use_cache=True)
        encoder_kv_states = core._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = core._get_encoder_attention_mask(
            inputs["input_ids"], inputs.get("attention_mask")
        )
        depth_gate, depth_mask = core._depth_gate_from_condition(
            input_ids=inputs["input_ids"],
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = core._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states, depth_mask, depth_gate
        )
        trajectory = core._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=core.config.mask_action_dim_padding,
        )
        action_context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=None,
            batch_size=trajectory.shape[0],
            seq_len=trajectory.shape[1],
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        flow_timesteps = [
            torch.full(
                (trajectory.shape[0],),
                index / num_steps,
                device=trajectory.device,
                dtype=torch.float32,
            )
            for index in range(num_steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(
                num_steps,
                trajectory.shape[0],
                trajectory.device,
                trajectory.dtype,
            ),
        )
        flow_inputs = self.modeling_module._ActionFlowInputs(
            trajectory=trajectory,
            context=action_context,
            modulations=modulation_cache,
            action_dim_is_pad=action_dim_is_pad,
        )
        dt = 1.0 / num_steps
        trajectory = flow_inputs.trajectory
        self._trace("action.flow.initial_state", trajectory)
        for index in range(num_steps):
            velocity = action_expert.forward_with_context(
                trajectory,
                flow_inputs.modulations[index].conditioning,
                context=flow_inputs.context,
                modulation=flow_inputs.modulations[index],
            )
            velocity = core._mask_action_dim_tensor(
                velocity,
                action_dim_is_pad=flow_inputs.action_dim_is_pad,
                enabled=core.config.mask_action_dim_padding,
            )
            self._trace(f"action.flow.step_{index:02d}.velocity", velocity)
            trajectory = trajectory + dt * velocity
            trajectory = core._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=flow_inputs.action_dim_is_pad,
                enabled=core.config.mask_action_dim_padding,
            )
            self._trace(f"flow.step_{index:02d}.state", trajectory)
            self._trace(f"action.flow.step_{index:02d}.state", trajectory)
        return trajectory

    def run_model(self, model, batch):
        inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if key not in {"noise_seed", "trajectory"}
        }
        batch_size = int(inputs["input_ids"].shape[0])
        action_dim = int(self.stats.get_action_dim("libero"))
        action_horizon = int(self.stats.get_action_horizon("libero"))
        action_dim_is_pad = model._build_action_dim_is_pad(
            action_dim=action_dim,
            max_action_dim=int(model.config.max_action_dim),
            batch_size=batch_size,
            device=self.device,
        )
        trajectory_dtype = model.model.action_expert.action_embed.weight.dtype
        if "trajectory" in batch:
            trajectory = batch["trajectory"].to(
                device=self.device, dtype=trajectory_dtype
            )
            expected = (batch_size, action_horizon, int(model.config.max_action_dim))
            if tuple(trajectory.shape) != expected:
                raise ValueError(f"MolmoAct2 explicit trajectory shape differs: {trajectory.shape}")
        else:
            trajectories = []
            for seed in batch["noise_seed"].reshape(-1).tolist():
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
                trajectories.append(
                    torch.randn(
                        (1, action_horizon, int(model.config.max_action_dim)),
                        device=self.device,
                        dtype=trajectory_dtype,
                        generator=generator,
                    )
                )
            trajectory = torch.cat(trajectories, dim=0)
        num_steps = int(os.environ.get("MOLMOACT2_STEPS", "10"))
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            actions = self._generate_actions_with_trajectory(
                model.model,
                inputs,
                trajectory=trajectory,
                action_dim_is_pad=action_dim_is_pad,
                action_horizon=action_horizon,
                num_steps=num_steps,
            )
        actions = model._slice_action_dim(actions, action_dim)
        actions = model._slice_action_chunk(
            actions,
            int(model.config.n_obs_steps),
            int(self.stats.get_n_action_steps("libero")),
        )
        actions = self.stats.unnormalize_action(actions, "libero")
        return actions.to(device=self.device, dtype=torch.float32)

    def extract_robot_output(self, output):
        return output


adapter = MolmoAct2Adapter()

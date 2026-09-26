"""Cosmos 3 Nano Policy DROID adapter using NVIDIA's official PyTorch code.

Required environment variables:

``COSMOS3_NANO_CHECKPOINT``
    Local snapshot of ``nvidia/Cosmos3-Nano-Policy-DROID``.
``COSMOS3_DROID_SAMPLE_DIR``
    Partial ``nvidia/Cosmos3-DROID`` snapshot containing the first data parquet,
    ``meta/tasks.parquet``, and 64 extracted PNG frames for each of the three
    cameras under ``frames/{wrist,exterior1,exterior2}``.

The model produces actions and a future-video latent in one four-step UniPC
generation.  Per-example diffusion seeds are explicit adapter inputs so the
target sample receives identical noise at every batch size.  Set
``COSMOS3_DECODE_VIDEO=1`` to additionally batch-decode the generated video
latents through the released Wan2.2 VAE.  The adapter defaults to NVIDIA's
eager inference switch because ``torch.compile`` captures lower-level kernels
before this library's dispatcher context is selected; set
``COSMOS3_TORCH_COMPILE=1`` to reproduce the upstream compiled serving path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


class Cosmos3NanoPolicyAdapter:
    name = "nvidia-cosmos3-nano-policy-droid-official-pytorch"
    checkpoint_environment_variable = "COSMOS3_NANO_CHECKPOINT"
    format_prompt_as_json: bool | None = None
    guidance_interval: tuple[float, float] | None = None

    def __init__(self) -> None:
        self.checkpoint = Path(
            os.environ.get(self.checkpoint_environment_variable, "")
        )
        self.sample_dir = Path(os.environ.get("COSMOS3_DROID_SAMPLE_DIR", ""))
        self.output_dir = Path(
            os.environ.get("COSMOS3_OUTPUT_DIR", "/tmp/cosmos3_batch_invariance")
        )
        self.decode_video = os.environ.get("COSMOS3_DECODE_VIDEO", "0") == "1"
        self.use_torch_compile = os.environ.get("COSMOS3_TORCH_COMPILE", "0") == "1"
        self.service = None

    def _validate_paths(self) -> None:
        checkpoint_files = (
            "config.json",
            "model.safetensors.index.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors",
            "vision_encoder/model.safetensors",
        )
        sample_files = (
            "success/data/chunk-000/file-000.parquet",
            "success/meta/tasks.parquet",
            "frames/wrist/001.png",
            "frames/wrist/064.png",
            "frames/exterior1/001.png",
            "frames/exterior1/064.png",
            "frames/exterior2/001.png",
            "frames/exterior2/064.png",
        )
        for root, filenames in (
            (self.checkpoint, checkpoint_files),
            (self.sample_dir, sample_files),
        ):
            for filename in filenames:
                if not (root / filename).is_file():
                    raise FileNotFoundError(f"missing required fixture: {root / filename}")

    def load_model(self):
        self._validate_paths()
        from batch_invariant_ops import (
            is_batch_invariant_mode_enabled,
            varlen_scaled_dot_product_attention_batch_invariant,
        )
        from cosmos_framework.model.generator.mot import attention as attention_module
        from cosmos_framework.model.generator.mot import (
            inference_text_kv_memory as text_kv_module,
        )
        from cosmos_framework.scripts.action_policy_server_robolab import (
            RobolabPolicyService,
            RobolabServerArgs,
        )

        def attention_repair_enabled() -> bool:
            return (
                is_batch_invariant_mode_enabled()
                and os.environ.get("_COSMOS3_DISABLE_ATTENTION_REPAIR", "0") != "1"
            )

        # Stock Cosmos deliberately selects dense attention for B=1 and packed
        # varlen attention for B>1.  Keep the stock branch in the baseline, but
        # use the same mathematically equivalent packed path at every batch size
        # while batch-invariant mode is active.
        original_use_varlen = attention_module._use_varlen
        if not getattr(original_use_varlen, "_batch_invariant_cosmos3", False):

            def use_varlen_independent_of_batch(num_samples, *, has_caption_offsets):
                if attention_repair_enabled():
                    return True
                return original_use_varlen(num_samples, has_caption_offsets=has_caption_offsets)

            use_varlen_independent_of_batch._batch_invariant_cosmos3 = True
            attention_module._use_varlen = use_varlen_independent_of_batch

        original_attention = attention_module.attention
        if not getattr(original_attention, "_batch_invariant_cosmos3", False):

            def attention_with_batch_invariant_varlen(
                query,
                key,
                value,
                *,
                is_causal=False,
                causal_type=None,
                scale=None,
                cumulative_seqlen_Q=None,
                cumulative_seqlen_KV=None,
                return_lse=False,
                **kwargs,
            ):
                if (
                    attention_repair_enabled()
                    and cumulative_seqlen_Q is not None
                    and cumulative_seqlen_KV is not None
                    and not return_lse
                ):
                    return varlen_scaled_dot_product_attention_batch_invariant(
                        query,
                        key,
                        value,
                        cumulative_seqlen_Q,
                        cumulative_seqlen_KV,
                        is_causal=is_causal,
                        causal_type=causal_type,
                        scale=scale,
                    )
                return original_attention(
                    query,
                    key,
                    value,
                    is_causal=is_causal,
                    causal_type=causal_type,
                    scale=scale,
                    cumulative_seqlen_Q=cumulative_seqlen_Q,
                    cumulative_seqlen_KV=cumulative_seqlen_KV,
                    return_lse=return_lse,
                    **kwargs,
                )

            attention_with_batch_invariant_varlen._batch_invariant_cosmos3 = True
            attention_module.attention = attention_with_batch_invariant_varlen
            text_kv_module.attention = attention_with_batch_invariant_varlen

        original_memory_init = text_kv_module.InferenceTextKVMemoryState.init
        if not getattr(original_memory_init, "_batch_invariant_cosmos3", False):

            def memory_init_independent_of_batch(self, hidden_states, device):
                original_memory_init(self, hidden_states, device)
                if not attention_repair_enabled() or self._kv_reorder_indices is not None:
                    return
                und_offsets = hidden_states.get("_causal_seq_offsets")
                gen_offsets = hidden_states.get("_full_only_seq_offsets")
                if not isinstance(und_offsets, torch.Tensor) or not isinstance(
                    gen_offsets, torch.Tensor
                ):
                    return
                und_offsets = text_kv_module.drop_pad_segment(hidden_states, und_offsets).to(
                    torch.int32
                )
                gen_offsets = text_kv_module.drop_pad_segment(hidden_states, gen_offsets).to(
                    torch.int32
                )
                if und_offsets.numel() != 2 or gen_offsets.numel() != 2:
                    return
                und_length = und_offsets[1] - und_offsets[0]
                gen_length = gen_offsets[1] - gen_offsets[0]
                total_length = int((und_length + gen_length).item())
                self._kv_reorder_indices = torch.arange(total_length, device=und_offsets.device)
                self._cumulative_seqlen_q = gen_offsets
                self._cumulative_seqlen_kv = und_offsets + gen_offsets
                self._max_seqlen_q = int(gen_length.item())
                self._max_seqlen_kv = total_length

            memory_init_independent_of_batch._batch_invariant_cosmos3 = True
            text_kv_module.InferenceTextKVMemoryState.init = memory_init_independent_of_batch

        use_torch_compile = self.use_torch_compile

        class ConfiguredRobolabPolicyService(RobolabPolicyService):
            def _build_setup_args(self, args, parallelism_overrides):
                setup = super()._build_setup_args(args, parallelism_overrides)
                return setup.model_copy(
                    update={
                        "use_torch_compile": use_torch_compile,
                        "use_cuda_graphs": False,
                    }
                )

        args = RobolabServerArgs(
            checkpoint_path=str(self.checkpoint),
            output_dir=self.output_dir,
            deterministic_seed=True,
            decode_video=self.decode_video,
            format_prompt_as_json=self.format_prompt_as_json,
            guidance_interval=self.guidance_interval,
        )
        self.service = ConfiguredRobolabPolicyService(args)
        return self.service

    def _image(self, camera: str, frame_number: int) -> np.ndarray:
        path = self.sample_dir / "frames" / camera / f"{frame_number:03d}.png"
        return np.array(Image.open(path).convert("RGB"), copy=True)

    def load_example_inputs(self, count: int):
        if self.service is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible DROID fixture contains 64 frames")

        data_path = self.sample_dir / "success/data/chunk-000/file-000.parquet"
        rows = (
            pq.read_table(
                data_path,
                columns=[
                    "observation.state.joint_positions",
                    "observation.state.gripper_position",
                    "task_index",
                ],
            )
            .slice(0, 64)
            .to_pylist()
        )
        tasks_table = pq.read_table(
            self.sample_dir / "success/meta/tasks.parquet",
            columns=["task_index", "task"],
        )
        tasks = {int(row["task_index"]): row["task"] for row in tasks_table.to_pylist()}

        target_frame = int(os.environ.get("COSMOS3_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("COSMOS3_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)

        examples = []
        for frame_index in frame_indices[:count]:
            row = rows[frame_index]
            frame_number = frame_index + 1
            task = tasks[int(row["task_index"])].split(" | ", 1)[0]
            gripper = np.asarray(
                [row["observation.state.gripper_position"]], dtype=np.float32
            ).reshape(1, 1)
            observation = {
                "prompt": task,
                "observation/wrist_image_left": self._image("wrist", frame_number),
                "observation/exterior_image_1_left": self._image("exterior1", frame_number),
                "observation/exterior_image_2_left": self._image("exterior2", frame_number),
                "observation/joint_position": np.asarray(
                    row["observation.state.joint_positions"], dtype=np.float32
                )[None],
                "observation/gripper_position": gripper,
            }
            examples.append(
                {
                    "observation": observation,
                    "sample": self.service._build_sample(observation),
                    "seed": 73_000 + frame_index,
                }
            )
        return examples

    @staticmethod
    def compose_batch(samples):
        from cosmos_framework.scripts.action_policy_server_robolab import (
            _build_data_batch_from_sample,
        )

        singleton_batches = [_build_data_batch_from_sample(sample["sample"]) for sample in samples]
        keys = singleton_batches[0].keys()
        data_batch = {
            key: [item for batch in singleton_batches for item in batch[key]] for key in keys
        }
        return {
            "data_batch": data_batch,
            "seeds": [sample["seed"] for sample in samples],
        }

    @staticmethod
    def reset_model(model) -> None:
        del model

    def run_model(self, service, batch):
        cfg = service.cfg
        samples = service.model.generate_samples_from_batch(
            batch["data_batch"],
            guidance=cfg.guidance,
            guidance_interval=(
                list(cfg.guidance_interval) if cfg.guidance_interval is not None else None
            ),
            seed=batch["seeds"],
            num_steps=cfg.num_steps,
            shift=cfg.shift,
        )

        actions = []
        for action in samples["action"]:
            action = action[:, : cfg.action_dim][cfg.history_length :].clone()
            action[:, -1] = 1.0 - action[:, -1]
            actions.append(action)
        output = {
            "action": torch.stack(actions),
            "vision_latent": torch.cat(samples["vision"], dim=0),
        }
        if self.decode_video:
            video = service.model.decode(output["vision_latent"])
            output["video"] = (
                ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(0, 2, 3, 4, 1)
            )
        return output

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = Cosmos3NanoPolicyAdapter()

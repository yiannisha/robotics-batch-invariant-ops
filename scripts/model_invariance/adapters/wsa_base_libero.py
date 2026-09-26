"""Official WSA Base PyTorch LIBERO adapter with explicit flow noise.

Required environment variables:

``WSA_SOURCE``
    Official ``zaleni/WSA`` source checkout with the inference-only compatibility
    patch recorded in ``scripts/model_invariance/patches`` applied.
``WSA_CHECKPOINT``
    Local native snapshot of ``zaleni/WSA-Base-LIBERO``.
``WSA_QWEN_METADATA``
    Metadata/tokenizer-only snapshot of ``Qwen/Qwen3-VL-2B-Instruct``.
``WSA_COSMOS_TOKENIZER``
    Snapshot containing the official Cosmos CI8x8 ``encoder.jit`` and
    ``decoder.jit``.
``WSA_SAMPLE_DIR``
    Public LeRobot LIBERO episode-zero fixture used by the other adapters.

The released WSA checkpoint contains the full learned Qwen3-VL, generation,
and action stacks.  Qwen metadata are used only by preprocessing and model
construction; no separate learned Qwen weights are loaded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors import safe_open


class WSABaseLiberoAdapter:
    name = "wsa-base-official-pytorch-libero"

    def __init__(self) -> None:
        self.source = Path(os.environ.get("WSA_SOURCE", ""))
        self.checkpoint = Path(os.environ.get("WSA_CHECKPOINT", ""))
        self.qwen_metadata = Path(os.environ.get("WSA_QWEN_METADATA", ""))
        self.cosmos_tokenizer = Path(os.environ.get("WSA_COSMOS_TOKENIZER", ""))
        self.sample_dir = Path(os.environ.get("WSA_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.processor = None
        self.resize = None
        self.state_stats: dict[str, np.ndarray] | None = None
        self.action_stats: dict[str, np.ndarray] | None = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        required = (
            self.source / "src/lerobot/policies/WSA_Base/modeling_wsa_base.py",
            self.source
            / "src/lerobot/policies/WSA_Base/transformers_replace/models/qwen3_vl/modeling_qwen3_vl.py",
            self.checkpoint / "config.json",
            self.checkpoint / "model.safetensors",
            self.checkpoint / "stats.json",
            self.qwen_metadata / "config.json",
            self.qwen_metadata / "tokenizer.json",
            self.cosmos_tokenizer / "encoder.jit",
            self.cosmos_tokenizer / "decoder.jit",
            self.sample_dir / "frames/image_001.png",
            self.sample_dir / "frames/image_064.png",
            self.sample_dir / "frames/image2_001.png",
            self.sample_dir / "frames/image2_064.png",
            self.sample_dir / "data/chunk-000/file-000.parquet",
            self.sample_dir / "tasks.parquet",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing WSA artifacts: {missing}")

    def _add_source_path(self) -> None:
        source = str(self.source / "src")
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)

    @staticmethod
    def _checkpoint_key_audit(policy, checkpoint_file: Path) -> None:
        with safe_open(checkpoint_file, framework="pt") as handle:
            saved = set(handle.keys())
        state = set(policy.state_dict())
        missing = state - saved
        unexpected = saved - state
        allowed_missing = {
            key
            for key in missing
            if key.startswith("model.cosmos.")
            or key
            == "model.qwen3_vl_with_expert.und_expert.model.language_model.embed_tokens.weight"
        }
        bad_missing = sorted(missing - allowed_missing)
        if bad_missing or unexpected:
            raise RuntimeError(
                "official WSA checkpoint key mismatch: "
                f"missing={bad_missing}, unexpected={sorted(unexpected)}"
            )
        qwen = policy.model.qwen3_vl_with_expert.und_expert
        if qwen.get_input_embeddings().weight.data_ptr() != qwen.lm_head.weight.data_ptr():
            raise RuntimeError("the omitted Qwen embedding alias is not tied to lm_head.weight")

    def load_model(self):
        self._validate_paths()
        self._add_source_path()

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.WSA_Base.configuration_wsa_base import WSABaseConfig  # noqa: F401
        from lerobot.policies.WSA_Base.modeling_wsa_base import WSABasePolicy
        from lerobot.policies.WSA_Base.transform_wsa_base import Qwen3_VLProcessorTransformFn
        from lerobot.transforms.core import ResizeImagesWithPadFn

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.lambda_3d = 0.0  # Official evaluation disables the training-only DA3 teacher.
        config.compile_model = False
        config.device = str(self.device)
        config.cosmos_device = str(self.device)
        config.qwen3_vl_pretrained_path = str(self.qwen_metadata)
        config.cosmos_tokenizer_path_or_name = str(self.cosmos_tokenizer)
        policy = WSABasePolicy.from_pretrained(
            self.checkpoint,
            config=config,
            strict=False,
        )
        self._checkpoint_key_audit(policy, self.checkpoint / "model.safetensors")
        policy.to(device=self.device, dtype=torch.bfloat16).eval()

        self.resize = ResizeImagesWithPadFn(height=224, width=224)
        self.processor = Qwen3_VLProcessorTransformFn(
            pretrained_model_name_or_path=str(self.qwen_metadata),
            max_length=int(config.tokenizer_max_length),
        )
        stats_root = json.loads((self.checkpoint / "stats.json").read_text())
        if len(stats_root) != 1:
            raise RuntimeError(f"expected one WSA statistics set, got {list(stats_root)}")
        stats = next(iter(stats_root.values()))
        self.state_stats = {
            name: np.asarray(stats["observation.state"][name], dtype=np.float32)
            for name in ("mean", "std")
        }
        self.action_stats = {
            name: np.asarray(stats["action"][name], dtype=np.float32) for name in ("mean", "std")
        }
        return policy

    def _image(self, camera: str, frame_number: int) -> torch.Tensor:
        array = np.array(
            Image.open(self.sample_dir / "frames" / f"{camera}_{frame_number:03d}.png").convert(
                "RGB"
            ),
            copy=True,
        )
        return torch.from_numpy(array).permute(2, 0, 1).float().div(255)

    def load_example_inputs(self, count: int):
        if self.processor is None or self.resize is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")

        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        rows = (
            pq.read_table(
                self.sample_dir / "data/chunk-000/file-000.parquet",
                columns=["observation.state", "task_index"],
            )
            .slice(0, 64)
            .to_pylist()
        )
        tasks = {
            int(row["task_index"]): row["__index_level_0__"]
            for row in pq.read_table(self.sample_dir / "tasks.parquet").to_pylist()
        }
        target_frame = int(os.environ.get("WSA_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("WSA_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)

        examples = []
        for frame_index in frame_indices[:count]:
            frame_number = frame_index + 1
            primary = self._image("image", frame_number)
            wrist = self._image("image2", frame_number)
            # Match the released server's one-frame request behavior: duplicate
            # the current frame into the two-frame history and mask a missing view.
            sample = {
                f"{OBS_IMAGES}.image0": torch.stack([primary, primary]),
                f"{OBS_IMAGES}.image1": torch.stack([wrist, wrist]),
                f"{OBS_IMAGES}.image2": torch.zeros(2, *primary.shape),
                OBS_STATE: torch.tensor(
                    rows[frame_index]["observation.state"], dtype=torch.float32
                ),
                "task": tasks[int(rows[frame_index]["task_index"])],
            }
            sample = self.resize(sample)
            sample[f"{OBS_IMAGES}.image0_mask"] = torch.tensor(True)
            sample[f"{OBS_IMAGES}.image1_mask"] = torch.tensor(True)
            sample[f"{OBS_IMAGES}.image2_mask"] = torch.tensor(False)
            sample = self.processor(sample)
            mean = torch.from_numpy(self.state_stats["mean"])
            std = torch.from_numpy(self.state_stats["std"])
            sample[OBS_STATE] = (sample[OBS_STATE] - mean) / (std + 1e-6)

            generator = torch.Generator(device=self.device).manual_seed(97_000 + frame_index)
            sample["noise"] = torch.randn(
                10,
                32,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            ).cpu()
            examples.append(
                {
                    key: value.detach().cpu()
                    for key, value in sample.items()
                    if isinstance(value, torch.Tensor)
                }
            )
        return examples

    @staticmethod
    def compose_batch(samples):
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}

    @staticmethod
    def reset_model(policy) -> None:
        policy.reset()

    def _prepare_model_inputs(self, policy, batch):
        from lerobot.utils.constants import OBS_PREFIX

        model_batch = {}
        for key, value in batch.items():
            if key == "noise":
                continue
            if value.dtype == torch.bool or not value.is_floating_point():
                model_batch[key] = value.to(self.device)
            else:
                model_batch[key] = value.to(device=self.device, dtype=torch.bfloat16)
        images, image_masks = policy._preprocess_images(model_batch)
        state = policy.prepare_state(model_batch)
        return (
            images,
            image_masks,
            model_batch[f"{OBS_PREFIX}pixel_values"],
            model_batch[f"{OBS_PREFIX}image_grid_thw"],
            model_batch[f"{OBS_PREFIX}input_ids"],
            model_batch[f"{OBS_PREFIX}attention_mask"],
            state,
            batch["noise"].to(self.device),
        )

    def _sample_actions_causal(self, policy, prepared):
        from lerobot.policies.WSA_Base.modeling_wsa_base import make_att_2d_masks

        images, image_masks, pixel_values, image_grid_thw, tokens, token_masks, state, noise = (
            prepared
        )
        model = policy.model
        batch_size = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            pixel_values, image_grid_thw, tokens, token_masks
        )
        self._trace("prefix.embeddings", prefix_embs)

        middle_images = images[:, :, :2]
        virtual_visual_tokens = model.infer_middle_visual_token_count(middle_images)
        middle_embs, middle_pad_masks, middle_att_masks = model.embed_middle_queries_only(
            batch_size=batch_size,
            device=state.device,
        )
        self._trace("middle.query_embeddings", middle_embs)
        zero_actions = torch.zeros(
            batch_size,
            model.config.chunk_size,
            model.config.max_action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        zero_time = torch.zeros(batch_size, device=state.device, dtype=torch.float32)
        _, suffix_pad_masks, suffix_att_masks = model.embed_suffix(state, zero_actions, zero_time)

        if (
            model.qwen3_vl_with_expert.und_expert.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(torch.bfloat16)
            middle_embs = middle_embs.to(torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)
        attention = model.build_training_attention_mask(
            pad_masks,
            att_masks,
            prefix_len=prefix_pad_masks.shape[1],
        )
        attention = model._prepare_attention_masks_4d(attention)
        position_ids, _ = model.get_position_ids_with_omitted_middle_visual_tokens(
            tokens,
            image_grid_thw,
            prefix_pad_masks,
            middle_pad_masks,
            suffix_pad_masks,
            virtual_visual_tokens,
        )

        step_size = torch.tensor(
            -1.0 / model.config.num_inference_steps,
            dtype=torch.float32,
            device=self.device,
        )
        actions = noise
        with model._temporary_attention_implementations(
            und_expert_impl="eager",
            gen_expert_impl="eager",
            act_expert_impl="eager",
        ):
            for step in range(model.config.num_inference_steps):
                timestep = torch.tensor(
                    1.0 + step * float(step_size),
                    dtype=torch.float32,
                    device=self.device,
                ).expand(batch_size)
                suffix_embs, _, _ = model.embed_suffix(
                    state,
                    actions.to(state.dtype),
                    timestep.to(state.dtype),
                )
                if prefix_embs.dtype == torch.bfloat16:
                    suffix_embs = suffix_embs.to(torch.bfloat16)
                outputs, _ = model.qwen3_vl_with_expert.forward(
                    attention_mask=attention,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
                    use_cache=False,
                )
                velocity = model.action_out_proj(
                    outputs[2][:, -model.config.chunk_size :].to(torch.float32)
                )
                self._trace(f"flow.step_{step:02d}.velocity", velocity)
                actions = actions + step_size * velocity
                self._trace(f"flow.step_{step:02d}.state", actions)
        return actions

    def _postprocess(self, policy, actions):
        del policy
        action_dim = self.action_stats["mean"].shape[0]
        actions = actions[..., :action_dim]
        self._trace("actions.normalized", actions)
        mean = torch.from_numpy(self.action_stats["mean"]).to(actions)
        std = torch.from_numpy(self.action_stats["std"]).to(actions)
        actions = actions * (std + 1e-6) + mean
        self._trace("actions.unnormalized", actions)
        return actions

    def run_model(self, policy, batch):
        prepared = self._prepare_model_inputs(policy, batch)
        self._trace("preprocess.state", prepared[-2])
        self._trace("input.noise", prepared[-1])
        return self._postprocess(policy, self._sample_actions_causal(policy, prepared))

    def run_official_model(self, policy, batch):
        prepared = self._prepare_model_inputs(policy, batch)
        actions, _ = policy.model.sample_actions(
            *prepared[:-1], noise=prepared[-1], decode_image=False
        )
        return self._postprocess(policy, actions)

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = WSABaseLiberoAdapter()

"""Serve genuine batched InternVLA-A1.5 inference for paired LIBERO runs.

The upstream websocket backend intentionally loops over ``payload.examples``
and therefore exposes only serial singleton execution.  This wrapper keeps
its official checkpoint loader, preprocessing, normalization, model, action
postprocessing, protocol, and server.  The narrow extension collates examples
into one policy call, supplies request-local noise, and optionally enables the
already-qualified batch-invariant operator implementations.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from batch_invariant_ops import set_batch_invariant_mode
from evaluation.LIBERO.policy_server.backends.policy_backend_internvla_a1_5 import (
    InternVLAA15Backend,
)
from evaluation.LIBERO.policy_server.tools.websocket_policy_server import WebsocketPolicyServer
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
CHECKPOINT_SHA256 = "222b6ed5dbd3224f441ad2f9165838a890aad5934282209fb635ca4daf073de0"
VLM_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def right_pad(value: torch.Tensor, length: int, fill: int | bool) -> torch.Tensor:
    if value.ndim != 1 or value.shape[0] > length:
        raise ValueError(f"invalid prompt tensor shape {tuple(value.shape)} for length {length}")
    return F.pad(value, (0, length - value.shape[0]), value=fill)


def collate_official_samples(
    samples: list[dict[str, Any]], *, canonical_prompt_length: int | None, pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Match the proven InternVLA adapter's stack layout with explicit prompt padding."""
    if not samples:
        raise ValueError("cannot collate an empty InternVLA request")
    prompt_keys = {
        f"{OBS_PREFIX}input_ids": pad_token_id,
        f"{OBS_PREFIX}attention_mask": 0,
        f"{OBS_PREFIX}fast_token_mask": False,
        # The official eval transform emits one label per prompt token and
        # fills the entire tensor with the training ignore index.  Prompt
        # lengths can differ across genuinely batched requests, so labels
        # must follow the same right-padding rule as the prompt tensors.
        "VQA.labels": -100,
    }
    observed_lengths = [int(sample[f"{OBS_PREFIX}input_ids"].shape[0]) for sample in samples]
    target_length = canonical_prompt_length or max(observed_lengths)
    if target_length < max(observed_lengths):
        raise ValueError(
            f"prompt exceeds canonical length: max={max(observed_lengths)} target={target_length}"
        )
    keys = {
        key for sample in samples for key, value in sample.items()
        if isinstance(value, torch.Tensor) and not isinstance(value, bool)
    }
    required = {
        f"{OBS_PREFIX}pixel_values", f"{OBS_PREFIX}image_grid_thw",
        f"{OBS_PREFIX}input_ids", f"{OBS_PREFIX}attention_mask",
        f"{OBS_PREFIX}fast_token_mask", OBS_STATE,
    }
    if not required.issubset(keys):
        raise KeyError(f"official InternVLA preprocessing omitted {sorted(required - keys)}")
    batch = {}
    for key in sorted(keys):
        values = [sample[key] for sample in samples]
        if key in prompt_keys:
            values = [right_pad(value, target_length, prompt_keys[key]) for value in values]
        try:
            batch[key] = torch.stack(values)
        except RuntimeError as error:
            raise RuntimeError(
                f"official InternVLA tensor {key} cannot be stacked: "
                f"{[tuple(value.shape) for value in values]}"
            ) from error
    return batch


def request_local_noise(
    seeds: list[int], shape: tuple[int, int], device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    values = []
    hashes = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        value = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        values.append(value)
        hashes.append(hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    return torch.stack(values), hashes


class PairedInternVLABackend:
    def __init__(self, official: InternVLAA15Backend, *, backend: str, condition: str) -> None:
        self.official = official
        self.backend = backend
        self.condition = condition
        self.max_batch_size, self.repaired = CONDITIONS[condition]
        tokenizer = official.processor.processor.tokenizer
        if tokenizer.pad_token_id is None:
            raise RuntimeError("pinned InternVLA tokenizer has no pad token ID")
        self.pad_token_id = int(tokenizer.pad_token_id)

    def metadata(self) -> dict[str, Any]:
        value = dict(self.official.metadata())
        value.update({
            "paired_condition": self.condition,
            "genuine_batched_policy_call": True,
            "maximum_request_batch": self.max_batch_size,
            "batch_invariant_ops": self.repaired,
            "request_local_noise": True,
            "prompt_padding": "fixed_right_650" if self.repaired else "longest_right",
            "optimized_graph_cache_policy": (
                "recapture_per_policy_call" if self.backend == "optimized" else None
            ),
        })
        return value

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("payload.examples must be a non-empty list")
        if len(examples) > self.max_batch_size:
            raise ValueError(
                f"condition {self.condition} permits at most {self.max_batch_size} examples, "
                f"received {len(examples)}"
            )
        noise_seeds = [example.get("noise_seed") for example in examples]
        if any(not isinstance(seed, int) or seed < 0 for seed in noise_seeds):
            raise ValueError("every paired example requires a nonnegative integer noise_seed")
        request_ids = [example.get("paired_request_id") for example in examples]
        if any(not isinstance(value, str) or not value for value in request_ids):
            raise ValueError("every paired example requires paired_request_id")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("paired_request_id values must be unique within a policy call")

        samples = [self.official._prepare_single(example) for example in examples]
        batch = collate_official_samples(
            samples,
            canonical_prompt_length=650 if self.repaired else None,
            pad_token_id=self.pad_token_id,
        )
        inputs = {key: value.to(self.official.device) for key, value in batch.items()}
        policy = self.official.policy
        core = policy.model
        state = policy.prepare_state(inputs)
        noise, noise_hashes = request_local_noise(
            noise_seeds,
            (policy.config.chunk_size, policy.config.max_action_dim),
            self.official.device,
        )

        # The pinned optimized graph cache is unsafe across changing B1/B4
        # request shapes.  The established adapter retains the optimized graph
        # kernels but captures once per complete call, which is reproduced here.
        if self.backend == "optimized":
            torch.cuda.synchronize()
            core._graphs.clear()
            core._static_buffers.clear()

        autocast_enabled = self.official.compute_dtype != torch.float32
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.official.device.type,
            dtype=self.official.compute_dtype,
            enabled=autocast_enabled,
        ):
            normalized = core.sample_actions(
                inputs[f"{OBS_PREFIX}pixel_values"],
                inputs[f"{OBS_PREFIX}image_grid_thw"],
                inputs[f"{OBS_PREFIX}input_ids"],
                inputs[f"{OBS_PREFIX}attention_mask"],
                state,
                fast_token_mask=inputs.get(f"{OBS_PREFIX}fast_token_mask"),
                noise=noise,
                num_steps=policy.config.num_inference_steps,
            )
        action_dim = policy.config.output_features[ACTION].shape[0]
        normalized_actions = normalized[..., :action_dim].detach().float().cpu().numpy()
        actions = self.official.postprocess_actions(
            self.official.denormalize_actions(normalized_actions)
        )
        return {
            "actions": actions,
            "action_space": self.official.action_space(),
            "action_dim": int(actions.shape[-1]),
            "chunk_size": int(policy.config.chunk_size),
            "request_ids": request_ids,
            "noise_sha256": noise_hashes,
            "physical_batch_size": len(examples),
            "postprocess": {"clip": True, "gripper_binarize": False},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("standard", "optimized"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    weight = args.checkpoint / "model.safetensors"
    vlm_weight = args.vlm / "model.safetensors-00001-of-00001.safetensors"
    if sha256(weight) != CHECKPOINT_SHA256 or sha256(vlm_weight) != VLM_SHA256:
        raise ValueError("InternVLA paired server inputs differ from pinned exact weights")

    official = InternVLAA15Backend(
        ckpt_path=str(args.checkpoint), device="cuda", stats_key="libero_10",
        robot_type="panda", resize_size=224, vlm_model_path=str(args.vlm),
        action_loss_only=True, inference_backend=args.backend,
    )
    backend = PairedInternVLABackend(official, backend=args.backend, condition=args.condition)
    server = WebsocketPolicyServer(
        policy=backend, host=args.host, port=args.port, idle_timeout=-1,
        metadata=backend.metadata(),
    )
    logging.info("paired InternVLA server running: backend=%s condition=%s", args.backend, args.condition)
    mode = set_batch_invariant_mode() if backend.repaired else contextlib.nullcontext()
    with mode:
        server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

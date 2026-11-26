"""Serve exact CogACT-Small actions for paired official SimplerEnv evaluation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle

from batch_invariant_ops import set_batch_invariant_mode


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
WEIGHT_SHA256 = "013cf0d4954e8a560064905ee03785ce3d0fa126697d4410cfde9a6754665e44"
WEIGHT_BYTES = 30_215_225_166
UNNORM_KEYS = {
    "google_robot": "fractal20220817_data",
    "widowx_bridge": "bridge_orig",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RequestState:
    """Exact released adaptive ensemble and sticky-gripper state per episode."""

    def __init__(self, policy_setup: str, adaptive_ensemble_class: type) -> None:
        self.policy_setup = policy_setup
        horizon = 7 if policy_setup == "widowx_bridge" else 2
        self.ensemble = adaptive_ensemble_class(horizon, 0.1)
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = np.array([0.0])
        self.previous_gripper_action: np.ndarray | None = None

    def process(self, raw_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(self.ensemble.ensemble_action(raw_actions))
        world = selected[:3]
        axes, angle = euler2axangle(*np.asarray(selected[3:6], dtype=np.float64))
        rotation = axes * angle
        current_gripper = selected[6:7]
        if self.policy_setup == "widowx_bridge":
            gripper = 2.0 * (current_gripper > 0.5) - 1.0
        else:
            if self.previous_gripper_action is None:
                relative_gripper = np.array([0])
                self.previous_gripper_action = current_gripper
            else:
                relative_gripper = self.previous_gripper_action - current_gripper
            if np.abs(relative_gripper) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper
                self.previous_gripper_action = current_gripper
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper = self.sticky_gripper_action
            if self.gripper_action_repeat == 10:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = np.array([0.0])
            gripper = relative_gripper
        return selected, np.concatenate([world, rotation, gripper]).astype(np.float64)


class PairedCogACTBackend:
    def __init__(
        self, adapter_module: Any, policy: Any, adaptive_ensemble_class: type,
        *, checkpoint: Path, condition: str,
    ) -> None:
        self.adapter_module = adapter_module
        self.adapter = adapter_module.adapter
        self.policy = policy
        self.adaptive_ensemble_class = adaptive_ensemble_class
        self.checkpoint = checkpoint
        self.condition = condition
        self.maximum_batch, self.repaired = CONDITIONS[condition]
        self.sessions: dict[str, RequestState] = {}

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": "2.1",
            "policy_type": "cogact_small",
            "checkpoint": str(self.checkpoint.resolve()),
            "checkpoint_weight_sha256": WEIGHT_SHA256,
            "task_suite": "simplerenv_cogact_semantic_12_variant",
            "paired_condition": self.condition,
            "genuine_batched_policy_call": True,
            "maximum_request_batch": self.maximum_batch,
            "batch_invariant_ops": self.repaired,
            "request_local_noise": True,
            "noise_shape": [16, 7],
            "cfg_scale": 1.5,
            "ddim_steps": 10,
            "unnorm_keys": UNNORM_KEYS,
            "action_chunk_size": 16,
            "action_dim": 7,
            "adaptive_ensemble_alpha": 0.1,
            "llama_construction_dtype": "torch.bfloat16",
            "llama_random_weight_initialization": "skipped_before_strict_checkpoint_load",
            "final_vlm_dtype": "torch.bfloat16",
            "final_action_head_dtype": str(next(self.policy.model.action_model.parameters()).dtype),
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= self.maximum_batch:
            raise ValueError(f"condition {self.condition} requires 1..{self.maximum_batch} examples")
        request_ids = [example.get("paired_request_id") for example in examples]
        if any(not isinstance(value, str) or not value for value in request_ids):
            raise ValueError("every CogACT example requires paired_request_id")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("paired_request_id values must be unique")
        policy_setups = [example.get("policy_setup") for example in examples]
        if len(set(policy_setups)) != 1 or policy_setups[0] not in UNNORM_KEYS:
            raise ValueError("one physical batch must use one supported CogACT policy setup")
        policy_setup = policy_setups[0]

        samples, noise_hashes = [], []
        for request_id, example in zip(request_ids, examples, strict=True):
            image = np.asarray(example.get("image"), dtype=np.uint8)
            instruction = example.get("task_description")
            # Msgpack may expose a read-only zero-copy array. Own the storage
            # before handing it to torch, even though inference does not mutate
            # request-local noise.
            noise = np.array(example.get("noise"), dtype=np.float32, copy=True)
            if (image.ndim != 3 or image.shape[-1] != 3
                    or not isinstance(instruction, str) or not instruction
                    or noise.shape != (16, 7) or not np.isfinite(noise).all()):
                raise ValueError("CogACT examples require HWC image, prompt, and finite [16,7] noise")
            if (example.get("reset") is True or request_id not in self.sessions
                    or self.sessions[request_id].policy_setup != policy_setup):
                self.sessions[request_id] = RequestState(
                    policy_setup, self.adaptive_ensemble_class,
                )
            samples.append({
                "image": Image.fromarray(image).convert("RGB"),
                "instruction": instruction,
                "noise": torch.from_numpy(noise).unsqueeze(0),
            })
            noise_hashes.append(hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest())

        batch = self.adapter.compose_batch(samples)
        # The released batch path hard-codes its offline example's statistics
        # key. Every physical evaluator batch has one robot setup, so select the
        # corresponding official key immediately around the synchronous call.
        self.adapter_module.UNNORM_KEY = UNNORM_KEYS[policy_setup]
        with torch.inference_mode():
            chunks = self.adapter.run_model(self.policy, batch).numpy()
        selected, actions = [], []
        for request_id, chunk in zip(request_ids, chunks, strict=True):
            chosen, action = self.sessions[request_id].process(chunk)
            selected.append(chosen)
            actions.append(action)
        return {
            "actions": np.stack(actions),
            "selected_raw_actions": np.stack(selected),
            "raw_action_chunks": chunks,
            "request_ids": request_ids,
            "noise_sha256": noise_hashes,
            "physical_batch_size": len(examples),
        }


def main() -> None:
    from evaluation.LIBERO.policy_server.tools.websocket_policy_server import WebsocketPolicyServer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    weight = args.checkpoint / "checkpoints/CogACT-Small.pt"
    if (not weight.is_file() or weight.stat().st_size != WEIGHT_BYTES
            or sha256(weight) != WEIGHT_SHA256):
        raise RuntimeError("CogACT-Small exact checkpoint differs")
    os.environ.update({
        "COGACT_CHECKPOINT": str(args.checkpoint.resolve()),
        "COGACT_SOURCE": "/workspace/models/cogact",
        "COGACT_OPENVLA_SOURCE": "/workspace/models/cogact-openvla",
        "COGACT_SAMPLE_DIR": "/workspace/assets/smolvla/fixture",
        "COGACT_LLAMA_METADATA": "/workspace/assets/cogact_llama2_metadata",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    from scripts.model_invariance.adapters import cogact_small as adapter_module
    from scripts.cogact_runtime import load_model_with_bf16_llama_construction

    policy = load_model_with_bf16_llama_construction(adapter_module.adapter)
    from sim_cogact.adaptive_ensemble import AdaptiveEnsembler

    backend = PairedCogACTBackend(
        adapter_module, policy, AdaptiveEnsembler,
        checkpoint=args.checkpoint, condition=args.condition,
    )
    server = WebsocketPolicyServer(
        policy=backend, host=args.host, port=args.port,
        idle_timeout=-1, metadata=backend.metadata(),
    )
    logging.info("paired CogACT server running: condition=%s", args.condition)
    mode = set_batch_invariant_mode() if backend.repaired else contextlib.nullcontext()
    with mode:
        server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

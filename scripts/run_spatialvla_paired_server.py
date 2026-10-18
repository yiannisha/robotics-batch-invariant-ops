"""Serve exact SpatialVLA actions for paired official SimplerEnv evaluation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
from collections import deque
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle
from transformers import AutoModel, AutoProcessor

from batch_invariant_ops import set_batch_invariant_mode
from evaluation.LIBERO.policy_server.tools.websocket_policy_server import WebsocketPolicyServer


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
CHECKPOINT_HASHES = {
    "model-00001-of-00002.safetensors": "713ee1bd116c8f77e2f56903d15e6c3f3dff6ea6b12fc71bfcc3bd4a53a2cc2b",
    "model-00002-of-00002.safetensors": "b8c87abc1b91598ad237c31ee2286c5c783bfb9f4142696d9b1c36e62f634a34",
    "config.json": "0c5d94be86f9e15b708650c4b829033ae99243f85507012b1dafa0263f799909",
    "model.safetensors.index.json": "6c351156136b2452db86d3a426ff30adc9d11363b43f46fa69fe79d76b6d5c79",
    "processor_config.json": "8712b6d5171f74e90f951ec7b148ab90509f28434e36b2da959527fa0ddb57ea",
    "processing_spatialvla.py": "c20b15460633f65809788c633936347fada2f2bbcfffba898d2f0b4b99807b33",
    "modeling_spatialvla.py": "a04008cf364e094079493e5982fa2b32fca12874b0a42b645e0aecab024e7236",
    "action_tokenizer.py": "b6b18eee0320e77156881aea4a92ea5f54dc278d43c1ea72c3a0deaa9d505a31",
    "configuration_spatialvla.py": "63e100324daf81d2cd7b6c5e6fafcace67d660586ef7b509bb38cc8fba72c1e3",
    "tokenizer.json": "2523a63c898ebf0a32c7282a2e459ef2c950a846c5f3172305089e4149b6b6c3",
    "tokenizer_config.json": "33f38858ac90afd5ac29942110dd3b98f40e1e194fca254714bab5d88f57cc65",
    "preprocessor_config.json": "bdb79dc3255a75c1c4be8ac1ddd4e2aa08b9a29eee0431755b7b0faa3b54e413",
    "generation_config.json": "79c657e6d0660218a91c360516cae5b8fa0cf1dd447476c8dcdb815c8ff994b1",
}
UNNORM_KEYS = {
    "google_robot": "fractal20220817_data/0.1.0",
    "widowx_bridge": "bridge_orig/1.0.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_artifacts(checkpoint: Path) -> list[dict[str, str]]:
    records = []
    for name, expected in CHECKPOINT_HASHES.items():
        path = checkpoint / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"SpatialVLA exact checkpoint artifact differs: {path}")
        records.append({"path": str(path.resolve()), "sha256": expected})
    return records


class RequestState:
    """Official per-episode action ensembling and sticky-gripper state."""

    def __init__(
        self, horizon: int, temperature: float = -0.8,
        policy_setup: str = "google_robot",
    ) -> None:
        if policy_setup not in UNNORM_KEYS:
            raise ValueError(f"unsupported SpatialVLA policy setup: {policy_setup}")
        self.horizon = horizon
        self.temperature = temperature
        self.policy_setup = policy_setup
        self.action_history: deque[np.ndarray] = deque(maxlen=horizon)
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = np.array([0.0])
        self.previous_gripper_action: np.ndarray | None = None

    def ensemble(self, current: np.ndarray) -> np.ndarray:
        self.action_history.append(np.asarray(current))
        count = len(self.action_history)
        if current.ndim == 1:
            predictions = np.stack(self.action_history)
        else:
            predictions = np.stack([
                prediction[index]
                for index, prediction in zip(range(count - 1, -1, -1), self.action_history)
            ])
        weights = np.exp(-self.temperature * np.arange(count))
        weights /= weights.sum()
        return np.sum(weights[:, None] * predictions, axis=0)

    def process(self, raw_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.ensemble(raw_actions)[None]
        selected = raw[0]
        world = np.asarray(selected[:3])
        axis, angle = euler2axangle(*np.asarray(selected[3:6], dtype=np.float64))
        rotation = axis * angle
        current_gripper = np.asarray(selected[6:7])
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
        action = np.concatenate([world, rotation, gripper]).astype(np.float64)
        return selected, action


class PairedSpatialVLABackend:
    def __init__(
        self, model: Any, processor: Any, *, condition: str,
        checkpoint: Path, artifact_files: list[dict[str, str]],
    ) -> None:
        self.model = model
        self.processor = processor
        self.condition = condition
        self.maximum_batch, self.repaired = CONDITIONS[condition]
        self.checkpoint = checkpoint
        self.artifact_files = artifact_files
        self.sessions: dict[str, RequestState] = {}
        if processor.num_obs_steps != 1 or processor.obs_delta != 1:
            raise RuntimeError("SpatialVLA checkpoint observation-history contract changed")
        if processor.action_chunk_size != 4:
            raise RuntimeError("SpatialVLA checkpoint action-chunk contract changed")

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": "2.1",
            "policy_type": "spatialvla",
            "checkpoint": str(self.checkpoint.resolve()),
            "task_suite": "simplerenv_spatialvla_semantic_12_variant",
            "paired_condition": self.condition,
            "genuine_batched_policy_call": True,
            "maximum_request_batch": self.maximum_batch,
            "batch_invariant_ops": self.repaired,
            "deterministic_policy": True,
            "unnorm_keys": UNNORM_KEYS,
            "image_size": [224, 224],
            "action_chunk_size": 4,
            "action_dim": 7,
            "action_ensemble_temperature": -0.8,
            "sticky_gripper_repeat": 10,
            "artifact_files": self.artifact_files,
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= self.maximum_batch:
            raise ValueError(f"condition {self.condition} requires 1..{self.maximum_batch} examples")
        request_ids = [example.get("paired_request_id") for example in examples]
        if any(not isinstance(value, str) or not value for value in request_ids):
            raise ValueError("every SpatialVLA example requires paired_request_id")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("paired_request_id values must be unique within a physical batch")
        images, prompts = [], []
        policy_setups = [example.get("policy_setup") for example in examples]
        if len(set(policy_setups)) != 1 or policy_setups[0] not in UNNORM_KEYS:
            raise ValueError("one physical batch must use one supported SpatialVLA policy setup")
        policy_setup = policy_setups[0]
        unnorm_key = UNNORM_KEYS[policy_setup]
        for request_id, example in zip(request_ids, examples):
            image = np.asarray(example.get("image"), dtype=np.uint8)
            prompt = example.get("task_description")
            if image.ndim != 3 or image.shape[-1] != 3 or not isinstance(prompt, str) or not prompt:
                raise ValueError("SpatialVLA examples require one HWC uint8 image and prompt")
            if (example.get("reset") is True or request_id not in self.sessions
                    or self.sessions[request_id].policy_setup != policy_setup):
                self.sessions[request_id] = RequestState(
                    self.processor.action_chunk_size, policy_setup=policy_setup,
                )
            images.append(Image.fromarray(cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)).convert("RGB"))
            prompts.append(prompt)
        inputs = self.processor(
            images=images, text=prompts, unnorm_key=unnorm_key,
            return_tensors="pt", do_normalize=False, padding=True,
        ).to(device="cuda", dtype=torch.bfloat16)
        input_length = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
            )[:, input_length:]
        raw_chunks, selected, actions = [], [], []
        for request_id, tokens in zip(request_ids, generated):
            decoded = self.processor.decode_actions(
                tokens.unsqueeze(0), unnorm_key=unnorm_key,
            )["actions"]
            decoded = np.asarray(decoded).copy()
            raw_chunks.append(decoded)
            chosen, action = self.sessions[request_id].process(decoded)
            selected.append(chosen)
            actions.append(action)
        return {
            "actions": np.stack(actions),
            "selected_raw_actions": np.stack(selected),
            "raw_action_chunks": np.stack(raw_chunks),
            "request_ids": request_ids,
            "physical_batch_size": len(examples),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    artifacts = exact_artifacts(args.checkpoint)
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval().cuda()
    backend = PairedSpatialVLABackend(
        model, processor, condition=args.condition,
        checkpoint=args.checkpoint, artifact_files=artifacts,
    )
    server = WebsocketPolicyServer(
        policy=backend, host=args.host, port=args.port,
        idle_timeout=-1, metadata=backend.metadata(),
    )
    logging.info("paired SpatialVLA server running: condition=%s", args.condition)
    mode = set_batch_invariant_mode() if backend.repaired else contextlib.nullcontext()
    with mode:
        server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

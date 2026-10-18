"""Serve exact WSA-Base-LIBERO policy calls for paired LIBERO-10 evaluation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from batch_invariant_ops import set_batch_invariant_mode
from evaluation.LIBERO.policy_server.tools.websocket_policy_server import WebsocketPolicyServer
from scripts.model_invariance.adapters.wsa_base_libero import WSABaseLiberoAdapter


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
EXACT_HASHES = {
    "checkpoint/model.safetensors": "603eacbb1a7da80dd2d41cbbd814a833bd57dec30601da5fda35651d6dd46e67",
    "checkpoint/config.json": "e0f3e9043e97bf84569c8f430a3e666417f00d454286e83289ef010e5e507b7e",
    "checkpoint/stats.json": "5f6b2728c416bf63908a802aef8c96885d31c6fbdd7e625ec1ca84d306361f3a",
    "checkpoint/train_config.json": "16d21b3fd4fa49601cf3bb31c8176f42b8dc14834ceb04d4d84dad2ffa48e548",
    "cosmos_tokenizer/encoder.jit": "84e4c12c10568faff20da744b51c45a6b1e4385af86e80ae56bf3d1b13cda413",
    "cosmos_tokenizer/decoder.jit": "8e50c647a04decc7e939670ef7b51bc0500e0ffb4836a306a93100b0b2e957de",
    "cosmos_tokenizer/config.json": "ebf911575616266497325c38329b4212b2dfe223a6ad6c9b578fcbd7f7552aa3",
    "cosmos_tokenizer/model_config.yaml": "69bc6fabb030be2ddba76c0a6944882348c3db0426baf31ced29ca02aeaba679",
}
QWEN_HASHES = {
    "config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_files(asset_root: Path, qwen: Path) -> list[dict[str, str]]:
    records = []
    for name, expected in EXACT_HASHES.items():
        path = asset_root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"WSA exact artifact differs: {path}")
        records.append({"path": str(path.resolve()), "sha256": expected})
    for name, expected in QWEN_HASHES.items():
        path = qwen / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"WSA Qwen processor metadata differs: {path}")
        records.append({"path": str(path.resolve()), "sha256": expected})
    return records


class PairedWSABackend:
    def __init__(
        self, adapter: WSABaseLiberoAdapter, policy: Any, *, condition: str,
        checkpoint: Path, task_descriptions: list[str], artifact_files: list[dict[str, str]],
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.condition = condition
        self.maximum_batch, self.repaired = CONDITIONS[condition]
        self.checkpoint = checkpoint
        self.task_descriptions = task_descriptions
        self.artifact_files = artifact_files

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": "2.1",
            "policy_type": "wsa_base",
            "checkpoint": str(self.checkpoint.resolve()),
            "task_suite": "libero_10",
            "paired_condition": self.condition,
            "genuine_batched_policy_call": True,
            "maximum_request_batch": self.maximum_batch,
            "batch_invariant_ops": self.repaired,
            "request_local_noise": True,
            "flow_steps": 10,
            "noise_shape": [10, 32],
            "image_history_interval": 15,
            "num_images_in_input": 3,
            "third_image_masked": True,
            "action_chunk_size": 10,
            "action_dim": 7,
            "control_mode": "absolute",
            "gripper_mode": "libero_open_prob",
            "preprocessing_owner": "server_official_policy_adapter",
            "artifact_files": self.artifact_files,
        }

    def _sample(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        task = example.get("task")
        if task not in self.task_descriptions:
            raise ValueError(f"task is outside pinned LIBERO-10 manifest: {task!r}")
        head = np.asarray(example.get("head_history"), dtype=np.uint8)
        wrist = np.asarray(example.get("wrist_history"), dtype=np.uint8)
        state = np.asarray(example.get("state"), dtype=np.float32).reshape(-1)
        noise = np.asarray(example.get("noise"), dtype=np.float32)
        if head.ndim != 4 or wrist.shape != head.shape or head.shape[0] != 2 or head.shape[-1] != 3:
            raise ValueError("WSA example requires two HWC frames for head and wrist")
        if state.shape != (8,) or noise.shape != (10, 32):
            raise ValueError("WSA example requires 8D state and [10,32] request-local noise")
        head_tensor = torch.from_numpy(np.ascontiguousarray(head)).permute(0, 3, 1, 2).float().div(255)
        wrist_tensor = torch.from_numpy(np.ascontiguousarray(wrist)).permute(0, 3, 1, 2).float().div(255)
        sample = {
            f"{OBS_IMAGES}.image0": head_tensor,
            f"{OBS_IMAGES}.image1": wrist_tensor,
            f"{OBS_IMAGES}.image2": torch.ones_like(head_tensor),
            OBS_STATE: torch.from_numpy(state),
            "task": task,
        }
        sample = self.adapter.resize(sample)
        sample[f"{OBS_IMAGES}.image0_mask"] = torch.tensor(True)
        sample[f"{OBS_IMAGES}.image1_mask"] = torch.tensor(True)
        sample[f"{OBS_IMAGES}.image2_mask"] = torch.tensor(False)
        sample = self.adapter.processor(sample)
        mean = torch.from_numpy(self.adapter.state_stats["mean"])
        std = torch.from_numpy(self.adapter.state_stats["std"])
        sample[OBS_STATE] = (sample[OBS_STATE] - mean) / (std + 1e-6)
        return {
            **{key: value.detach().cpu() for key, value in sample.items() if isinstance(value, torch.Tensor)},
            "noise": torch.from_numpy(noise.copy()),
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= self.maximum_batch:
            raise ValueError(f"condition {self.condition} requires 1..{self.maximum_batch} examples")
        request_ids = [example.get("paired_request_id") for example in examples]
        if any(not isinstance(value, str) or not value for value in request_ids):
            raise ValueError("every WSA example requires paired_request_id")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("paired_request_id values must be unique")
        batch = self.adapter.compose_batch([self._sample(example) for example in examples])
        with torch.no_grad():
            actions = self.adapter.run_official_model(self.policy, batch).detach().float().cpu().numpy()
        return {
            "actions": actions,
            "request_ids": request_ids,
            "physical_batch_size": len(examples),
            "action_chunk_size": 10,
            "action_dim": 7,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--qwen-metadata", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    artifact_files = exact_files(args.asset_root, args.qwen_metadata)
    manifest = json.loads(args.task_manifest.read_text())
    descriptions = manifest.get("task_descriptions")
    if (manifest.get("task_suite") != "libero_10" or manifest.get("task_ids") != list(range(10))
            or not isinstance(descriptions, list) or len(descriptions) != 10):
        raise ValueError("invalid pinned WSA LIBERO-10 task manifest")

    os.environ.update({
        "WSA_SOURCE": "/workspace/models/wsa",
        "WSA_CHECKPOINT": str(args.asset_root / "checkpoint"),
        "WSA_QWEN_METADATA": str(args.qwen_metadata),
        "WSA_COSMOS_TOKENIZER": str(args.asset_root / "cosmos_tokenizer"),
        "WSA_SAMPLE_DIR": "/workspace/assets/smolvla/full_replay",
        "WSA_REPLAY_MANIFEST": str(
            Path("/workspace/robotics-batch-invariant-ops/results/cross_model/wsa_base/full/replay_manifest.json")
        ),
    })
    adapter = WSABaseLiberoAdapter()
    policy = adapter.load_model()
    backend = PairedWSABackend(
        adapter, policy, condition=args.condition, checkpoint=args.asset_root / "checkpoint",
        task_descriptions=descriptions, artifact_files=artifact_files,
    )
    server = WebsocketPolicyServer(
        policy=backend, host=args.host, port=args.port, idle_timeout=-1,
        metadata=backend.metadata(),
    )
    logging.info("paired WSA server running: condition=%s", args.condition)
    mode = set_batch_invariant_mode() if backend.repaired else contextlib.nullcontext()
    with mode:
        server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

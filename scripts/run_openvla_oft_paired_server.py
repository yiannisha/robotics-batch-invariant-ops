"""Serve exact OpenVLA-OFT LIBERO-Spatial policy calls for paired evaluation."""

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
from PIL import Image

from batch_invariant_ops import set_batch_invariant_mode
from evaluation.LIBERO.policy_server.tools.websocket_policy_server import WebsocketPolicyServer
from scripts.model_invariance.adapters.openvla_oft import OpenVLAOFTAdapter


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
EXACT_HASHES = {
    "model-00001-of-00004.safetensors": "2809bd7be9422315c5ecbe91eea612f5b02925f37e0051728ad45bc993c79251",
    "model-00002-of-00004.safetensors": "a00a7c5f2b6586ccfc89c693a9c36f3552ff455ff2b1bfea3e92642e4cd2b6d3",
    "model-00003-of-00004.safetensors": "a894b7230a08b471af55c57dd7385fd3b51fae2c4d9307a36a8017ef57abf22a",
    "model-00004-of-00004.safetensors": "a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60",
    "action_head--150000_checkpoint.pt": "809858636cf0a65009dd567d2f4e116249442790f02b8fe31f24500ea6118908",
    "proprio_projector--150000_checkpoint.pt": "438d28e81e125166d0771762424aaf017de3a8daaebde06a5cb71157e62b3bf3",
    "config.json": "bcb688a66f3e94a42311b77d37b7e1484886701f7b2aa8e36d165dfccac92ae6",
    "dataset_statistics.json": "3f8b7854c8634ea6ccfd6ef6b8f983e3b30b067c2cdaf690b152c0899d5d9282",
}
DETERMINISTIC_CONTRACT = {
    "inference_head": "l1_regression",
    "stochastic_request_inputs": "none",
    "do_sample": False,
}
DETERMINISTIC_SHA256 = hashlib.sha256(
    json.dumps(DETERMINISTIC_CONTRACT, sort_keys=True).encode()
).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(checkpoint: Path) -> list[dict[str, str]]:
    records = []
    for name, expected in EXACT_HASHES.items():
        path = checkpoint / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"OpenVLA-OFT exact checkpoint file differs: {path}")
        records.append({"path": str(path.resolve()), "sha256": expected})
    return records


class PairedOpenVLABackend:
    def __init__(
        self, adapter: OpenVLAOFTAdapter, policy: Any, *, checkpoint: Path,
        condition: str, task_descriptions: list[str], checkpoint_files: list[dict[str, str]],
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.checkpoint = checkpoint
        self.condition = condition
        self.maximum_batch, self.repaired = CONDITIONS[condition]
        self.task_descriptions = task_descriptions
        self.checkpoint_files = checkpoint_files
        self.adapter.canonical_prompt_length = max(
            len(self.adapter.processor.tokenizer(
                f"In: What action should the robot take to {task.lower()}?\nOut:"
            )["input_ids"])
            for task in task_descriptions
        )
        if self.adapter.processor.tokenizer.pad_token_id is None:
            raise RuntimeError("OpenVLA-OFT tokenizer lacks pad_token_id")

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": "2.1",
            "policy_type": "openvla_oft",
            "checkpoint": str(self.checkpoint.resolve()),
            "task_suite": "libero_spatial",
            "paired_condition": self.condition,
            "genuine_batched_policy_call": True,
            "maximum_request_batch": self.maximum_batch,
            "batch_invariant_ops": self.repaired,
            "request_local_noise": False,
            "deterministic_contract": DETERMINISTIC_CONTRACT,
            "deterministic_contract_sha256": DETERMINISTIC_SHA256,
            "prompt_padding": "canonical_right" if self.repaired else "longest_in_batch_right",
            "canonical_prompt_length": self.adapter.canonical_prompt_length,
            "preprocessing_owner": "server_official_policy_adapter",
            "center_crop_area": 0.9,
            "num_images_in_input": 2,
            "use_proprio": True,
            "action_chunk_size": 8,
            "action_dim": 7,
            "checkpoint_files": self.checkpoint_files,
        }

    def _sample(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        task = example.get("task")
        if task not in self.task_descriptions:
            raise ValueError(f"task is outside pinned LIBERO-Spatial manifest: {task!r}")
        images = example.get("image")
        state = np.asarray(example.get("state"), dtype=np.float32).reshape(-1)
        if not isinstance(images, list) or len(images) != 2 or state.shape != (8,):
            raise ValueError("OpenVLA-OFT example requires two images and 8D state")
        prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
        processed = []
        for raw in images:
            array = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)[::-1, ::-1])
            image = self.adapter._center_crop(Image.fromarray(array, mode="RGB"))
            processed.append(self.adapter.processor(prompt, image))
        if not torch.equal(processed[0]["input_ids"], processed[1]["input_ids"]):
            raise RuntimeError("OpenVLA primary/wrist prompt tokenization differs")
        stats = self.adapter.stats["proprio"]
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        normalized_state = np.clip(2 * (state - low) / (high - low + 1e-8) - 1, -1, 1)
        return {
            "input_ids": processed[0]["input_ids"],
            "attention_mask": processed[0]["attention_mask"],
            "pixel_values": torch.cat(
                [processed[0]["pixel_values"], processed[1]["pixel_values"]], dim=1
            ),
            "proprio": torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0),
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= self.maximum_batch:
            raise ValueError(f"condition {self.condition} requires 1..{self.maximum_batch} examples")
        request_ids = [example.get("paired_request_id") for example in examples]
        if any(not isinstance(value, str) or not value for value in request_ids):
            raise ValueError("every example requires paired_request_id")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("paired_request_id values must be unique")
        samples = [self._sample(example) for example in examples]
        batch = self.adapter.compose_batch(samples)
        with torch.no_grad():
            actions = self.adapter.run_model(self.policy, batch).detach().float().cpu().numpy()
        return {
            "actions": actions,
            "request_ids": request_ids,
            "physical_batch_size": len(examples),
            "deterministic_contract_sha256": DETERMINISTIC_SHA256,
            "action_chunk_size": 8,
            "action_dim": 7,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    checkpoint_files = verify_checkpoint(args.checkpoint)
    manifest = json.loads(args.task_manifest.read_text())
    task_descriptions = manifest.get("task_descriptions")
    if (manifest.get("task_suite") != "libero_spatial" or manifest.get("task_ids") != list(range(10))
            or not isinstance(task_descriptions, list) or len(task_descriptions) != 10):
        raise ValueError("invalid pinned OpenVLA-OFT LIBERO-Spatial task manifest")

    os.environ.update({
        "OPENVLA_OFT_CHECKPOINT": str(args.checkpoint),
        "OPENVLA_OFT_SAMPLE_DIR": "/workspace/assets/smolvla/full_replay",
        "OPENVLA_OFT_REPLAY_MANIFEST": str(
            Path("/workspace/robotics-batch-invariant-ops/results/cross_model/openvla_oft/full/replay_manifest.json")
        ),
    })
    adapter = OpenVLAOFTAdapter()
    policy = adapter.load_model()
    backend = PairedOpenVLABackend(
        adapter, policy, checkpoint=args.checkpoint, condition=args.condition,
        task_descriptions=task_descriptions, checkpoint_files=checkpoint_files,
    )
    server = WebsocketPolicyServer(
        policy=backend, host=args.host, port=args.port, idle_timeout=-1,
        metadata=backend.metadata(),
    )
    logging.info("paired OpenVLA-OFT server running: condition=%s", args.condition)
    mode = set_batch_invariant_mode() if backend.repaired else contextlib.nullcontext()
    with mode:
        server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

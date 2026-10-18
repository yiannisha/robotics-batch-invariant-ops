"""Serve exact RDT-1B ManiSkill policy calls for paired five-task evaluation."""

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
from scripts.model_invariance.adapters.rdt_maniskill import RDTManiSkillAdapter


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
TASKS = (
    "PegInsertionSide-v1", "PickCube-v1", "StackCube-v1", "PlugCharger-v1", "PushCube-v1",
)
STATE_MIN = np.asarray((
    -0.7463043928146362, -0.0801204964518547, -0.4976441562175751,
    -2.657780647277832, -0.5742632150650024, 1.8309762477874756,
    -2.2423808574676514, 0.0,
), dtype=np.float32)
STATE_MAX = np.asarray((
    0.7645499110221863, 1.4967026710510254, 0.4650936424732208,
    -0.3866899907588959, 0.5505855679512024, 3.2900545597076416,
    2.5737812519073486, 0.03999999910593033,
), dtype=np.float32)
MANISKILL_INDICES = (0, 1, 2, 3, 4, 5, 6, 10)
EXACT_HASHES = {
    "rdt/mp_rank_00_model_states.pt": "3fa89cbf7bbb1b2559b29f9ae153b431541cdaa0b3f1ee44b83b9934599b9fb8",
    "lang_embeds/text_embed_PegInsertionSide-v1.pt": "da798aab9df67a230259095c0d087772d10ff2dd089bd71762a1b6d1696dbbf3",
    "lang_embeds/text_embed_PickCube-v1.pt": "a9ac30bc88ef12b171e9323b8a2c64f0b0ae9a54715deb061c021deaadb37f7a",
    "lang_embeds/text_embed_PlugCharger-v1.pt": "c38f80c549ec1d48f74ffdeafff8dd9f02ca64de8c3bddd7fe3cca173925922d",
    "lang_embeds/text_embed_PushCube-v1.pt": "536272882840dd1d2aef759b7551da1515b9c0cbe45e491a025169efadf51471",
    "lang_embeds/text_embed_StackCube-v1.pt": "ca0a1f758d183e4c541b412efd2ffafacae7bdc7220d03f7e9dca3e93bb28985",
}
SIGLIP_HASHES = {
    "model.safetensors": "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a",
    "config.json": "adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b",
    "preprocessor_config.json": "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(checkpoint: Path, siglip: Path) -> list[dict[str, str]]:
    records = []
    for root, hashes in ((checkpoint, EXACT_HASHES), (siglip, SIGLIP_HASHES)):
        for name, expected in hashes.items():
            path = root / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"RDT exact artifact differs: {path}")
            records.append({"path": str(path.resolve()), "sha256": expected})
    return records


class PairedRDTBackend:
    def __init__(
        self, adapter: RDTManiSkillAdapter, policy: Any, *, condition: str,
        checkpoint: Path, language: dict[str, torch.Tensor], artifacts: list[dict[str, str]],
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.condition = condition
        self.maximum_batch, self.repaired = CONDITIONS[condition]
        self.checkpoint = checkpoint
        self.language = language
        self.artifacts = artifacts
        self.canonical_language_tokens = max(value.shape[1] for value in language.values())

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": "2.1", "policy_type": "rdt_1b",
            "checkpoint": str(self.checkpoint.resolve()), "task_suite": "maniskill_five",
            "paired_condition": self.condition, "genuine_batched_policy_call": True,
            "maximum_request_batch": self.maximum_batch,
            "batch_invariant_ops": self.repaired, "request_local_noise": True,
            "diffusion_steps": 5, "noise_shape": [64, 128],
            "language_padding": "fixed_right_30" if self.repaired else "longest_in_batch_right",
            "canonical_language_tokens": self.canonical_language_tokens,
            "image_history": 2, "num_cameras": 3, "action_chunk_size": 64,
            "executed_action_stride": 4, "action_dim": 8,
            "control_mode": "pd_joint_pos", "control_frequency": 25,
            "preprocessing_owner": "server_official_maniskill_adapter",
            "artifact_files": self.artifacts,
        }

    def _image_tokens(self, examples: list[dict[str, Any]]) -> torch.Tensor:
        processor = self.policy.processor
        color = tuple(int(value * 255) for value in processor.image_mean)
        size = processor.size
        background = Image.fromarray(
            np.ones((size["height"], size["width"], 3), dtype=np.uint8)
            * np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
        )
        tensors = []
        for example in examples:
            history = np.asarray(example.get("image_history"), dtype=np.uint8)
            valid = example.get("history_valid")
            if history.ndim != 4 or history.shape[0] != 2 or history.shape[-1] != 3:
                raise ValueError("RDT example requires two HWC history images")
            if not isinstance(valid, list) or len(valid) != 2 or any(type(value) is not bool for value in valid):
                raise ValueError("RDT example requires two history validity flags")
            slots = [Image.fromarray(history[0]) if valid[0] else background, background, background,
                     Image.fromarray(history[1]) if valid[1] else background, background, background]
            for image in slots:
                square = self.adapter._expand_to_square(image, color)
                tensors.append(processor.preprocess(square, return_tensors="pt")["pixel_values"][0])
        pixels = torch.stack(tensors).to(self.adapter.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            tokens = self.policy.vision(pixels).detach()
        batch = len(examples)
        return tokens.reshape(batch, -1, self.policy.vision.hidden_size).cpu()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= self.maximum_batch:
            raise ValueError(f"condition {self.condition} requires 1..{self.maximum_batch} examples")
        request_ids = [example.get("paired_request_id") for example in examples]
        if any(not isinstance(value, str) or not value for value in request_ids) or len(set(request_ids)) != len(request_ids):
            raise ValueError("RDT paired_request_id values must be present and unique")
        task_names = [example.get("task_name") for example in examples]
        if any(name not in self.language for name in task_names):
            raise ValueError(f"RDT request has task outside pinned suite: {task_names}")
        token_count = self.canonical_language_tokens if self.repaired else max(
            self.language[name].shape[1] for name in task_names
        )
        language = torch.zeros(len(examples), token_count, 4096, dtype=torch.bfloat16)
        language_mask = torch.zeros(len(examples), token_count, dtype=torch.bool)
        state = torch.zeros(len(examples), 1, 128, dtype=torch.bfloat16)
        action_mask = torch.zeros(len(examples), 1, 128, dtype=torch.bfloat16)
        noise = torch.empty(len(examples), 64, 128, dtype=torch.bfloat16)
        for index, (example, task_name) in enumerate(zip(examples, task_names)):
            embedding = self.language[task_name][0]
            language[index, :embedding.shape[0]] = embedding
            language_mask[index, :embedding.shape[0]] = True
            joints = np.asarray(example.get("proprio"), dtype=np.float32).reshape(-1)
            raw_noise = np.array(example.get("noise"), dtype=np.float32, copy=True)
            if joints.shape != (8,) or raw_noise.shape != (64, 128):
                raise ValueError("RDT example requires 8D proprio and [64,128] noise")
            normalized = (joints - STATE_MIN) / (STATE_MAX - STATE_MIN) * 2 - 1
            state[index, 0, list(MANISKILL_INDICES)] = torch.from_numpy(normalized).to(torch.bfloat16)
            action_mask[index, 0, list(MANISKILL_INDICES)] = 1
            noise[index] = torch.from_numpy(raw_noise).to(torch.bfloat16)
        batch = {
            "language": language, "language_mask": language_mask,
            "image_tokens": self._image_tokens(examples), "state": state,
            "action_mask": action_mask,
            "control_frequency": torch.full((len(examples),), 25, dtype=torch.long),
            "noise": noise,
        }
        with torch.no_grad():
            actions = self.adapter.run_model(self.policy, batch).float().numpy()
        return {
            "actions": actions, "request_ids": request_ids,
            "physical_batch_size": len(examples), "action_chunk_size": 64, "action_dim": 8,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--siglip", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    artifacts = verify_artifacts(args.checkpoint, args.siglip)
    manifest = json.loads(args.task_manifest.read_text())
    if (manifest.get("task_suite") != "maniskill_five"
            or manifest.get("task_names") != list(TASKS) or manifest.get("task_ids") != list(range(5))):
        raise ValueError("invalid pinned RDT ManiSkill task manifest")
    os.environ.update({
        "RDT_CHECKPOINT": str(args.checkpoint), "RDT_SIGLIP": str(args.siglip),
        "RDT_SOURCE": "/workspace/models/rdt",
        "RDT_SAMPLE_DIR": "/workspace/assets/lerobot_libero/fixture",
    })
    adapter = RDTManiSkillAdapter()
    policy = adapter.load_model()
    language = {
        task: torch.load(
            args.checkpoint / "lang_embeds" / f"text_embed_{task}.pt",
            map_location="cpu", weights_only=True,
        ).to(torch.bfloat16)
        for task in TASKS
    }
    backend = PairedRDTBackend(
        adapter, policy, condition=args.condition, checkpoint=args.checkpoint,
        language=language, artifacts=artifacts,
    )
    server = WebsocketPolicyServer(
        policy=backend, host=args.host, port=args.port, idle_timeout=-1,
        metadata=backend.metadata(),
    )
    logging.info("paired RDT server running: condition=%s", args.condition)
    mode = set_batch_invariant_mode() if backend.repaired else contextlib.nullcontext()
    with mode:
        server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

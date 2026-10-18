"""Serve exact GR00T N1.7 calls for paired LIBERO-10 evaluation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from batch_invariant_ops import set_batch_invariant_mode
from groot_runtime import load_official_libero_policy
from scripts.model_invariance.adapters.groot_n17 import ACTION_KEYS, GrootN17Adapter


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
CHECKPOINT_HASHES = {
    "model-00001-of-00002.safetensors": "39b7beaadd9a06c87502f2b741a36f939b84941f37fef757c6177e54e34a5eff",
    "model-00002-of-00002.safetensors": "8c4fa56f1b4f25a1a842c811e8a12d2bf6d77d73942f9a04f5b876a11bd6d703",
    "model.safetensors.index.json": "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
    "config.json": "ce763f74ad1cda2ca9e67491229cbc5810d83cae675cb4dad839a69a9731fd97",
    "processor_config.json": "545bb44c7d8caaf5ac82df55b90831271a1469a4e48adcc9a5f469f064f4e121",
    "statistics.json": "cbd3765acaec31fb65109304fa0c2ebd637732ddbe1b3cab0cc11bf244a498e1",
}
METADATA_HASHES = {
    "config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
NOISE_SHAPE = (40, 132)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_files(checkpoint: Path, metadata: Path) -> list[dict[str, str]]:
    records = []
    for root, expected_files in ((checkpoint, CHECKPOINT_HASHES), (metadata, METADATA_HASHES)):
        for name, expected in expected_files.items():
            path = root / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"GR00T exact artifact differs: {path}")
            records.append({"path": str(path.resolve()), "sha256": expected})
    return records


class PairedGrootPolicy:
    """Official processor/model path with request-local flow noise."""

    def __init__(
        self,
        policy: Any,
        *,
        condition: str,
        checkpoint: Path,
        task_descriptions: list[str],
        artifact_files: list[dict[str, str]],
    ) -> None:
        self.policy = policy
        self.condition = condition
        self.maximum_batch, self.repaired = CONDITIONS[condition]
        self.checkpoint = checkpoint
        self.task_descriptions = task_descriptions
        self.artifact_files = artifact_files
        self.adapter = GrootN17Adapter()
        self.adapter.processor = policy.processor
        self.adapter.collate_fn = policy.collate_fn
        self.adapter.embodiment = policy.embodiment_tag

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": "1.0",
            "policy_type": "groot_n17",
            "checkpoint": str(self.checkpoint.resolve()),
            "task_suite": "libero_10",
            "paired_condition": self.condition,
            "maximum_request_batch": self.maximum_batch,
            "batch_invariant_ops": self.repaired,
            "genuine_batched_policy_call": True,
            "request_local_noise": True,
            "noise_shape": list(NOISE_SHAPE),
            "flow_steps": 4,
            "action_horizon": 16,
            "executed_action_steps": 8,
            "action_dim": 7,
            "processor": "official_Gr00tN1d7Processor",
            "environment_action_transform": "official_LiberoEnv.step",
            "artifact_files": self.artifact_files,
        }

    def _sample(self, example: dict[str, Any]) -> dict[str, Any]:
        from gr00t.data.types import MessageType, VLAStepData

        task = example.get("task")
        if task not in self.task_descriptions:
            raise ValueError(f"task is outside pinned LIBERO-10 manifest: {task!r}")
        image = np.asarray(example.get("image"), dtype=np.uint8)
        wrist = np.asarray(example.get("wrist_image"), dtype=np.uint8)
        state = np.asarray(example.get("state"), dtype=np.float32).reshape(-1)
        noise = np.asarray(example.get("noise"), dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 3 or wrist.shape != image.shape:
            raise ValueError("GR00T example requires equal HWC RGB image and wrist_image")
        if state.shape != (8,) or noise.shape != NOISE_SHAPE:
            raise ValueError("GR00T example requires 8D state and [40,132] request-local noise")
        states = {
            "x": state[0:1][None], "y": state[1:2][None], "z": state[2:3][None],
            "roll": state[3:4][None], "pitch": state[4:5][None],
            "yaw": state[5:6][None], "gripper": state[6:8][None],
        }
        step = VLAStepData(
            images={"image": image[None], "wrist_image": wrist[None]},
            states=states,
            actions={},
            text=task,
            embodiment=self.policy.embodiment_tag,
        )
        processed = self.policy.processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": step}]
        )
        return {"processed": processed, "states": states, "noise": torch.from_numpy(noise.copy())}

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        examples = observation.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= self.maximum_batch:
            raise ValueError(
                f"condition {self.condition} requires 1..{self.maximum_batch} examples"
            )
        request_ids = [example.get("paired_request_id") for example in examples]
        if any(not isinstance(value, str) or not value for value in request_ids):
            raise ValueError("every GR00T example requires paired_request_id")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("paired_request_id values must be unique")
        batch = self.adapter.compose_batch([self._sample(example) for example in examples])
        with torch.inference_mode():
            normalized = self.adapter._get_action_with_noise(
                self.policy.model, batch["inputs"], batch["noise"]
            )
        decoded = self.policy.processor.decode_action(
            normalized.float().cpu().numpy(), self.policy.embodiment_tag, batch["states"]
        )
        actions = np.concatenate([decoded[key] for key in ACTION_KEYS], axis=-1).astype(np.float32)
        if actions.shape != (len(examples), 16, 7):
            raise RuntimeError(f"unexpected GR00T action shape: {actions.shape}")
        return {
            "actions": actions,
            "request_ids": request_ids,
            "physical_batch_size": len(examples),
        }, {}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del options
        return {}

    def get_modality_config(self) -> dict[str, Any]:
        return self.metadata()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/workspace/assets/groot_n17/repo/libero_10"),
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("/workspace/assets/groot_n17/processor"),
    )
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    artifacts = exact_files(args.checkpoint, args.metadata)
    manifest = json.loads(args.task_manifest.read_text())
    descriptions = manifest.get("task_descriptions")
    if (
        manifest.get("task_suite") != "libero_10"
        or manifest.get("task_ids") != list(range(10))
        or not isinstance(descriptions, list)
        or len(descriptions) != 10
        or manifest.get("artifact_files") != artifacts
    ):
        raise ValueError("invalid pinned GR00T LIBERO-10 task/checkpoint manifest")

    from gr00t.policy.server_client import PolicyServer

    policy = load_official_libero_policy(args.checkpoint)
    backend = PairedGrootPolicy(
        policy,
        condition=args.condition,
        checkpoint=args.checkpoint,
        task_descriptions=descriptions,
        artifact_files=artifacts,
    )
    mode = set_batch_invariant_mode() if backend.repaired else contextlib.nullcontext()
    with mode, PolicyServer(policy=backend, host=args.host, port=args.port) as server:
        print(json.dumps(backend.metadata(), sort_keys=True), flush=True)
        server.run()


if __name__ == "__main__":
    main()

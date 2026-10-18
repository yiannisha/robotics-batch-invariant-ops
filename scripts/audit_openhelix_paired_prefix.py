"""Independently validate a resumable prefix of one OpenHelix CALVIN condition."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = {
    "native_singleton": (False, False),
    "native_dynamic": (False, True),
    "patched_singleton": (True, False),
    "patched_dynamic": (True, True),
}
DYNAMIC_BATCH_SCHEDULE = (1, 2, 4, 8)
TARGET_POSITION_SCHEDULE = ("first", "middle", "last")
BASE_SEED = 20261015
ROOT = Path(__file__).resolve().parents[1]
OPENHELIX_REVISION = "d88867fa56977f401a7c89d62b9ebce87d8078f0"
CALVIN_REVISION = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_REVISION = "1431a46bd36bde5903fb6345e68b5ccc30def666"
CHECKPOINT_REVISION = "1a6fabbe148af045ed0710d007ed24c36fa83a53"
POLICY_SHA256 = "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638"
OFFICIAL_EVALUATOR_SHA256 = "416b7f2b4a55ef359cca4ee3d8fe53d7506bc7c9a815e2d6f87386498329df5a"
PAIRED_WRAPPER_SHA256 = "c75b2404ead35fab690a491fead706ac176ef34689983ac9e4f17440e2d73acf"
ADAPTER_SHA256 = "1dbc435cb7913c0f322b7d3a4b6140e2ae2c0de58882c75287b7a8493c41f7e2"
REPLAY_MANIFEST_SHA256 = "e9b41acf80b5f027e823a0a48160ac5da293219b211e1a2a757d081923129bf8"
RUNTIME_PROBE_SHA256 = "499172bd05c11299cc0923a0621134b22524af4d5f90a930cb36f2bcceb6d732"
EVALUATOR_ASSETS_SHA256 = "93aa3a858864deecb2bdf2dcf3950b2e82c8d3a51512f919feef0c48d2c5b503"
EXPECTED_ARRAYS = {
    "actions": (7,),
    "robot_states": (15,),
    "scene_states": (24,),
    "rewards": (),
    "dones": (),
    "subtask_indices": (),
    "policy_call_indices": (),
    "action_chunk_indices": (),
    "batch_sizes": (),
    "target_positions": (),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        advised_through = 0
        try:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                position = stream.tell()
                if position - advised_through >= 64 * 1024 * 1024:
                    try:
                        os.posix_fadvise(
                            stream.fileno(), advised_through,
                            position - advised_through, os.POSIX_FADV_DONTNEED,
                        )
                        advised_through = position
                    except (AttributeError, OSError):
                        pass
        finally:
            try:
                os.posix_fadvise(
                    stream.fileno(), advised_through, 0, os.POSIX_FADV_DONTNEED
                )
            except (AttributeError, OSError):
                pass
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_bundle_sha256(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def request_seed(sequence_index: int, subtask_index: int, policy_call: int) -> int:
    payload = f"{BASE_SEED}:{sequence_index}:{subtask_index}:{policy_call}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (2**63 - 1)


def target_position(batch_size: int, policy_call: int) -> int:
    label = TARGET_POSITION_SCHEDULE[policy_call % len(TARGET_POSITION_SCHEDULE)]
    if label == "first" or batch_size == 1:
        return 0
    if label == "middle":
        return batch_size // 2
    return batch_size - 1


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    if set(arrays) != set(EXPECTED_ARRAYS):
        raise ValueError(f"OpenHelix trajectory array keys differ: {path}")
    lengths = {value.shape[0] for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"OpenHelix trajectory array lengths differ: {path}")
    for name, trailing_shape in EXPECTED_ARRAYS.items():
        if arrays[name].shape[1:] != trailing_shape:
            raise ValueError(f"OpenHelix trajectory shape differs for {name}: {path}")
    for name in ("actions", "robot_states", "scene_states", "rewards"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"OpenHelix trajectory contains nonfinite {name}: {path}")
    return arrays


def audit(root: Path) -> dict[str, Any]:
    condition = root.name
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported OpenHelix condition directory: {condition}")
    repaired, dynamic = CONDITIONS[condition]
    manifest_path = root / "condition_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("model") != "openhelix"
        or manifest.get("condition") != condition
        or manifest.get("invariant_ops") is not repaired
        or manifest.get("execution_mode")
        != ("dynamic_batch" if dynamic else "serial_singleton")
        or manifest.get("sequence_start") != 0
        or manifest.get("sequence_count") != 1000
        or manifest.get("official_sequence_count") != 1000
        or manifest.get("policy_calls_per_subtask") != 60
        or manifest.get("execute_len") != 20
        or manifest.get("request_local_noise") is not True
        or manifest.get("paired_noise_seed") != BASE_SEED
        or manifest.get("openhelix_revision") != OPENHELIX_REVISION
        or manifest.get("calvin_revision") != CALVIN_REVISION
        or manifest.get("calvin_env_revision") != CALVIN_ENV_REVISION
        or manifest.get("checkpoint_revision") != CHECKPOINT_REVISION
        or manifest.get("policy_sha256") != POLICY_SHA256
        or manifest.get("official_evaluator_sha256") != OFFICIAL_EVALUATOR_SHA256
        or manifest.get("paired_wrapper_sha256") != PAIRED_WRAPPER_SHA256
        or manifest.get("replay_manifest_sha256") != REPLAY_MANIFEST_SHA256
        or manifest.get("runtime_probe_sha256") != RUNTIME_PROBE_SHA256
        or manifest.get("evaluator_assets_sha256") != EVALUATOR_ASSETS_SHA256
    ):
        raise ValueError("OpenHelix condition manifest differs")
    provenance_files = {
        Path(manifest["policy_path"]): POLICY_SHA256,
        Path(manifest["official_evaluator_source"]): OFFICIAL_EVALUATOR_SHA256,
        ROOT / "scripts/run_openhelix_paired_calvin.py": PAIRED_WRAPPER_SHA256,
        ROOT / "scripts/model_invariance/adapters/openhelix_calvin.py": ADAPTER_SHA256,
        Path(manifest["replay_manifest"]): REPLAY_MANIFEST_SHA256,
        Path(manifest["runtime_probe"]): RUNTIME_PROBE_SHA256,
        Path(manifest["evaluator_assets"]): EVALUATOR_ASSETS_SHA256,
    }
    for source, expected_digest in provenance_files.items():
        if not source.is_file() or file_sha256(source) != expected_digest:
            raise ValueError(f"OpenHelix pinned provenance differs: {source}")

    metadata_files = sorted((root / "rollouts").glob("sequence_*.json.gz"))
    array_files = sorted((root / "rollouts").glob("sequence_*.npz"))
    metadata_indices = [int(path.name[9:13]) for path in metadata_files]
    array_indices = [int(path.name[9:13]) for path in array_files]
    if metadata_indices != list(range(len(metadata_indices))):
        raise ValueError("OpenHelix metadata is not one unique contiguous prefix")
    if array_indices != metadata_indices:
        raise ValueError("OpenHelix metadata/trajectory prefix differs")

    successes = 0
    tasks_completed = 0
    policy_calls = 0
    environment_steps = 0
    for sequence_index, metadata_path in enumerate(metadata_files):
        with gzip.open(metadata_path, "rt") as stream:
            row = json.load(stream)
        arrays_path = root / "rollouts" / f"sequence_{sequence_index:04d}.npz"
        schedule = row.get("request_schedule")
        noise_hashes = row.get("request_noise_sha256")
        if (
            row.get("schema_version") != 1
            or row.get("sequence_index") != sequence_index
            or row.get("initialization_id") != sequence_index
            or row.get("explicit_init_state_id") != sequence_index
            or row.get("condition") != condition
            or not isinstance(row.get("success"), bool)
            or not isinstance(row.get("tasks_completed"), int)
            or not 0 <= row["tasks_completed"] <= 5
            or row["success"] != (row["tasks_completed"] == 5)
            or row.get("sequence_length") != 5
            or not isinstance(row.get("task_sequence"), list)
            or len(row["task_sequence"]) != 5
            or not isinstance(row.get("initial_condition"), dict)
            or row.get("initial_condition_sha256")
            != canonical_sha256(row["initial_condition"])
            or not valid_hash(row.get("initial_observation_sha256"))
            or not isinstance(schedule, list)
            or not isinstance(noise_hashes, list)
            or len(schedule) != len(noise_hashes)
            or row.get("policy_calls") != len(schedule)
            or row.get("request_schedule_sha256") != canonical_sha256(schedule)
            or (noise_hashes and row.get("first_noise_sha256") != noise_hashes[0])
            or any(not valid_hash(value) for value in noise_hashes)
            or row.get("trajectory_npz_path") != str(arrays_path.resolve())
            or row.get("trajectory_npz_sha256") != file_sha256(arrays_path)
        ):
            raise ValueError(f"OpenHelix rollout metadata is incomplete or inconsistent: {metadata_path}")
        expected_request_pairs = []
        previous_subtask = -1
        previous_call = -1
        request_by_pair = {}
        for call_index, item in enumerate(schedule):
            if not isinstance(item, dict):
                raise ValueError(f"OpenHelix request schedule is malformed: {metadata_path}")
            subtask = item.get("subtask_index")
            call = item.get("policy_call")
            expected_batch = DYNAMIC_BATCH_SCHEDULE[call % 4] if dynamic and isinstance(call, int) else 1
            if (
                not isinstance(subtask, int)
                or not 0 <= subtask < 5
                or not isinstance(call, int)
                or not 0 <= call < 60
                or item.get("noise_seed") != request_seed(sequence_index, subtask, call)
                or item.get("noise_sha256") != noise_hashes[call_index]
                or item.get("batch_size") != expected_batch
                or item.get("target_position") != target_position(expected_batch, call)
            ):
                raise ValueError(f"OpenHelix request schedule is inconsistent: {metadata_path}")
            if subtask == previous_subtask:
                if call != previous_call + 1:
                    raise ValueError(f"OpenHelix policy-call order differs: {metadata_path}")
            else:
                if subtask != previous_subtask + 1 or call != 0:
                    raise ValueError(f"OpenHelix subtask order differs: {metadata_path}")
            pair = (subtask, call)
            expected_request_pairs.append(pair)
            request_by_pair[pair] = item
            previous_subtask, previous_call = pair
        expected_subtasks = 5 if row["tasks_completed"] == 5 else row["tasks_completed"] + 1
        if (
            len({pair[0] for pair in expected_request_pairs}) != expected_subtasks
            or (row["tasks_completed"] < 5 and previous_call != 59)
        ):
            raise ValueError(f"OpenHelix terminal request schedule differs: {metadata_path}")
        arrays = load_arrays(arrays_path)
        if (
            row.get("environment_steps") != arrays["actions"].shape[0]
            or row.get("trajectory_arrays_sha256") != array_bundle_sha256(arrays)
            or not all(
                0 <= int(value) < 5 for value in arrays["subtask_indices"]
            )
            or not all(
                0 <= int(value) < 60 for value in arrays["policy_call_indices"]
            )
            or not all(
                0 <= int(value) < 20 for value in arrays["action_chunk_indices"]
            )
            or not all(math.isfinite(float(value)) for value in arrays["rewards"])
        ):
            raise ValueError(f"OpenHelix trajectory evidence is inconsistent: {arrays_path}")
        observed_request_pairs = []
        previous_pair = None
        previous_chunk = -1
        for step in range(arrays["actions"].shape[0]):
            pair = (
                int(arrays["subtask_indices"][step]),
                int(arrays["policy_call_indices"][step]),
            )
            chunk = int(arrays["action_chunk_indices"][step])
            if pair != previous_pair:
                observed_request_pairs.append(pair)
                previous_chunk = -1
            request = request_by_pair.get(pair)
            if (
                request is None
                or chunk != previous_chunk + 1
                or int(arrays["batch_sizes"][step]) != request["batch_size"]
                or int(arrays["target_positions"][step]) != request["target_position"]
            ):
                raise ValueError(f"OpenHelix trajectory/request mapping differs: {arrays_path}")
            previous_pair = pair
            previous_chunk = chunk
        if observed_request_pairs != expected_request_pairs:
            raise ValueError(f"OpenHelix trajectory request order differs: {arrays_path}")
        successes += int(row["success"])
        tasks_completed += row["tasks_completed"]
        policy_calls += row["policy_calls"]
        environment_steps += row["environment_steps"]

    return {
        "schema_version": 1,
        "evidence_tier": "validated_partial_prefix_not_campaign_completion",
        "model_id": "openhelix",
        "condition": condition,
        "integrity_passed": True,
        "complete": len(metadata_files) == 1000,
        "sequences_validated": len(metadata_files),
        "planned_sequences": 1000,
        "full_sequence_successes": successes,
        "tasks_completed": tasks_completed,
        "task_opportunities": 5 * len(metadata_files),
        "policy_calls": policy_calls,
        "environment_steps": environment_steps,
        "condition_manifest_path": str(manifest_path.resolve()),
        "condition_manifest_sha256": file_sha256(manifest_path),
        "adapter_sha256": ADAPTER_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

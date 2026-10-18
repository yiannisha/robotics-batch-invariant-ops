"""Independent fail-closed audit for RDT-1B ManiSkill rollout evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
TASKS = (
    "PegInsertionSide-v1",
    "PickCube-v1",
    "StackCube-v1",
    "PlugCharger-v1",
    "PushCube-v1",
)
PINNED = {
    "checkpoint_sha256": "3fa89cbf7bbb1b2559b29f9ae153b431541cdaa0b3f1ee44b83b9934599b9fb8",
    "official_evaluator_sha256": "6cd11d35d82c6aff6cd4625a2acdb54f8ab1cad0ab954942443450e3b57bb3dd",
    "official_wrapper_sha256": "503644ff977704f212e007bc6cd05e4ea7ce84c66324bc31eccf36da31555bb0",
    "official_policy_sha256": "b7d1aacec1a59df2de8b5bfbecf96e409abc18ca2db7442ae2d44fe5b647d953",
    "paired_wrapper_sha256": "3a63ff9dfc7cac4cbe5692de197fb5ee428e250e2aab68b9b70bf9ae452f26ca",
    "shared_rollout_wrapper_sha256": "91b42622ebed23a2f8b50049f5de6c4326bc876e53e3e7ccb5a7ef8d07b72f4e",
    "task_manifest_sha256": "3dbfd9af709388a3e9662563e8f5fb8528e85e67d2acb34a184a6d96ef030efe",
}
TASK_MANIFEST = Path(
    "/workspace/robotics-batch-invariant-ops/results/cross_model/rdt_1b/"
    "full/closed_loop/paired/evaluator_task_manifest.json"
)
BASE_SEED = 20241201


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def finite_vector(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        isinstance(number, (int, float))
        and not isinstance(number, bool)
        and math.isfinite(number)
        for number in value
    )


def expected_noise_hash(task: int, initialization: int) -> str:
    sequence = np.random.SeedSequence([BASE_SEED, task, initialization, 0, 0x524454])
    noise = np.random.default_rng(sequence).standard_normal((64, 128), dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()


def audit(root: Path) -> dict:
    if not TASK_MANIFEST.is_file() or sha256(TASK_MANIFEST) != PINNED["task_manifest_sha256"]:
        raise ValueError("RDT pinned evaluator task manifest is missing or changed")
    task_manifest = json.loads(TASK_MANIFEST.read_text())
    if (
        task_manifest.get("task_names") != list(TASKS)
        or task_manifest.get("task_ids") != list(range(5))
        or task_manifest.get("base_seed") != BASE_SEED
        or task_manifest.get("initializations_per_task") != 50
    ):
        raise ValueError("RDT evaluator task manifest differs from the pinned suite")

    manifest = json.loads((root / "condition_manifest.json").read_text())
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("unknown RDT condition")
    batch_size, repaired = CONDITIONS[condition]
    episodes_per_task = manifest.get("episodes_per_task")
    task_count = manifest.get("task_count")
    if (
        not isinstance(episodes_per_task, int)
        or not 1 <= episodes_per_task <= 50
        or not isinstance(task_count, int)
        or not 1 <= task_count <= 5
    ):
        raise ValueError("invalid RDT probe/full cardinality")
    required = {
        "schema_version": 1,
        "model": "rdt_1b",
        "condition": condition,
        "checkpoint": "/workspace/assets/rdt/checkpoint",
        "seed": BASE_SEED,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "maniskill_five",
        "episode_length": 400,
        "reward_mode": "sparse",
        "control_mode": "pd_joint_pos",
        "control_frequency": 25,
        "action_chunk_size": 64,
        "executed_action_stride": 4,
        "request_local_noise": True,
        "noise_shape": [64, 128],
        "diffusion_steps": 5,
        "genuine_batched_policy_call": True,
        "language_padding": "fixed_right_30" if repaired else "longest_in_batch_right",
        "explicit_init_state_by_initialization_id": True,
        **PINNED,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("RDT manifest differs from pinned evaluator contract")
    if manifest.get("artifact_files") != task_manifest.get("artifact_files"):
        raise ValueError("RDT condition and task manifests disagree about exact artifacts")

    metadata = json.loads((root / "server_metadata.json").read_text())
    expected_metadata = {
        "protocol_version": "2.1",
        "policy_type": "rdt_1b",
        "checkpoint": "/workspace/assets/rdt/checkpoint",
        "task_suite": "maniskill_five",
        "paired_condition": condition,
        "genuine_batched_policy_call": True,
        "maximum_request_batch": batch_size,
        "batch_invariant_ops": repaired,
        "request_local_noise": True,
        "diffusion_steps": 5,
        "noise_shape": [64, 128],
        "language_padding": "fixed_right_30" if repaired else "longest_in_batch_right",
        "image_history": 2,
        "num_cameras": 3,
        "action_chunk_size": 64,
        "executed_action_stride": 4,
        "action_dim": 8,
        "control_mode": "pd_joint_pos",
        "control_frequency": 25,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("RDT live server metadata differs from the requested condition")
    if metadata.get("artifact_files") != manifest.get("artifact_files"):
        raise ValueError("RDT live server loaded a different artifact set")
    canonical_tokens = metadata.get("canonical_language_tokens")
    if not isinstance(canonical_tokens, int) or canonical_tokens <= 0:
        raise ValueError("RDT live server did not pin canonical language width")

    counts, successes = Counter(), Counter()
    seen: set[tuple[int, int]] = set()
    files = sorted((root / "rollouts").glob("*.json"))
    files += sorted((root / "rollouts").glob("*.json.gz"))
    for path in files:
        payload = read_json(path)
        task = payload.get("task_id")
        ids = payload.get("explicit_init_state_ids")
        seeds = payload.get("seeds")
        rows = payload.get("episodes")
        if (
            payload.get("task_group") != "maniskill_five"
            or task not in range(task_count)
            or payload.get("task_name") != TASKS[task]
            or not isinstance(ids, list)
            or not ids
            or len(ids) > batch_size
            or ids != list(range(ids[0], ids[0] + len(ids)))
            or any(index not in range(episodes_per_task) for index in ids)
            or seeds != [BASE_SEED + index for index in ids]
            or not isinstance(rows, list)
            or len(rows) != len(ids)
        ):
            raise ValueError(f"invalid RDT physical rollout group: {path}")
        if batch_size == 1 and len(ids) != 1:
            raise ValueError(f"singleton RDT rollout is physically batched: {path}")
        for row, initialization, reset_seed in zip(rows, ids, seeds):
            key = (task, initialization)
            if key in seen:
                raise ValueError(f"duplicate RDT task/reset: {key}")
            seen.add(key)
            states = row.get("trajectory")
            actions = row.get("action_sequence")
            rewards = row.get("rewards")
            dones = row.get("done")
            physical_batches = row.get("policy_batch_sizes")
            success = row.get("success")
            if (
                row.get("task") != task
                or row.get("task_group") != "maniskill_five"
                or row.get("task_name") != TASKS[task]
                or row.get("initialization_id") != initialization
                or row.get("reset_seed") != reset_seed
                or not isinstance(success, bool)
                or not valid_hash(row.get("initial_observation_sha256"))
                or row.get("first_noise_sha256") != expected_noise_hash(task, initialization)
                or not isinstance(states, list)
                or not states
                or not isinstance(actions, list)
                or not isinstance(rewards, list)
                or not isinstance(dones, list)
                or not len(states) == len(actions) == len(rewards) == len(dones)
                or len(states) > 400
                or any(not finite_vector(state, 8) for state in states)
                or any(not finite_vector(action, 8) for action in actions)
                or any(
                    not isinstance(reward, (int, float))
                    or isinstance(reward, bool)
                    or not math.isfinite(reward)
                    for reward in rewards
                )
                or any(not isinstance(done, bool) for done in dones)
                or not isinstance(physical_batches, list)
                or not physical_batches
                or physical_batches[0] != len(ids)
                or any(
                    not isinstance(size, int) or not 1 <= size <= len(ids)
                    for size in physical_batches
                )
            ):
                raise ValueError(f"incomplete or inconsistent RDT episode: {path}")
            counts[task] += 1
            successes[task] += int(success)

    for task in range(task_count):
        observed = {
            initialization
            for observed_task, initialization in seen
            if observed_task == task
        }
        if observed != set(range(len(observed))):
            raise ValueError(f"noncontiguous RDT reset prefix for task {task}")
        if counts[task] and any(
            counts[previous] != episodes_per_task for previous in range(task)
        ):
            raise ValueError(f"RDT task {task} started before its predecessor completed")

    expected_total = task_count * episodes_per_task
    info_path = root / "official/eval_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        overall = info.get("overall", {})
        if (
            overall.get("n_episodes") != sum(counts.values())
            or overall.get("n_success") != sum(successes.values())
        ):
            raise ValueError("RDT raw episodes and aggregate metrics differ")
    return {
        "schema_version": 1,
        "evidence_tier": (
            "checkpoint_matched_probe"
            if expected_total < 250
            else "running_prefix_only_not_paired_completion"
        ),
        "model_id": "rdt_1b",
        "condition": condition,
        "task_count": task_count,
        "episodes_per_task": episodes_per_task,
        "episodes_cached": sum(counts.values()),
        "successes_cached": sum(successes.values()),
        "planned_episodes": expected_total,
        "complete": sum(counts.values()) == expected_total,
        "integrity_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.condition_root), sort_keys=True))


if __name__ == "__main__":
    main()

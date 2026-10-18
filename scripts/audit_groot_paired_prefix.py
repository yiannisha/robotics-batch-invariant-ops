"""Independent fail-closed audit for GR00T N1.7 LIBERO-10 rollout evidence."""

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
PINNED = {
    "checkpoint_index_sha256": "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
    "official_evaluator_sha256": "588af9189bd2f9b5acf964e60a30c84bc2445843f7617d012bc5ef410706b13a",
    "official_environment_sha256": "49a8017517f1f21b7c21bf5f599244189720c83785763e4ab05a3cd0292cf5b1",
    "official_policy_sha256": "ad1183a3ce1ebed6cace4a9cb8c5a123e823a0760ebf46f3694639a5ad1521ec",
    "runtime_helper_sha256": "9473303abc2a380532c43c7aff2ba20dededbb48584108030d25f2366971c5f4",
    "paired_wrapper_sha256": "ca27aa11424d57b17fc933fcb24106d395e47348c0655785c7b635ae746cd371",
    "shared_rollout_wrapper_sha256": "72a4e5d7a80a02d017c8a613ffa6c82a44b0aa8a8d610480b290b88b00bd179b",
    "task_manifest_sha256": "e48840a0aa18d822bfc445e420854634d0024104f1dc82a6945a17aa48c26a85",
}
TASK_MANIFEST = Path(
    "/workspace/robotics-batch-invariant-ops/results/cross_model/groot_n17/"
    "full/closed_loop/paired/evaluator_task_manifest.json"
)
SEED = 20261005


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
    sequence = np.random.SeedSequence([SEED, task, initialization, 0, 0x47524F4F54])
    noise = np.random.default_rng(sequence).standard_normal((40, 132), dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()


def audit(root: Path) -> dict:
    if not TASK_MANIFEST.is_file() or sha256(TASK_MANIFEST) != PINNED["task_manifest_sha256"]:
        raise ValueError("GR00T pinned evaluator task manifest is missing or changed")
    task_manifest = json.loads(TASK_MANIFEST.read_text())
    descriptions = task_manifest.get("task_descriptions")
    if not isinstance(descriptions, list) or len(descriptions) != 10:
        raise ValueError("GR00T task manifest does not contain exactly ten descriptions")

    manifest = json.loads((root / "condition_manifest.json").read_text())
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("unknown GR00T condition")
    batch_size, repaired = CONDITIONS[condition]
    episodes_per_task = manifest.get("episodes_per_task")
    task_count = manifest.get("task_count")
    if (
        not isinstance(episodes_per_task, int)
        or not 1 <= episodes_per_task <= 50
        or not isinstance(task_count, int)
        or not 1 <= task_count <= 10
    ):
        raise ValueError("invalid GR00T probe/full cardinality")
    required = {
        "schema_version": 1,
        "model": "groot_n17",
        "condition": condition,
        "checkpoint": "/workspace/assets/groot_n17/repo/libero_10",
        "seed": SEED,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "episode_length": 720,
        "num_steps_wait": 10,
        "num_open_loop_steps": 8,
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "noise_shape": [40, 132],
        "flow_steps": 4,
        "genuine_batched_policy_call": True,
        "official_processor_and_decode": True,
        "official_environment_action_transform": True,
        **PINNED,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("GR00T manifest differs from pinned evaluator contract")
    if manifest.get("artifact_files") != task_manifest.get("artifact_files"):
        raise ValueError("GR00T condition and task manifests disagree about exact artifacts")

    metadata = json.loads((root / "server_metadata.json").read_text())
    expected_metadata = {
        "protocol_version": "1.0",
        "policy_type": "groot_n17",
        "task_suite": "libero_10",
        "paired_condition": condition,
        "maximum_request_batch": batch_size,
        "batch_invariant_ops": repaired,
        "genuine_batched_policy_call": True,
        "request_local_noise": True,
        "noise_shape": [40, 132],
        "flow_steps": 4,
        "action_horizon": 16,
        "executed_action_steps": 8,
        "action_dim": 7,
        "processor": "official_Gr00tN1d7Processor",
        "environment_action_transform": "official_LiberoEnv.step",
        "checkpoint": "/workspace/assets/groot_n17/repo/libero_10",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("GR00T live server metadata differs from the requested condition")
    if metadata.get("artifact_files") != manifest.get("artifact_files"):
        raise ValueError("GR00T live server loaded a different artifact set")

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
            payload.get("task_group") != "libero_10"
            or task not in range(task_count)
            or not isinstance(ids, list)
            or not ids
            or len(ids) > batch_size
            or ids != list(range(ids[0], ids[0] + len(ids)))
            or any(index not in range(episodes_per_task) for index in ids)
            or seeds != [SEED + index for index in ids]
            or not isinstance(rows, list)
            or len(rows) != len(ids)
        ):
            raise ValueError(f"invalid GR00T physical rollout group: {path}")
        if batch_size == 1 and len(ids) != 1:
            raise ValueError(f"singleton GR00T rollout is physically batched: {path}")
        for row, initialization, reset_seed in zip(rows, ids, seeds):
            key = (task, initialization)
            if key in seen:
                raise ValueError(f"duplicate GR00T task/reset: {key}")
            seen.add(key)
            states = row.get("trajectory")
            actions = row.get("action_sequence")
            rewards = row.get("rewards")
            dones = row.get("done")
            physical_batches = row.get("policy_batch_sizes")
            success = row.get("success")
            if (
                row.get("task") != task
                or row.get("task_group") != "libero_10"
                or row.get("task_description") != descriptions[task]
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
                or len(states) > 720
                or any(not finite_vector(state, 8) for state in states)
                or any(not finite_vector(action, 7) for action in actions)
                or any(
                    not isinstance(reward, (int, float)) or not math.isfinite(reward)
                    for reward in rewards
                )
                or any(not isinstance(done, bool) for done in dones)
                or success != any(dones)
                or (success and dones[-1] is not True)
                or not isinstance(physical_batches, list)
                or not physical_batches
                or physical_batches[0] != len(ids)
                or any(
                    not isinstance(size, int) or not 1 <= size <= len(ids)
                    for size in physical_batches
                )
            ):
                raise ValueError(f"incomplete or inconsistent GR00T episode: {path}")
            counts[task] += 1
            successes[task] += int(success)

    for task in range(task_count):
        observed = {
            initialization
            for observed_task, initialization in seen
            if observed_task == task
        }
        if observed != set(range(len(observed))):
            raise ValueError(f"noncontiguous GR00T reset prefix for task {task}")
        if counts[task] and any(
            counts[previous] != episodes_per_task for previous in range(task)
        ):
            raise ValueError(f"GR00T task {task} started before its predecessor completed")

    expected_total = task_count * episodes_per_task
    info_path = root / "official/eval_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        overall = info.get("overall", {})
        if (
            overall.get("n_episodes") != sum(counts.values())
            or overall.get("n_success") != sum(successes.values())
        ):
            raise ValueError("GR00T raw episodes and aggregate metrics differ")
    return {
        "schema_version": 1,
        "evidence_tier": (
            "checkpoint_matched_probe"
            if expected_total < 500
            else "running_prefix_only_not_paired_completion"
        ),
        "model_id": "groot_n17",
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

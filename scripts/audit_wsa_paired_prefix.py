"""Independent fail-closed audit for WSA-Base LIBERO-10 rollout evidence."""

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
    "checkpoint_sha256": "603eacbb1a7da80dd2d41cbbd814a833bd57dec30601da5fda35651d6dd46e67",
    "official_evaluator_sha256": "329bb21d118d904f7a39cfb422ebe6c9a1acddc9995cb438721dea42ce7f9f82",
    "official_server_sha256": "31d38d891fef6f703263aa457fa139d1ad8880c809cdebfccdd098841ce91f2a",
    "official_policy_sha256": "6886ccfc03dab7d9d52f59eb676dbccacafa9645288ef63deb3472af3ec5c838",
    "paired_wrapper_sha256": "7172e366fc61fe8148ad1e080f90026a2a4c1b52ce5f9c671fb730dda753e3b3",
    "shared_rollout_wrapper_sha256": "b0f4a1d53716cf8b6da936f310a9f26111572a43a8f50b0755bf19f245c967c2",
    "task_manifest_sha256": "76a42e3b2e8e3bc3c6591775b2e3cc993947dbb277096dd1a9ab3ee66dfcaf7b",
}
TASK_MANIFEST = Path(
    "/workspace/robotics-batch-invariant-ops/results/cross_model/wsa_base/"
    "full/closed_loop/paired/evaluator_task_manifest.json"
)
SEED = 20261005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    sequence = np.random.SeedSequence([SEED, task, initialization, 0, 0x575341])
    noise = np.random.default_rng(sequence).standard_normal((10, 32), dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()


def audit(root: Path) -> dict:
    if not TASK_MANIFEST.is_file() or sha256(TASK_MANIFEST) != PINNED["task_manifest_sha256"]:
        raise ValueError("WSA pinned evaluator task manifest is missing or changed")
    task_manifest = json.loads(TASK_MANIFEST.read_text())
    descriptions = task_manifest.get("task_descriptions")
    if not isinstance(descriptions, list) or len(descriptions) != 10:
        raise ValueError("WSA task manifest does not contain exactly ten descriptions")

    manifest = json.loads((root / "condition_manifest.json").read_text())
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("unknown WSA condition")
    batch_size, repaired = CONDITIONS[condition]
    episodes_per_task = manifest.get("episodes_per_task")
    task_count = manifest.get("task_count")
    if (
        not isinstance(episodes_per_task, int)
        or not 1 <= episodes_per_task <= 50
        or not isinstance(task_count, int)
        or not 1 <= task_count <= 10
    ):
        raise ValueError("invalid WSA probe/full cardinality")
    required = {
        "schema_version": 1,
        "model": "wsa_base",
        "condition": condition,
        "checkpoint": "/workspace/assets/wsa/checkpoint",
        "seed": SEED,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "episode_length": 520,
        "num_steps_wait": 10,
        "num_open_loop_steps": 10,
        "image_history_interval": 15,
        "control_mode": "absolute",
        "gripper_mode": "libero_open_prob",
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "noise_shape": [10, 32],
        "flow_steps": 10,
        "genuine_batched_policy_call": True,
        **PINNED,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("WSA manifest differs from pinned evaluator contract")
    if manifest.get("artifact_files") != task_manifest.get("artifact_files"):
        raise ValueError("WSA condition and task manifests disagree about exact artifacts")

    metadata = json.loads((root / "server_metadata.json").read_text())
    expected_metadata = {
        "protocol_version": "2.1",
        "policy_type": "wsa_base",
        "task_suite": "libero_10",
        "paired_condition": condition,
        "maximum_request_batch": batch_size,
        "batch_invariant_ops": repaired,
        "genuine_batched_policy_call": True,
        "request_local_noise": True,
        "noise_shape": [10, 32],
        "action_chunk_size": 10,
        "action_dim": 7,
        "flow_steps": 10,
        "num_images_in_input": 3,
        "third_image_masked": True,
        "control_mode": "absolute",
        "gripper_mode": "libero_open_prob",
        "image_history_interval": 15,
        "checkpoint": "/workspace/assets/wsa/checkpoint",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("WSA live server metadata differs from the requested condition")
    if metadata.get("artifact_files") != manifest.get("artifact_files"):
        raise ValueError("WSA live server loaded a different artifact set")

    counts, successes = Counter(), Counter()
    seen: set[tuple[int, int]] = set()
    rollout_root = root / "rollouts"
    plain_files = {path.name: path for path in rollout_root.glob("*.json")}
    compressed_files = {
        path.name.removesuffix(".gz"): path
        for path in rollout_root.glob("*.json.gz")
    }
    duplicate_names = set(plain_files) & set(compressed_files)
    if duplicate_names:
        raise ValueError(
            "duplicate compressed and uncompressed WSA rollout groups: "
            + ", ".join(sorted(duplicate_names))
        )
    files = sorted([*plain_files.values(), *compressed_files.values()])
    for path in files:
        if path.name.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        else:
            payload = json.loads(path.read_text())
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
            raise ValueError(f"invalid WSA physical rollout group: {path}")
        if batch_size == 1 and len(ids) != 1:
            raise ValueError(f"singleton WSA rollout is physically batched: {path}")
        for row, initialization, reset_seed in zip(rows, ids, seeds):
            key = (task, initialization)
            if key in seen:
                raise ValueError(f"duplicate WSA task/reset: {key}")
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
                or len(states) > 520
                or any(not finite_vector(state, 8) for state in states)
                or any(not finite_vector(action, 7) for action in actions)
                or any(not isinstance(reward, (int, float)) or not math.isfinite(reward) for reward in rewards)
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
                raise ValueError(f"incomplete or inconsistent WSA episode: {path}")
            counts[task] += 1
            successes[task] += int(success)

    for task in range(task_count):
        observed = {
            initialization
            for observed_task, initialization in seen
            if observed_task == task
        }
        if observed != set(range(len(observed))):
            raise ValueError(f"noncontiguous WSA reset prefix for task {task}")
        if counts[task] and any(
            counts[previous] != episodes_per_task for previous in range(task)
        ):
            raise ValueError(f"WSA task {task} started before its predecessor completed")

    expected_total = task_count * episodes_per_task
    info_path = root / "official/eval_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        overall = info.get("overall", {})
        if (
            overall.get("n_episodes") != sum(counts.values())
            or overall.get("n_success") != sum(successes.values())
        ):
            raise ValueError("WSA raw episodes and aggregate metrics differ")
    return {
        "schema_version": 1,
        "evidence_tier": (
            "checkpoint_matched_probe"
            if expected_total < 500
            else "running_prefix_only_not_paired_completion"
        ),
        "model_id": "wsa_base",
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

"""Read-only integrity check for a running LeRobot LIBERO condition prefix.

This catches malformed, duplicated, or noncontiguous cached rollouts early.
It never promotes a prefix to the full four-condition closed-loop result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CONDITIONS = {
    "native_singleton": 1, "native_dynamic": 4,
    "patched_singleton": 1, "patched_dynamic": 4,
}
FIXED_ENVIRONMENT_BATCH_SIZE = 4
PAIRED_PROTOCOL_VERSION = 2
STATE_DIMENSIONS = {
    "xvla": 20,
    "smolvla": 8,
    "pi_fast": 8,
    "univla_baai": 8,
    "molmoact2": 8,
}


def rollout_paths(directory: Path) -> list[Path]:
    return sorted((*directory.glob("*.json"), *directory.glob("*.json.gz")))


def read_rollout(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def rollout_tensor_path(path: Path) -> Path:
    return path.with_suffix("").with_suffix(".pt") if path.suffix == ".gz" else path.with_suffix(".pt")


def valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def finite_vector(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        isinstance(number, (int, float)) and math.isfinite(number) for number in value
    )


def audit(root: Path) -> dict:
    manifest_path = root / "condition_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    condition = manifest.get("condition")
    policy_batch_size = CONDITIONS.get(condition)
    protocol_version = manifest.get("paired_protocol_version", 1)
    if protocol_version == 1:
        environment_batch_size = policy_batch_size
        execution_mode = "legacy_environment_batch_tracks_policy_batch"
    elif protocol_version == PAIRED_PROTOCOL_VERSION:
        environment_batch_size = manifest.get("environment_batch_size")
        execution_mode = manifest.get("execution_mode")
        expected_mode = "serial_singleton" if policy_batch_size == 1 else "dynamic_batch"
        if (manifest.get("policy_batch_size") != policy_batch_size
                or environment_batch_size != FIXED_ENVIRONMENT_BATCH_SIZE
                or execution_mode != expected_mode
                or manifest.get("smoke_test") is not False):
            raise ValueError("paired protocol does not isolate policy batch from physical environment batch")
    else:
        raise ValueError("unknown LeRobot paired protocol version")
    if (policy_batch_size is None or manifest.get("batch_size") != policy_batch_size
            or manifest.get("episodes_per_task") != 50
            or manifest.get("explicit_init_state_by_initialization_id") is not True
            or manifest.get("model") not in STATE_DIMENSIONS
            or not isinstance(manifest.get("seed"), int)):
        raise ValueError("not a pinned 50-reset LeRobot LIBERO condition")
    if manifest["model"] in ("univla_baai", "molmoact2") and (
        manifest.get("schema_version") != 2
        or protocol_version != PAIRED_PROTOCOL_VERSION
    ):
        raise ValueError("exact-checkpoint paired campaign requires protocol-v2 evidence")
    if manifest.get("task_suite") != "libero_10":
        raise ValueError("paired evaluator task suite is not pinned to LIBERO-Long")
    if manifest.get("task_ids", list(range(10))) != list(range(10)):
        raise ValueError("full paired evaluator must cover all ten LIBERO-Long tasks")
    if manifest.get("model") == "xvla" and (
        manifest.get("control_mode") != "absolute"
        or manifest.get("episode_length") != 800
    ):
        raise ValueError("X-VLA paired evaluator does not match its official absolute/800 recipe")
    if manifest.get("model") == "pi_fast" and (
        manifest.get("control_mode") != "relative"
        or manifest.get("episode_length") != 520
    ):
        raise ValueError("PI0-FAST paired evaluator is not pinned to relative-control LIBERO-Long")
    if manifest.get("model") in ("smolvla", "univla_baai", "molmoact2") and (
        manifest.get("control_mode") != "relative"
        or manifest.get("episode_length") != 520
    ):
        model_label = {
            "smolvla": "SmolVLA",
            "univla_baai": "UniVLA",
            "molmoact2": "MolmoAct2",
        }[manifest["model"]]
        raise ValueError(
            f"{model_label} paired evaluator is not pinned to relative-control LIBERO-Long"
        )
    if manifest.get("model") == "univla_baai" and (
        manifest.get("request_local_noise") is not False
        or manifest.get("deterministic_equivalent") is not True
        or not valid_hash(manifest.get("deterministic_contract_sha256"))
    ):
        raise ValueError("UniVLA deterministic decoding contract is not pinned")
    if manifest.get("model") == "molmoact2" and manifest.get("request_local_noise") is not True:
        raise ValueError("MolmoAct2 request-local stochastic contract is not pinned")
    seed = manifest["seed"]
    state_dimension = STATE_DIMENSIONS[manifest["model"]]
    keys = set()
    first_noise_hashes_by_task = defaultdict(set)
    counts = Counter()
    physical_extra = Counter()
    planned_successes = 0
    files = rollout_paths(root / "rollouts")
    for path in files:
        if not rollout_tensor_path(path).is_file():
            raise ValueError(f"cached evidence lacks official rollout tensor {path}")
        payload = read_rollout(path)
        task = payload.get("task_id")
        seeds = payload.get("seeds")
        episodes = payload.get("episodes")
        explicit_init_state_ids = payload.get("explicit_init_state_ids")
        if (payload.get("task_group") != "libero_10" or task not in range(10)
                or not isinstance(seeds, list) or len(seeds) != environment_batch_size
                or explicit_init_state_ids != [int(value) - seed for value in seeds]
                or not isinstance(episodes, list) or len(episodes) != environment_batch_size
                or sorted(seeds) != list(range(seeds[0], seeds[0] + environment_batch_size))):
            raise ValueError(f"invalid physical rollout batch {path}")
        for episode, reset_seed in zip(episodes, seeds):
            initialization = reset_seed - seed
            if (episode.get("task") != task or episode.get("task_group") != "libero_10"
                    or episode.get("reset_seed") != reset_seed
                    or episode.get("initialization_id") != initialization
                    or initialization < 0 or initialization >= 52):
                raise ValueError(f"rollout task/reset identity differs at {path}")
            key = (task, initialization)
            if key in keys:
                raise ValueError(f"duplicate paired task/reset {key}")
            keys.add(key)
            states = episode.get("trajectory")
            actions = episode.get("action_sequence")
            rewards = episode.get("rewards")
            dones = episode.get("done")
            if (not isinstance(episode.get("success"), bool)
                    or not valid_hash(episode.get("initial_observation_sha256"))
                    or (protocol_version == PAIRED_PROTOCOL_VERSION
                        and not valid_hash(episode.get("initial_raw_observation_sha256")))
                    or not valid_hash(episode.get("first_noise_sha256"))
                    or not isinstance(states, list) or not states
                    or not isinstance(actions, list)
                    or not isinstance(rewards, list)
                    or not isinstance(dones, list)
                    or not len(states) == len(actions) == len(rewards) == len(dones)
                    or any(not finite_vector(state, state_dimension) for state in states)
                    or any(not finite_vector(action, 7) for action in actions)):
                raise ValueError(f"incomplete or nonfinite state/action/outcome evidence in {path}")
            if initialization < 50:
                counts[task] += 1
                planned_successes += int(episode["success"])
            else:
                if environment_batch_size != 4:
                    raise ValueError("singleton evaluator emitted an extra physical reset")
                physical_extra[task] += 1
            first_noise_hashes_by_task[task].add(episode["first_noise_sha256"])
    for task in range(10):
        planned = {index for observed_task, index in keys if observed_task == task and index < 50}
        if planned != set(range(len(planned))):
            raise ValueError(f"noncontiguous paired initialization prefix for task {task}")
        extra = {index for observed_task, index in keys if observed_task == task and index >= 50}
        if extra and (environment_batch_size != 4 or planned != set(range(50)) or extra != {50, 51}):
            raise ValueError(f"unexpected final physical B4 slots for task {task}")
        if counts[task] and any(counts[previous] != 50 for previous in range(task)):
            raise ValueError(f"task {task} started before a previous task finished")
    if manifest["model"] == "pi_fast":
        contract = manifest.get("deterministic_decoding_contract")
        expected = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        if not contract or any(hashes != {expected} for hashes in first_noise_hashes_by_task.values()):
            raise ValueError("PI0-FAST prefix differs from deterministic decoder contract")
    elif manifest["model"] == "univla_baai":
        expected = manifest["deterministic_contract_sha256"]
        if any(hashes != {expected} for hashes in first_noise_hashes_by_task.values()):
            raise ValueError("UniVLA prefix differs from deterministic decoder contract")
    elif any(len(first_noise_hashes_by_task[task]) != sum(observed_task == task for observed_task, _ in keys)
             for task in range(10)):
        raise ValueError("stochastic paired requests reused a first-noise tensor within one task")
    return {
        "schema_version": 1,
        "evidence_tier": "running_prefix_only_not_paired_completion",
        "model_id": manifest["model"], "condition": condition,
        "paired_protocol_version": protocol_version,
        "policy_batch_size": policy_batch_size,
        "physical_batch_size": environment_batch_size,
        "environment_batch_size": environment_batch_size,
        "execution_mode": execution_mode,
        "causal_pairing_eligible": protocol_version == PAIRED_PROTOCOL_VERSION,
        "environment_state_dimension": state_dimension,
        "rollout_files": len(files),
        "planned_episodes_cached": sum(counts.values()),
        "planned_successes_cached": planned_successes,
        "physical_extra_slots": sum(physical_extra.values()),
        "per_task_planned_counts": {str(task): counts[task] for task in range(10)},
        "all_ten_tasks_have_50_resets": all(counts[task] == 50 for task in range(10)),
        "integrity_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.condition_root), sort_keys=True))


if __name__ == "__main__":
    main()

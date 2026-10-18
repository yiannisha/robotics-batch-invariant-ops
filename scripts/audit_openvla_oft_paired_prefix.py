"""Independent fail-closed audit for OpenVLA-OFT LIBERO-Spatial evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
PINNED = {
    "checkpoint_index_sha256": "ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659",
    "official_evaluator_sha256": "701a20b6ca2942aa36e0e7d567b25647268b936a23af79c5893534f10bec7802",
    "official_libero_utils_sha256": "489c16dc8220845b2936ddb81aafd7469061d2009b0edddb2f85703bc51a204f",
    "official_policy_sha256": "b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6",
    "paired_wrapper_sha256": "c6c55d65321f4416238f6bc3b6e232574a383527c99da838e562d2615396cf53",
    "shared_rollout_wrapper_sha256": "54fde040e86d8b100398417befd2bcf6ef68a53c381a19cd63a150b20cf3c3f0",
    "task_manifest_sha256": "79211e0ec45330237276af4a520ebd6022b3485853c790a22b6fa02fdf7871b5",
    "deterministic_decoding_sha256": "30231c9e6c335f309ffd2fde0e10e5159d624a6c96ec38c0ea6e39b3d7bf819e",
}


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
        isinstance(number, (int, float)) and math.isfinite(number) for number in value
    )


def audit(root: Path) -> dict:
    manifest = json.loads((root / "condition_manifest.json").read_text())
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("unknown OpenVLA-OFT condition")
    batch_size, repaired = CONDITIONS[condition]
    episodes_per_task = manifest.get("episodes_per_task")
    task_count = manifest.get("task_count", 10)
    if (not isinstance(episodes_per_task, int) or not 1 <= episodes_per_task <= 50
            or not isinstance(task_count, int) or not 1 <= task_count <= 10):
        raise ValueError("invalid OpenVLA-OFT probe/full cardinality")
    required = {
        "schema_version": 1,
        "model": "openvla_oft",
        "condition": condition,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "libero_spatial",
        "episode_length": 220,
        "num_steps_wait": 10,
        "num_open_loop_steps": 8,
        "explicit_init_state_by_initialization_id": True,
        "genuine_batched_policy_call": True,
        "center_crop_area": 0.9,
        "num_images_in_input": 2,
        "use_proprio": True,
        "prompt_padding": "canonical_right" if repaired else "longest_in_batch_right",
        **PINNED,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("OpenVLA-OFT manifest differs from pinned evaluator contract")
    seed = manifest.get("seed")
    if not isinstance(seed, int):
        raise ValueError("OpenVLA-OFT seed is not pinned")

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
        if (payload.get("task_group") != "libero_spatial" or task not in range(task_count)
                or not isinstance(ids, list) or not ids or len(ids) > batch_size
                or ids != list(range(ids[0], ids[0] + len(ids)))
                or any(index not in range(episodes_per_task) for index in ids)
                or seeds != [seed + index for index in ids]
                or not isinstance(rows, list) or len(rows) != len(ids)):
            raise ValueError(f"invalid OpenVLA-OFT physical rollout group: {path}")
        if batch_size == 1 and len(ids) != 1:
            raise ValueError(f"singleton OpenVLA-OFT rollout is physically batched: {path}")
        for row, initialization, reset_seed in zip(rows, ids, seeds):
            key = (task, initialization)
            if key in seen:
                raise ValueError(f"duplicate OpenVLA-OFT task/reset: {key}")
            seen.add(key)
            states = row.get("trajectory")
            actions = row.get("action_sequence")
            rewards = row.get("rewards")
            dones = row.get("done")
            physical_batches = row.get("policy_batch_sizes")
            if (row.get("task") != task or row.get("task_group") != "libero_spatial"
                    or row.get("initialization_id") != initialization
                    or row.get("reset_seed") != reset_seed
                    or not isinstance(row.get("success"), bool)
                    or not valid_hash(row.get("initial_observation_sha256"))
                    or row.get("first_noise_sha256") != PINNED["deterministic_decoding_sha256"]
                    or not isinstance(states, list) or not states
                    or not isinstance(actions, list) or not isinstance(rewards, list)
                    or not isinstance(dones, list)
                    or not len(states) == len(actions) == len(rewards) == len(dones)
                    or any(not finite_vector(state, 8) for state in states)
                    or any(not finite_vector(action, 7) for action in actions)
                    or not isinstance(physical_batches, list) or not physical_batches
                    or physical_batches[0] != len(ids)
                    or any(not isinstance(size, int) or not 1 <= size <= len(ids)
                           for size in physical_batches)):
                raise ValueError(f"incomplete or inconsistent OpenVLA-OFT episode: {path}")
            counts[task] += 1
            successes[task] += int(row["success"])

    for task in range(task_count):
        observed = {initialization for observed_task, initialization in seen if observed_task == task}
        if observed != set(range(len(observed))):
            raise ValueError(f"noncontiguous OpenVLA-OFT reset prefix for task {task}")
        if counts[task] and any(counts[previous] != episodes_per_task for previous in range(task)):
            raise ValueError(f"OpenVLA-OFT task {task} started before its predecessor completed")
    expected_total = task_count * episodes_per_task
    info_path = root / "official/eval_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        if (info.get("overall", {}).get("n_episodes") != sum(counts.values())
                or info.get("overall", {}).get("n_success") != sum(successes.values())):
            raise ValueError("OpenVLA-OFT raw episodes and aggregate metrics differ")
    return {
        "schema_version": 1,
        "evidence_tier": (
            "checkpoint_matched_probe" if expected_total < 500
            else "running_prefix_only_not_paired_completion"
        ),
        "model_id": "openvla_oft",
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

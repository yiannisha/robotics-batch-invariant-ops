"""Independent fail-closed audit for a running InternVLA LIBERO prefix."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
POLICY_HASHES = {
    "standard": "3c499b6979691b148bce55db85b30760bd40b3b8ed418203a66609ff34c0dd2c",
    "optimized": "e7bb04dc4a1cb83e5c032f8a02623da044b1c2621e9603f9e74d7a5bedee3572",
}
CHECKPOINT_SHA256 = "222b6ed5dbd3224f441ad2f9165838a890aad5934282209fb635ca4daf073de0"
VLM_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
OFFICIAL_EVALUATOR_SHA256 = "8090d19b19202a6a2241c8b521c8f6ce98a5557907cd7bbd17db68a75c883874"
OFFICIAL_BACKEND_SHA256 = "c78e11cf9b0c9ed6d74e260718168e347fd96e388c405f6a65270f246b0dcffe"
CLIENT_SHA256 = "b335a9347ef9709b6e719330b5aeb7f6d61b98608e98665eb11d03972ecad704"
SINGLETON_SERVER_SHA256 = "329cf78b3367555a0440616959ee8515e9490116530792b55db0ea3d1ef50bdf"
DYNAMIC_PADDING_SERVER_SHA256 = "7f125a360bd3a249369147bae7865cc110a75d61ad1abc50c95b6e0246152980"


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
        raise ValueError("unknown InternVLA condition")
    batch_size, repaired = CONDITIONS[condition]
    backend = manifest.get("backend")
    expected_model = f"internvla_a15_{backend}"
    required = {
        "schema_version": 1,
        "model": expected_model,
        "condition": condition,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vlm_sha256": VLM_SHA256,
        "official_evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
        "official_backend_sha256": OFFICIAL_BACKEND_SHA256,
        "official_policy_sha256": POLICY_HASHES.get(backend),
        "paired_wrapper_sha256": CLIENT_SHA256,
        # The completed singleton condition predates the VQA-label padding
        # fix.  That change is a no-op for physical B=1, while all genuinely
        # batched and subsequent conditions must pin the repaired wrapper.
        "shared_rollout_wrapper_sha256": (
            SINGLETON_SERVER_SHA256
            if backend == "standard" and condition == "native_singleton"
            else DYNAMIC_PADDING_SERVER_SHA256
        ),
        "episodes_per_task": 50,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "episode_length": 520,
        "num_steps_wait": 10,
        "replan_steps": 8,
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "genuine_batched_policy_call": True,
        "prompt_padding": "fixed_right_650" if repaired else "longest_right",
        "optimized_graph_cache_policy": (
            "recapture_per_policy_call" if backend == "optimized" else None
        ),
    }
    if backend not in POLICY_HASHES or any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("InternVLA manifest differs from the pinned evaluator contract")
    seed = manifest.get("seed")
    if not isinstance(seed, int):
        raise ValueError("InternVLA reset seed is not pinned")

    keys: set[tuple[int, int]] = set()
    counts = Counter()
    successes = Counter()
    noise_hashes: dict[int, set[str]] = defaultdict(set)
    files = sorted((root / "rollouts").glob("*.json"))
    files += sorted((root / "rollouts").glob("*.json.gz"))
    for path in files:
        payload = read_json(path)
        task = payload.get("task_id")
        ids = payload.get("explicit_init_state_ids")
        seeds = payload.get("seeds")
        episodes = payload.get("episodes")
        if (payload.get("task_group") != "libero_10" or task not in range(10)
                or not isinstance(ids, list) or not ids or len(ids) > batch_size
                or ids != list(range(ids[0], ids[0] + len(ids)))
                or any(index not in range(50) for index in ids)
                or seeds != [seed + index for index in ids]
                or not isinstance(episodes, list) or len(episodes) != len(ids)):
            raise ValueError(f"invalid InternVLA physical rollout group: {path}")
        if batch_size == 1 and len(ids) != 1:
            raise ValueError(f"singleton InternVLA rollout is physically batched: {path}")
        for episode, initialization, reset_seed in zip(episodes, ids, seeds):
            key = (task, initialization)
            if key in keys:
                raise ValueError(f"duplicate InternVLA task/reset: {key}")
            keys.add(key)
            states = episode.get("trajectory")
            actions = episode.get("action_sequence")
            rewards = episode.get("rewards")
            dones = episode.get("done")
            policy_batches = episode.get("policy_batch_sizes")
            if (episode.get("task") != task or episode.get("task_group") != "libero_10"
                    or episode.get("initialization_id") != initialization
                    or episode.get("reset_seed") != reset_seed
                    or not isinstance(episode.get("success"), bool)
                    or not valid_hash(episode.get("initial_observation_sha256"))
                    or not valid_hash(episode.get("first_noise_sha256"))
                    or episode["first_noise_sha256"] in noise_hashes[task]
                    or not isinstance(states, list) or not states
                    or not isinstance(actions, list) or not isinstance(rewards, list)
                    or not isinstance(dones, list)
                    or not len(states) == len(actions) == len(rewards) == len(dones)
                    or any(not finite_vector(state, 8) for state in states)
                    or any(not finite_vector(action, 7) for action in actions)
                    or not isinstance(policy_batches, list) or not policy_batches
                    or policy_batches[0] != len(ids)
                    or any(not isinstance(size, int) or not 1 <= size <= len(ids)
                           for size in policy_batches)):
                raise ValueError(f"incomplete or inconsistent InternVLA episode: {path}")
            noise_hashes[task].add(episode["first_noise_sha256"])
            counts[task] += 1
            successes[task] += int(episode["success"])

    for task in range(10):
        observed = {initialization for observed_task, initialization in keys if observed_task == task}
        if observed != set(range(len(observed))):
            raise ValueError(f"noncontiguous InternVLA initialization prefix for task {task}")
        if counts[task] and any(counts[previous] != 50 for previous in range(task)):
            raise ValueError(f"InternVLA task {task} started before task {task - 1} completed")
    return {
        "schema_version": 1,
        "evidence_tier": "running_prefix_only_not_paired_completion",
        "model_id": expected_model,
        "backend": backend,
        "condition": condition,
        "physical_batch_size_maximum": batch_size,
        "rollout_files": len(files),
        "episodes_cached": sum(counts.values()),
        "successes_cached": sum(successes.values()),
        "per_task_cached": {str(task): counts[task] for task in range(10)},
        "complete": all(counts[task] == 50 for task in range(10)),
        "integrity_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.condition_root), sort_keys=True))


if __name__ == "__main__":
    main()

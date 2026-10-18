"""Read-only fail-closed audit for a running UVA LIBERO paired condition."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = {
    "checkpoint_sha256": "4554b9bb2f4f3d44a7e51d5c7bd42a7740451874ac6cf1fea91e088a949f0dbf",
    "official_repo_commit": "fe3dec81af565fbb2d686072722f12340be31293",
    "official_evaluator_sha256": "566937e82f1aa8ed3b9b582a2a70cb5a0721e3102eb3f177c60c434b89808f9c",
    "official_policy_sha256": "aefc1acc772d5566f48c93390054d868eeda310b7288d1aac0c95d0b57079129",
    "official_libero_runner_sha256": "33e8d112120d253060166a5c0e1982c5b08dc7ea0e50845c6963c1017bfac7c0",
    "official_libero_environment_sha256": "868fee8b95a21d2c0c35ba7a836484b844e9db1cfdffef135b0d3825ab960df9",
    "paired_adapter_sha256": "864104b0a110495897de590fd1b56810628404f96610bc742d895d88c63e77a6",
    "paired_wrapper_sha256": "578cdd696488ef41713584e857fe87adf722791e2457bb684816f364583f973b",
    "rotation_conversions_sha256": "93735af31713ecf526cd4fb582a8914cf80bb242f4fb4d3ca82e35b6afd4d9ea",
}
CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
PACKAGES = {
    "transformers": "4.28.0",
    "tokenizers": "0.13.3",
    "huggingface_hub": "0.25.2",
    "zarr": "2.16.1",
    "numcodecs": "0.11.0",
    "gym": "0.21.0",
    "kornia": "0.8.0",
}


def require_hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} is not a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error


def load_json(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as stream:
                return json.load(stream)
        return json.loads(path.read_text())
    except Exception as error:
        raise ValueError(f"cannot read JSON evidence {path}: {error}") from error


def audit(condition_root: Path) -> dict[str, Any]:
    manifest_path = condition_root / "condition_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("UVA condition manifest is missing")
    manifest = load_json(manifest_path)
    for key, expected in EXPECTED.items():
        if manifest.get(key) != expected:
            raise ValueError(f"UVA manifest {key} differs from the pinned value")
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("UVA condition is unknown")
    batch_size, repaired = CONDITIONS[condition]
    path_kind = manifest.get("path")
    expected_model = "uva_joint" if path_kind == "joint" else "uva"
    if path_kind not in ("action", "joint") or manifest.get("model") != expected_model:
        raise ValueError("UVA model/path pairing is invalid")
    if (manifest.get("batch_size") != batch_size
            or manifest.get("invariant_ops") is not repaired
            or manifest.get("task_suite") != "libero_10"
            or manifest.get("episode_length") != 500
            or manifest.get("n_obs_steps") != 16
            or manifest.get("n_action_steps") != 8
            or manifest.get("control_mode") != "absolute"
            or manifest.get("explicit_init_state_by_initialization_id") is not True
            or manifest.get("request_local_noise") is not True
            or manifest.get("genuine_batched_policy_call") is not True
            or manifest.get("dependency_versions") != PACKAGES):
        raise ValueError("UVA manifest execution contract differs")
    episodes_per_task = manifest.get("episodes_per_task")
    task_ids = manifest.get("task_ids")
    if (not isinstance(episodes_per_task, int) or not 1 <= episodes_per_task <= 50
            or not isinstance(task_ids, list) or not task_ids
            or len(set(task_ids)) != len(task_ids)
            or any(type(task) is not int or task not in range(10) for task in task_ids)):
        raise ValueError("UVA manifest reset/task schedule is invalid")
    full = episodes_per_task == 50 and task_ids == list(range(10))
    expected_tier = "full_paired_condition" if full else "checkpoint_matched_probe"
    if manifest.get("evidence_tier") != expected_tier:
        raise ValueError("UVA evidence tier does not match its cardinality")

    seen: set[tuple[int, int]] = set()
    successes = 0
    rollout_files = sorted((condition_root / "rollouts").glob("*.json"))
    rollout_files += sorted((condition_root / "rollouts").glob("*.json.gz"))
    for rollout_path in rollout_files:
        payload = load_json(rollout_path)
        task = payload.get("task_id")
        ids = payload.get("explicit_init_state_ids")
        seeds = payload.get("seeds")
        episodes = payload.get("episodes")
        if (task not in task_ids or payload.get("task_group") != "libero_10"
                or not isinstance(ids, list) or not 1 <= len(ids) <= batch_size
                or ids != list(range(ids[0], ids[0] + len(ids)))
                or ids[0] % batch_size != 0
                or any(identifier not in range(episodes_per_task) for identifier in ids)
                or seeds != [manifest["seed"] + identifier for identifier in ids]
                or not isinstance(episodes, list) or len(episodes) != len(ids)):
            raise ValueError(f"UVA rollout group has an invalid schedule: {rollout_path}")
        suffix = ".json.gz" if rollout_path.suffix == ".gz" else ".json"
        expected_name = f"libero_10_{task}_initializations_{ids[0]}_{ids[-1]}{suffix}"
        if rollout_path.name != expected_name:
            raise ValueError(f"UVA rollout filename disagrees with payload: {rollout_path}")
        for index, episode in enumerate(episodes):
            identifier = ids[index]
            key = (task, identifier)
            if key in seen:
                raise ValueError(f"duplicate UVA episode {key}")
            seen.add(key)
            if (episode.get("task") != task
                    or episode.get("task_group") != "libero_10"
                    or episode.get("initialization_id") != identifier
                    or episode.get("reset_seed") != seeds[index]
                    or not isinstance(episode.get("task_description"), str)
                    or not episode["task_description"]):
                raise ValueError(f"UVA episode identity differs: {key}")
            require_hash(episode.get("initial_observation_sha256"), f"{key} observation")
            require_hash(episode.get("first_noise_sha256"), f"{key} noise")
            trajectory = episode.get("trajectory")
            actions = episode.get("action_sequence")
            rewards = episode.get("rewards")
            dones = episode.get("done")
            length = len(actions) if isinstance(actions, list) else -1
            if (not 1 <= length <= 500 or not isinstance(trajectory, list)
                    or not isinstance(rewards, list) or not isinstance(dones, list)
                    or len(trajectory) != length or len(rewards) != length
                    or len(dones) != length
                    or any(not isinstance(row, list) or len(row) != 8 for row in trajectory)
                    or any(not isinstance(row, list) or len(row) != 7 for row in actions)
                    or any(not all(math.isfinite(float(value)) for value in row)
                           for row in trajectory + actions)
                    or any(not math.isfinite(float(value)) for value in rewards)
                    or any(type(value) is not bool for value in dones)):
                raise ValueError(f"UVA episode trajectory is malformed: {key}")
            success = episode.get("success")
            if type(success) is not bool or success != any(float(value) == 1.0 for value in rewards):
                raise ValueError(f"UVA episode success does not match rewards: {key}")
            if success and (not dones[-1] or float(rewards[-1]) != 1.0):
                raise ValueError(f"successful UVA episode does not terminate on success: {key}")
            planned = math.ceil(length / 8)
            policy_batches = episode.get("policy_batch_sizes")
            inference_s = episode.get("inference_s")
            if (not isinstance(policy_batches, list) or len(policy_batches) != planned
                    or any(type(value) is not int or not 1 <= value <= len(ids)
                           for value in policy_batches)
                    or not isinstance(inference_s, list) or len(inference_s) != planned
                    or any(not math.isfinite(float(value)) or float(value) <= 0
                           for value in inference_s)):
                raise ValueError(f"UVA policy-call evidence is malformed: {key}")
            if path_kind == "joint":
                hashes = episode.get("prediction_video_sha256")
                if not isinstance(hashes, list) or len(hashes) != planned:
                    raise ValueError(f"UVA joint video hashes are incomplete: {key}")
                for position, value in enumerate(hashes):
                    require_hash(value, f"{key} video {position}")
            elif "prediction_video_sha256" in episode:
                raise ValueError(f"action-only UVA episode contains joint video evidence: {key}")
            successes += int(success)

    planned = {(task, initialization) for task in task_ids for initialization in range(episodes_per_task)}
    if not seen.issubset(planned):
        raise ValueError("UVA prefix contains an episode outside the manifest schedule")
    per_task = {
        str(task): sum(1 for observed_task, _ in seen if observed_task == task)
        for task in task_ids
    }
    complete = seen == planned
    summary_path = condition_root / "official/eval_info.json"
    if complete:
        if not summary_path.is_file():
            raise ValueError("complete UVA condition lacks official summary")
        summary = load_json(summary_path)
        overall = summary.get("overall", {})
        if (overall.get("n_episodes") != len(planned)
                or overall.get("n_success") != successes):
            raise ValueError("UVA summary differs from raw episodes")
    elif summary_path.exists():
        raise ValueError("incomplete UVA prefix must not claim a final summary")
    return {
        "schema_version": 1,
        "model": expected_model,
        "path": path_kind,
        "condition": condition,
        "integrity_passed": True,
        "evidence_tier": expected_tier if complete else "running_prefix_only_not_completion",
        "planned_episodes": len(planned),
        "episodes_cached": len(seen),
        "successes_cached": successes,
        "per_task_cached": per_task,
        "rollout_files": len(rollout_files),
        "complete": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.condition_root), sort_keys=True))


if __name__ == "__main__":
    main()

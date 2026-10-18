"""Audit physical reset identity across four LeRobot paired conditions.

This is independent of success outcomes. A condition set is paired only when
every requested task/initialization has the same initial physical observation
and request-local noise in all four execution alternatives.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


CONDITIONS = (
    "native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic",
)


def rollout_paths(directory: Path) -> list[Path]:
    return sorted((*directory.glob("*.json"), *directory.glob("*.json.gz")))


def read_rollout(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path, task_count: int = 10) -> dict:
    if task_count < 1:
        raise ValueError("task_count must be positive")
    by_condition = {}
    sources = {}
    protocol_versions = set()
    models = set()
    for condition in CONDITIONS:
        manifest = json.loads((root / condition / "condition_manifest.json").read_text())
        models.add(manifest.get("model"))
        protocol_version = manifest.get("paired_protocol_version", 1)
        protocol_versions.add(protocol_version)
        directory = root / condition / "rollouts"
        rows = {}
        paths = rollout_paths(directory)
        for path in paths:
            payload = read_rollout(path)
            for episode in payload.get("episodes", []):
                initialization = episode.get("initialization_id")
                if not isinstance(initialization, int) or initialization >= 50:
                    continue
                key = (episode.get("task"), initialization)
                if key in rows:
                    raise ValueError(f"duplicate reset {key} in {condition}")
                rows[key] = episode
        expected = {
            (task, initialization)
            for task in range(task_count) for initialization in range(50)
        }
        if set(rows) != expected:
            raise ValueError(
                f"{condition} has {len(rows)}/{50 * task_count} requested resets"
            )
        by_condition[condition] = rows
        sources[condition] = {
            "rollout_json_files": len(paths),
            "paired_protocol_version": protocol_version,
            "policy_batch_size": manifest.get("policy_batch_size", manifest.get("batch_size")),
            "environment_batch_size": manifest.get("environment_batch_size", manifest.get("batch_size")),
            "condition_manifest_sha256": sha256(root / condition / "condition_manifest.json"),
            "official_eval_info_sha256": sha256(root / condition / "official/eval_info.json"),
        }
    requires_fixed_environment_protocol = bool(
        models & {"xvla", "pi_fast", "smolvla", "molmoact2"}
    )
    protocol_isolation_valid = (
        protocol_versions == {2}
        and all(source["environment_batch_size"] == 4 for source in sources.values())
    ) if requires_fixed_environment_protocol else True
    raw_sensor_mismatches = []
    sensor_mismatches = []
    within_implementation_sensor_mismatches = []
    noise_mismatches = []
    same_batch_sensor_mismatches = []
    for key in sorted(by_condition[CONDITIONS[0]]):
        raw_sensors = {
            condition: by_condition[condition][key].get("initial_raw_observation_sha256")
            for condition in CONDITIONS
        }
        sensors = {condition: by_condition[condition][key]["initial_observation_sha256"]
                   for condition in CONDITIONS}
        noises = {condition: by_condition[condition][key]["first_noise_sha256"]
                  for condition in CONDITIONS}
        if requires_fixed_environment_protocol and (
            len(set(raw_sensors.values())) != 1 or not all(
                isinstance(value, str) and len(value) == 64 for value in raw_sensors.values()
            )
        ):
            raw_sensor_mismatches.append({"task": key[0], "initialization_id": key[1],
                                          "hashes": raw_sensors})
        if len(set(sensors.values())) != 1:
            sensor_mismatches.append({"task": key[0], "initialization_id": key[1],
                                      "hashes": sensors})
        if (sensors["native_singleton"] != sensors["native_dynamic"]
                or sensors["patched_singleton"] != sensors["patched_dynamic"]):
            within_implementation_sensor_mismatches.append({
                "task": key[0], "initialization_id": key[1], "hashes": sensors,
            })
        if len(set(noises.values())) != 1:
            noise_mismatches.append({"task": key[0], "initialization_id": key[1],
                                     "hashes": noises})
        if (sensors["native_singleton"] != sensors["patched_singleton"]
                or sensors["native_dynamic"] != sensors["patched_dynamic"]):
            same_batch_sensor_mismatches.append({"task": key[0], "initialization_id": key[1]})
    task_counts = Counter(item["task"] for item in sensor_mismatches)
    raw_task_counts = Counter(item["task"] for item in raw_sensor_mismatches)
    fully_paired = (
        protocol_isolation_valid
        and not raw_sensor_mismatches
        and not within_implementation_sensor_mismatches
        and not noise_mismatches
    )
    return {
        "schema_version": 2,
        "evidence_tier": "reset_pairing_audit_not_closed_loop_completion",
        "root": str(root.resolve()),
        "conditions": sources,
        "tasks": task_count,
        "requested_resets_per_condition": 50 * task_count,
        "protocol_isolation_valid": protocol_isolation_valid,
        "raw_initial_observation_mismatch_count": len(raw_sensor_mismatches),
        "raw_initial_observation_mismatches_by_task": {
            str(task): raw_task_counts[task] for task in sorted(raw_task_counts)
        },
        "initial_observation_mismatch_count": len(sensor_mismatches),
        "initial_observation_mismatches_by_task": {
            str(task): task_counts[task] for task in sorted(task_counts)
        },
        "request_noise_mismatch_count": len(noise_mismatches),
        "within_implementation_initial_observation_mismatch_count": len(
            within_implementation_sensor_mismatches
        ),
        "same_batch_native_patched_sensor_mismatch_count": len(same_batch_sensor_mismatches),
        "fully_paired": fully_paired,
        "raw_initial_observation_mismatches": raw_sensor_mismatches,
        "initial_observation_mismatches": sensor_mismatches,
        "within_implementation_initial_observation_mismatches": (
            within_implementation_sensor_mismatches
        ),
        "request_noise_mismatches": noise_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-count", type=int, default=10)
    args = parser.parse_args()
    report = audit(args.root, args.task_count)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "protocol_isolation_valid", "raw_initial_observation_mismatch_count",
        "raw_initial_observation_mismatches_by_task",
        "initial_observation_mismatch_count", "initial_observation_mismatches_by_task",
        "within_implementation_initial_observation_mismatch_count",
        "request_noise_mismatch_count", "same_batch_native_patched_sensor_mismatch_count",
        "fully_paired",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()

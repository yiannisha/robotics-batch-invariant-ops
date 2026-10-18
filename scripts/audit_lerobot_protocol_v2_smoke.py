"""Audit a focused serial-singleton versus dynamic LeRobot protocol-v2 smoke."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def rows(root: Path, expected_condition: str) -> dict[tuple[int, int], dict]:
    manifest = json.loads((root / "condition_manifest.json").read_text())
    expected_policy_batch = 1 if expected_condition.endswith("singleton") else 4
    expected_mode = "serial_singleton" if expected_policy_batch == 1 else "dynamic_batch"
    if (manifest.get("paired_protocol_version") != 2
            or manifest.get("condition") != expected_condition
            or manifest.get("policy_batch_size") != expected_policy_batch
            or manifest.get("environment_batch_size") != 4
            or manifest.get("execution_mode") != expected_mode):
        raise ValueError(f"{root} is not the expected protocol-v2 {expected_condition} condition")
    result = {}
    paths = sorted((*((root / "rollouts").glob("*.json")), *((root / "rollouts").glob("*.json.gz"))))
    for path in paths:
        for episode in read_json(path).get("episodes", []):
            key = (episode.get("task"), episode.get("initialization_id"))
            if key in result:
                raise ValueError(f"duplicate smoke request {key} in {root}")
            result[key] = episode
    if not result:
        raise ValueError(f"{root} contains no completed smoke requests")
    return result


def audit(singleton_root: Path, dynamic_root: Path) -> dict:
    singleton = rows(singleton_root, "native_singleton")
    dynamic = rows(dynamic_root, "native_dynamic")
    if set(singleton) != set(dynamic):
        raise ValueError("smoke conditions cover different task/initialization keys")
    comparisons = []
    for task, initialization in sorted(singleton):
        left = singleton[(task, initialization)]
        right = dynamic[(task, initialization)]
        left_action = left.get("action_sequence", [[]])[0]
        right_action = right.get("action_sequence", [[]])[0]
        if (not isinstance(left_action, list) or not isinstance(right_action, list)
                or len(left_action) != len(right_action) or not left_action):
            raise ValueError("smoke request lacks comparable first actions")
        maximum = max(abs(float(a) - float(b)) for a, b in zip(left_action, right_action))
        if not math.isfinite(maximum):
            raise ValueError("smoke action difference is nonfinite")
        comparisons.append({
            "task": task,
            "initialization_id": initialization,
            "raw_observation_match": (
                left.get("initial_raw_observation_sha256")
                == right.get("initial_raw_observation_sha256")
            ),
            "processed_observation_match": (
                left.get("initial_observation_sha256")
                == right.get("initial_observation_sha256")
            ),
            "request_noise_match": left.get("first_noise_sha256") == right.get("first_noise_sha256"),
            "first_action_exact": left_action == right_action,
            "first_action_max_abs": maximum,
        })
    isolation_passed = all(
        row["raw_observation_match"]
        and row["processed_observation_match"]
        and row["request_noise_match"]
        for row in comparisons
    )
    return {
        "schema_version": 1,
        "evidence_tier": "protocol_smoke_not_success_rate_result",
        "singleton_root": str(singleton_root.resolve()),
        "dynamic_root": str(dynamic_root.resolve()),
        "requests_compared": len(comparisons),
        "isolation_passed": isolation_passed,
        "first_actions_different": sum(not row["first_action_exact"] for row in comparisons),
        "maximum_first_action_abs_difference": max(
            row["first_action_max_abs"] for row in comparisons
        ),
        "comparisons": comparisons,
    }


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--singleton-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.singleton_root, args.dynamic_root)
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "requests_compared", "isolation_passed", "first_actions_different",
        "maximum_first_action_abs_difference",
    )}, sort_keys=True))
    if not report["isolation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

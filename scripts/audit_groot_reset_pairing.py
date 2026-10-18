"""Audit reset/noise pairing and outcome disagreements across GR00T conditions."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = (
    "native_singleton",
    "native_dynamic",
    "patched_singleton",
    "patched_dynamic",
)


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())
COMPARISONS = (
    ("native_singleton", "native_dynamic"),
    ("patched_singleton", "patched_dynamic"),
    ("native_singleton", "patched_singleton"),
    ("native_dynamic", "patched_dynamic"),
)


def load_condition(root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = {}
    paths = sorted((root / "rollouts").glob("*.json"))
    paths += sorted((root / "rollouts").glob("*.json.gz"))
    for path in paths:
        payload = read_json(path)
        for row in payload.get("episodes", []):
            key = (row.get("task"), row.get("initialization_id"))
            if not all(isinstance(value, int) for value in key) or key in rows:
                raise ValueError(f"invalid or duplicate GR00T key {key}: {path}")
            rows[key] = row
    return rows


def first_action_divergence(left: list, right: list) -> dict[str, Any]:
    common = min(len(left), len(right))
    first = None
    maximum = 0.0
    for index in range(common):
        lhs = np.asarray(left[index], dtype=np.float64)
        rhs = np.asarray(right[index], dtype=np.float64)
        if lhs.shape != (7,) or rhs.shape != (7,) or not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
            raise ValueError("GR00T action trajectory contains an invalid vector")
        error = float(np.max(np.abs(lhs - rhs)))
        maximum = max(maximum, error)
        if first is None and error != 0.0:
            first = index
    if first is None and len(left) != len(right):
        first = common
    return {
        "first_divergent_step": first,
        "maximum_common_step_abs_error": maximum,
        "left_steps": len(left),
        "right_steps": len(right),
    }


def audit(root: Path, require_complete: bool) -> dict[str, Any]:
    conditions = {condition: load_condition(root / condition) for condition in CONDITIONS}
    expected = {(task, initialization) for task in range(10) for initialization in range(50)}
    if require_complete and any(set(rows) != expected for rows in conditions.values()):
        raise ValueError("one or more GR00T conditions lack the exact 500-key schedule")

    all_keys = set().union(*(set(rows) for rows in conditions.values()))
    pairing_checks = 0
    for key in sorted(all_keys):
        available = [rows[key] for rows in conditions.values() if key in rows]
        anchors = {
            (
                row.get("reset_seed"),
                row.get("initial_observation_sha256"),
                row.get("first_noise_sha256"),
            )
            for row in available
        }
        if len(anchors) != 1:
            raise ValueError(f"GR00T reset/noise mismatch for paired key {key}")
        pairing_checks += max(0, len(available) - 1)

    comparisons = []
    for left_name, right_name in COMPARISONS:
        left, right = conditions[left_name], conditions[right_name]
        shared = sorted(set(left) & set(right))
        n10 = n01 = agreements = 0
        divergent = 0
        first_divergences = []
        maximum_error = 0.0
        for key in shared:
            left_success = left[key].get("success")
            right_success = right[key].get("success")
            if not isinstance(left_success, bool) or not isinstance(right_success, bool):
                raise ValueError(f"GR00T outcome is not boolean for {key}")
            if left_success and not right_success:
                n10 += 1
            elif right_success and not left_success:
                n01 += 1
            else:
                agreements += 1
            divergence = first_action_divergence(
                left[key].get("action_sequence", []), right[key].get("action_sequence", [])
            )
            if divergence["first_divergent_step"] is not None:
                divergent += 1
                first_divergences.append(divergence["first_divergent_step"])
            maximum_error = max(maximum_error, divergence["maximum_common_step_abs_error"])
        comparisons.append({
            "left": left_name,
            "right": right_name,
            "paired_episodes": len(shared),
            "outcome_agreements": agreements,
            "left_only_successes": n10,
            "right_only_successes": n01,
            "outcome_disagreements": n10 + n01,
            "trajectory_divergences": divergent,
            "earliest_action_divergence_step": min(first_divergences) if first_divergences else None,
            "maximum_action_abs_error": maximum_error,
        })

    if not all(math.isfinite(row["maximum_action_abs_error"]) for row in comparisons):
        raise ValueError("non-finite GR00T comparison metric")
    return {
        "schema_version": 1,
        "model_id": "groot_n17",
        "require_complete": require_complete,
        # This key is the shared contract consumed by the lossless compressor
        # and completed-evaluator promoter.  Prefix audits deliberately remain
        # false even when every currently observed reset is internally paired.
        "fully_paired": require_complete,
        "condition_episode_counts": {
            condition: len(rows) for condition, rows in conditions.items()
        },
        "pairing_checks": pairing_checks,
        "pairing_integrity_passed": True,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.paired_root, args.require_complete)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

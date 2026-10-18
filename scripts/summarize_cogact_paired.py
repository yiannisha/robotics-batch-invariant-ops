"""Reduce independently verified CogACT raw rollouts into paired results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = (
    "native_singleton",
    "native_dynamic",
    "patched_singleton",
    "patched_dynamic",
)
PAIRS = tuple(
    (left, right)
    for index, left in enumerate(CONDITIONS)
    for right in CONDITIONS[index + 1 :]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_bootstrap(differences: list[int], seed: int, resamples: int = 10_000) -> dict:
    generator = random.Random(seed)
    count = len(differences)
    means = [
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    return {
        "difference": statistics.fmean(differences),
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
        "method": "paired_percentile_bootstrap",
        "resamples": resamples,
        "seed": seed,
    }


def first_divergence(left: list[Any], right: list[Any]) -> int | None:
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def summarize(root: Path, task_manifest_path: Path) -> dict:
    root = root.resolve()
    task_manifest_path = task_manifest_path.resolve()
    task_manifest = json.loads(task_manifest_path.read_text())
    audit_path = root / "reset_pairing_audit.json"
    pairing_audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    if pairing_audit.get("fully_paired") is not True:
        raise RuntimeError("refusing to summarize CogACT before raw pairing passes")

    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    condition_manifests: dict[str, dict[str, str]] = {}
    condition_eval_info: dict[str, dict[str, str]] = {}
    for condition in CONDITIONS:
        folder = root / condition
        manifest_path = folder / "condition_manifest.json"
        info_path = folder / "official/eval_info.json"
        condition_manifests[condition] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        }
        condition_eval_info[condition] = {
            "path": str(info_path.resolve()),
            "sha256": sha256(info_path),
        }
        paths = sorted((folder / "rollouts").glob("*.json"))
        paths += sorted((folder / "rollouts").glob("*.json.gz"))
        for path in paths:
            payload = read_json(path)
            for episode in payload["episodes"]:
                key = (episode["task"], episode["initialization_id"])
                if condition in groups.setdefault(key, {}):
                    raise RuntimeError(f"duplicate CogACT episode {condition} {key}")
                row = dict(episode)
                row.update(
                    {
                        "condition": condition,
                        "source_rollout_path": str(path.resolve()),
                        "explicit_initialization_id": episode["initialization_id"],
                    }
                )
                groups[key][condition] = row

    expected = sum(task["initializations"] for task in task_manifest["tasks"])
    if len(groups) != expected or any(set(value) != set(CONDITIONS) for value in groups.values()):
        raise RuntimeError("CogACT paired reduction lacks full four-condition coverage")
    episodes_path = root / "episodes.jsonl.gz"
    lines = [
        json.dumps(groups[key][condition], sort_keys=True, separators=(",", ":"))
        for key in sorted(groups)
        for condition in CONDITIONS
    ]
    episodes_path.write_bytes(
        gzip.compress(("\n".join(lines) + "\n").encode(), compresslevel=6, mtime=0)
    )

    ordered_keys = sorted(groups)
    success_counts = {
        condition: sum(bool(groups[key][condition]["success"]) for key in ordered_keys)
        for condition in CONDITIONS
    }
    disagreement_counts: dict[str, int] = {}
    action_divergence_counts: dict[str, int] = {}
    trajectory_divergence_counts: dict[str, int] = {}
    paired_confidence_intervals: dict[str, dict] = {}
    divergence_details: dict[str, dict[str, Any]] = {}
    for pair_index, (left, right) in enumerate(PAIRS):
        name = f"{left}|{right}"
        differences = [
            int(groups[key][right]["success"]) - int(groups[key][left]["success"])
            for key in ordered_keys
        ]
        action_first = [
            first_divergence(
                groups[key][left]["action_sequence"], groups[key][right]["action_sequence"]
            )
            for key in ordered_keys
        ]
        trajectory_first = [
            first_divergence(groups[key][left]["trajectory"], groups[key][right]["trajectory"])
            for key in ordered_keys
        ]
        disagreement_counts[name] = sum(value != 0 for value in differences)
        action_divergence_counts[name] = sum(value is not None for value in action_first)
        trajectory_divergence_counts[name] = sum(value is not None for value in trajectory_first)
        paired_confidence_intervals[name] = paired_bootstrap(
            differences, 20261005 + pair_index
        )
        divergence_details[name] = {
            "first_action_divergence_step_by_reset": action_first,
            "first_trajectory_divergence_state_by_reset": trajectory_first,
        }

    per_task = []
    for task in task_manifest["tasks"]:
        keys = sorted(key for key in groups if key[0] == task["task_id"])
        per_task.append(
            {
                "task_id": task["task_id"],
                "task_name": task["name"],
                "initializations": len(keys),
                "success_counts": {
                    condition: sum(bool(groups[key][condition]["success"]) for key in keys)
                    for condition in CONDITIONS
                },
                "disagreement_counts": {
                    f"{left}|{right}": sum(
                        groups[key][left]["success"] != groups[key][right]["success"]
                        for key in keys
                    )
                    for left, right in PAIRS
                },
                "action_divergence_counts": {
                    f"{left}|{right}": sum(
                        groups[key][left]["action_sequence"]
                        != groups[key][right]["action_sequence"]
                        for key in keys
                    )
                    for left, right in PAIRS
                },
                "trajectory_divergence_counts": {
                    f"{left}|{right}": sum(
                        groups[key][left]["trajectory"] != groups[key][right]["trajectory"]
                        for key in keys
                    )
                    for left, right in PAIRS
                },
            }
        )

    summary = {
        "schema_version": 1,
        "evidence_tier": "full_paired_closed_loop",
        "model_id": "cogact_small",
        "conditions": list(CONDITIONS),
        "task_suite": task_manifest["task_suite"],
        "task_count": len(task_manifest["tasks"]),
        "initializations_per_condition": expected,
        "paired_episode_rows": len(lines),
        "episodes_path": str(episodes_path.resolve()),
        "episodes_sha256": sha256(episodes_path),
        "task_manifest_path": str(task_manifest_path),
        "task_manifest_sha256": sha256(task_manifest_path),
        "reset_pairing_audit_path": str(audit_path.resolve()),
        "reset_pairing_audit_sha256": sha256(audit_path),
        "condition_manifests": condition_manifests,
        "condition_eval_info": condition_eval_info,
        "success_counts": success_counts,
        "disagreement_counts": disagreement_counts,
        "action_divergence_counts": action_divergence_counts,
        "trajectory_divergence_counts": trajectory_divergence_counts,
        "paired_confidence_intervals": paired_confidence_intervals,
        "divergence_details": divergence_details,
        "per_task": per_task,
    }
    summary_path = root / "paired_summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return {
        "model_id": "cogact_small",
        "paired_episode_rows": len(lines),
        "episodes_sha256": summary["episodes_sha256"],
        "tasks": len(per_task),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.task_manifest), indent=2))


if __name__ == "__main__":
    main()

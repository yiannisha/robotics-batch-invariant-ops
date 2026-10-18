"""Independently audit full four-condition CogACT-Small reset/noise pairing."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from audit_cogact_paired_prefix import (
    CONDITIONS,
    TASK_MANIFEST,
    TASK_MANIFEST_SHA256,
    audit as audit_prefix,
    sha256,
)


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def audit(root: Path, task_manifest_path: Path = TASK_MANIFEST) -> dict[str, Any]:
    root = root.resolve()
    task_manifest_path = task_manifest_path.resolve()
    issues: list[str] = []
    if (
        task_manifest_path != TASK_MANIFEST.resolve()
        or not task_manifest_path.is_file()
        or sha256(task_manifest_path) != TASK_MANIFEST_SHA256
    ):
        issues.append("task manifest is not the pinned CogACT 12-task schedule")
        tasks: list[dict[str, Any]] = []
    else:
        task_manifest = json.loads(task_manifest_path.read_text())
        tasks = task_manifest.get("tasks", [])
        if (
            task_manifest.get("model_id") != "cogact_small"
            or task_manifest.get("task_count") != 12
            or task_manifest.get("initializations_per_condition") != 496
            or len(tasks) != 12
            or [task.get("initializations") for task in tasks]
            != [24, 24, 24, 24, 50, 50, 50, 50, 50, 50, 50, 50]
        ):
            issues.append("task manifest cardinality or model identity differs")
            tasks = []

    expected_pairs = {
        (task["task_id"], initialization)
        for task in tasks
        for initialization in range(task["initializations"])
    }
    cross_condition: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    condition_records: dict[str, dict[str, Any]] = {}
    for condition, (batch_size, _repaired) in CONDITIONS.items():
        folder = root / condition
        try:
            prefix = audit_prefix(folder)
        except Exception as error:
            issues.append(f"{condition}: prefix audit failed: {type(error).__name__}: {error}")
            prefix = {
                "integrity_passed": False,
                "complete": False,
                "episodes_cached": 0,
                "successes_cached": 0,
                "planned_episodes": 496,
            }
        if (
            prefix.get("integrity_passed") is not True
            or prefix.get("complete") is not True
            or prefix.get("evidence_tier") != "running_prefix_only_not_paired_completion"
            or prefix.get("planned_episodes") != 496
            or prefix.get("episodes_cached") != 496
        ):
            issues.append(f"{condition}: condition is not a complete audited 496-episode campaign")

        rows: dict[tuple[int, int], dict[str, Any]] = {}
        groups: set[tuple[int, tuple[int, ...]]] = set()
        rollout_paths = sorted((folder / "rollouts").glob("*.json"))
        rollout_paths += sorted((folder / "rollouts").glob("*.json.gz"))
        for rollout_path in rollout_paths:
            try:
                payload = read_json(rollout_path)
            except Exception as error:
                issues.append(f"{condition}: unreadable rollout {rollout_path}: {error}")
                continue
            task = payload.get("task_id")
            ids = payload.get("explicit_initialization_ids")
            if not isinstance(task, int) or not isinstance(ids, list):
                issues.append(f"{condition}: malformed physical group {rollout_path}")
                continue
            group = (task, tuple(ids))
            if group in groups:
                issues.append(f"{condition}: duplicate physical group {group}")
            groups.add(group)
            episodes = payload.get("episodes")
            if not isinstance(episodes, list):
                issues.append(f"{condition}: missing episode list {rollout_path}")
                continue
            for episode in episodes:
                key = (episode.get("task"), episode.get("initialization_id"))
                if key in rows:
                    issues.append(f"{condition}: duplicate episode {key}")
                    continue
                rows[key] = episode
                cross_condition.setdefault(key, {})[condition] = episode

        expected_groups = {
            (
                task["task_id"],
                tuple(range(start, min(start + batch_size, task["initializations"]))),
            )
            for task in tasks
            for start in range(0, task["initializations"], batch_size)
        }
        if groups != expected_groups:
            issues.append(
                f"{condition}: physical group schedule differs "
                f"(observed={len(groups)}, expected={len(expected_groups)})"
            )
        if set(rows) != expected_pairs:
            issues.append(
                f"{condition}: reset coverage differs "
                f"(observed={len(rows)}, expected={len(expected_pairs)})"
            )
        condition_records[condition] = {
            "physical_groups": len(groups),
            "expected_physical_groups": len(expected_groups),
            "episodes": len(rows),
            "successes": sum(bool(row.get("success")) for row in rows.values()),
            "condition_manifest_path": str((folder / "condition_manifest.json").resolve()),
            "condition_manifest_sha256": (
                sha256(folder / "condition_manifest.json")
                if (folder / "condition_manifest.json").is_file()
                else None
            ),
            "official_eval_info_path": str((folder / "official/eval_info.json").resolve()),
            "official_eval_info_sha256": (
                sha256(folder / "official/eval_info.json")
                if (folder / "official/eval_info.json").is_file()
                else None
            ),
        }

    condition_set = set(CONDITIONS)
    reset_mismatches = 0
    noise_mismatches = 0
    for key in sorted(expected_pairs):
        group = cross_condition.get(key, {})
        if set(group) != condition_set:
            issues.append(f"paired reset {key} lacks all four conditions")
            continue
        if (
            len({row.get("initial_observation_sha256") for row in group.values()}) != 1
            or len({row.get("reset_seed") for row in group.values()}) != 1
        ):
            reset_mismatches += 1
            issues.append(f"paired reset {key} differs physically across conditions")
        noises = [row.get("noise_sha256") for row in group.values()]
        if any(not isinstance(value, list) for value in noises) or any(
            value != noises[0] for value in noises[1:]
        ):
            noise_mismatches += 1
            issues.append(f"paired request-local noise schedule differs at {key}")

    report = {
        "schema_version": 1,
        "model_id": "cogact_small",
        "evidence_tier": "independent_raw_pairing_audit",
        "task_manifest_path": str(task_manifest_path),
        "task_manifest_sha256": (
            sha256(task_manifest_path) if task_manifest_path.is_file() else None
        ),
        "expected_resets_per_condition": len(expected_pairs),
        "conditions": condition_records,
        "initial_observation_or_seed_mismatch_count": reset_mismatches,
        "request_local_noise_mismatch_count": noise_mismatches,
        "fully_paired": not issues,
        "issues": issues,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, default=TASK_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.root, args.task_manifest)
    output = args.output or args.root / "reset_pairing_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "issues"}, indent=2))
    if report["issues"]:
        print(json.dumps({"issues": report["issues"][:20]}, indent=2))
    raise SystemExit(0 if report["fully_paired"] else 1)


if __name__ == "__main__":
    main()

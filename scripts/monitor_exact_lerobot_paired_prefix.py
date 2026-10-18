"""Continuously audit UniVLA/MolmoAct2 paired LIBERO rollout prefixes."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from audit_exact_lerobot_paired_prefix import audit, audit_readiness


CONDITIONS = (
    "native_singleton",
    "native_dynamic",
    "patched_singleton",
    "patched_dynamic",
)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def rollout_count(condition_root: Path) -> int:
    rollouts = condition_root / "rollouts"
    return sum(1 for _ in rollouts.glob("*.json")) + sum(
        1 for _ in rollouts.glob("*.json.gz")
    )


def validate_due_prefixes(
    campaign_root: Path,
    step: int,
    audit_fn: Callable[[Path], dict] = audit,
) -> dict:
    model_id = campaign_root.parents[2].name
    try:
        readiness = audit_readiness(model_id)
        readiness_error = None
    except Exception as error:
        readiness = {
            "model_id": model_id,
            "exact_checkpoint_ready": False,
            "pinned_provenance_files_verified": 0,
            "sealed_checkpoint_members_verified": 0,
        }
        readiness_error = f"{type(error).__name__}: {error}"
    conditions = {}
    for condition in CONDITIONS:
        root = campaign_root / condition
        count = rollout_count(root)
        manifest_present = (root / "condition_manifest.json").is_file()
        prior_path = root / "prefix_audit.json"
        try:
            prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            prior = {}
        validated_files = prior.get("rollout_files", 0)
        due = count == 130 or count // step > validated_files // step
        if manifest_present and count and due:
            result = audit_fn(root)
            result["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(prior_path, result)
            validated_files = result["rollout_files"]
            planned_episodes = result["planned_episodes_cached"]
            integrity_passed = result.get("integrity_passed") is True
            provenance_passed = result.get("provenance_integrity_passed") is True
        else:
            planned_episodes = prior.get("planned_episodes_cached", 0)
            integrity_passed = prior.get("integrity_passed") is True
            provenance_passed = prior.get("provenance_integrity_passed") is True
        conditions[condition] = {
            "manifest_present": manifest_present,
            "raw_rollout_files": count,
            "validated_rollout_files": validated_files,
            "validated_planned_episodes": planned_episodes,
            "integrity_passed": integrity_passed,
            "provenance_integrity_passed": provenance_passed,
            "complete": planned_episodes == 500 and integrity_passed and provenance_passed,
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitor_pid": os.getpid(),
        "validation_step_rollout_files": step,
        "campaign_root": str(campaign_root.resolve()),
        "checkpoint_readiness": readiness,
        "checkpoint_readiness_error": readiness_error,
        "conditions": conditions,
        "all_conditions_complete": all(row["complete"] for row in conditions.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, action="append", required=True)
    parser.add_argument("--step-rollout-files", type=int, default=7)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.step_rollout_files < 1 or args.interval_seconds <= 0:
        raise ValueError("monitor step and interval must be positive")
    while True:
        states = []
        for campaign_root in args.campaign_root:
            state = validate_due_prefixes(campaign_root, args.step_rollout_files)
            write_json_atomic(campaign_root / "prefix_monitor_state.json", state)
            states.append(state)
        print(json.dumps(states, sort_keys=True), flush=True)
        if args.once or all(state["all_conditions_complete"] for state in states):
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

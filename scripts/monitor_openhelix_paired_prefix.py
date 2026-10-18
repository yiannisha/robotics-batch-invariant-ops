"""Periodically run the independent OpenHelix prefix audit during the campaign."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.audit_openhelix_paired_prefix import audit


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


def count_rollouts(root: Path) -> int:
    return sum(1 for _ in (root / "rollouts").glob("sequence_*.json.gz"))


def validate_due_prefixes(paired: Path, step: int) -> dict:
    conditions = {}
    for condition in CONDITIONS:
        root = paired / condition
        count = count_rollouts(root)
        prior_path = root / "prefix_audit.json"
        try:
            prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            prior = {}
        validated = prior.get("sequences_validated", 0)
        due = count == 1000 or count // step > validated // step
        if count and due:
            result = audit(root)
            result["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(prior_path, result)
            validated = result["sequences_validated"]
        conditions[condition] = {
            "raw_sequences": count,
            "validated_sequences": validated,
            "complete": validated == 1000,
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitor_pid": os.getpid(),
        "validation_step": step,
        "conditions": conditions,
        "all_conditions_complete": all(
            row["complete"] for row in conditions.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.step < 1 or args.interval_seconds <= 0:
        raise ValueError("monitor step and interval must be positive")
    state_path = args.paired / "prefix_monitor_state.json"
    while True:
        state = validate_due_prefixes(args.paired, args.step)
        write_json_atomic(state_path, state)
        print(json.dumps(state, sort_keys=True), flush=True)
        if args.once or state["all_conditions_complete"]:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

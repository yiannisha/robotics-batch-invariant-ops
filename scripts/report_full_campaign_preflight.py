"""Inventory current artifacts and raw full-run progress without claiming completion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_full_cross_model_campaign import CONFIGURATIONS


def timing_counts(path: Path) -> dict:
    if not path.is_file():
        return {"timed": 0, "warmups": 0, "environment_records": 0}
    counts = Counter()
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            counts[item.get("record_type")] += 1
    return {
        "timed": counts["timing"],
        "warmups": counts["warmup"],
        "environment_records": counts["environment"],
    }


def inspect(root: Path, identifier: str) -> dict:
    directory = root / identifier
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    artifacts = provenance.get("artifacts", {})
    missing = [
        {"name": name, "path": item["path"], "expected_bytes": item.get("bytes")}
        for name, item in artifacts.items()
        if isinstance(item, dict) and item.get("path") and not Path(item["path"]).is_file()
    ]
    full = directory / "full"
    return {
        "id": identifier,
        "pilot_provenance_present": bool(provenance),
        "adapter": provenance.get("adapter"),
        "missing_pinned_artifacts": missing,
        "frozen_replay_manifest_present": (full / "replay_manifest.json").is_file(),
        "numerical_freeze_present": (full / "numerical_freeze.json").is_file(),
        "policy_timing": timing_counts(full / "performance" / "policy_timings.jsonl"),
        "four_mode_timing": timing_counts(full / "performance" / "serving_timings.jsonl"),
        "heldout_raw_present": any((full / "heldout" / filename).is_file() for filename in ("invariance.jsonl", "invariance.jsonl.gz")),
        "operator_raw_present": (full / "operator" / "raw.jsonl").is_file(),
        "queue_raw_present": (full / "serving" / "requests.jsonl").is_file(),
        "evaluator_attempt_present": (full / "closed_loop" / "evaluator_attempt.json").is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    parser.add_argument("--output", type=Path, default=Path("results/cross_model/full_preflight.json"))
    args = parser.parse_args()
    paths = [inspect(args.results, identifier) for identifier in CONFIGURATIONS]
    report = {
        "schema_version": 1,
        "evidence_tier": "preflight_only",
        "warning": "This inventory is not the fail-closed full-coverage audit.",
        "configuration_count": len(paths),
        "paths_missing_local_pinned_artifacts": sum(bool(item["missing_pinned_artifacts"]) for item in paths),
        "paths_with_frozen_replay_manifest": sum(item["frozen_replay_manifest_present"] for item in paths),
        "raw_policy_timed_calls": sum(item["policy_timing"]["timed"] for item in paths),
        "raw_four_mode_timed_calls": sum(item["four_mode_timing"]["timed"] for item in paths),
        "configurations": paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "configurations"}, indent=2))


if __name__ == "__main__":
    main()

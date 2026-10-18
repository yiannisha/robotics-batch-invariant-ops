"""Render progress-only per-path and aggregate views of the independent gate.

These files are navigation aids, never substitutes for full_coverage_audit.json.
Only an audit component with ``complete: true`` is labeled complete. Pilot
counts and running evaluator rollouts remain partial evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


COUNT_FIELDS = frozenset({
    "timed", "warmups", "arrangements", "fidelity", "records", "profiles",
    "requests", "configurations", "comparisons", "episodes",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_raw_progress(value: object) -> bool:
    if isinstance(value, dict):
        if any(isinstance(value.get(key), int) and not isinstance(value[key], bool)
               and value[key] > 0 for key in COUNT_FIELDS):
            return True
        return any(has_raw_progress(child) for child in value.values())
    if isinstance(value, list):
        return any(has_raw_progress(child) for child in value)
    return False


def evaluator_rollout_count(directory: Path) -> dict[str, int]:
    paired = directory / "closed_loop" / "paired"
    return {
        condition: len(list((paired / condition / "rollouts").glob("*.json")))
        for condition in (
            "native_singleton", "native_dynamic",
            "patched_singleton", "patched_dynamic",
        )
    }


def component_status(name: str, component: dict, directory: Path) -> str:
    if name == "closed_loop":
        status = component.get("status")
        if status == "not_applicable" and component.get("complete"):
            return "not_applicable"
        if status == "blocked":
            return "blocked_accepted" if component.get("complete") else "blocked_unaccepted"
        if status == "not_applicable":
            return "not_applicable_unaccepted"
        if component.get("complete"):
            return "complete"
        if status in ("probe_succeeded_requires_paired_campaign", "completed"):
            return "partial"
        if any(evaluator_rollout_count(directory).values()):
            return "partial"
        return "missing"
    if component.get("complete"):
        return "complete"
    return "partial" if has_raw_progress(component) else "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("results/cross_model/full_coverage_audit.json"))
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    args = parser.parse_args()
    if not args.audit.is_file():
        raise FileNotFoundError("run the independent full-coverage audit first")
    audit = json.loads(args.audit.read_text())
    records = audit.get("configurations", [])
    if (audit.get("schema_version") != 1 or audit.get("configuration_count") != 29
            or len(records) != 29 or len({row.get("id") for row in records}) != 29):
        raise ValueError("expected the independent 29-path full-coverage audit")
    component_names = sorted({name for row in records for name in row.get("components", {})})
    audit_hash = sha256(args.audit)
    generated_at = datetime.now(timezone.utc).isoformat()
    paths = []
    for row in records:
        identifier = row["id"]
        directory = args.results / identifier / "full"
        statuses = {
            name: component_status(name, row.get("components", {}).get(name, {}), directory)
            for name in component_names
        }
        path_report = {
            "schema_version": 1,
            "evidence_tier": "progress_only_not_full_coverage",
            "generated_at_utc": generated_at,
            "independent_audit_path": str(args.audit.resolve()),
            "independent_audit_sha256": audit_hash,
            "model_id": identifier,
            "full_audit_complete": row.get("complete") is True,
            "component_status": statuses,
            "issue_count": len(row.get("issues", [])),
            "active_paired_rollouts_by_condition": evaluator_rollout_count(directory),
        }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "status.json").write_text(json.dumps(path_report, indent=2) + "\n")
        paths.append(path_report)
    aggregate = {
        "schema_version": 1,
        "evidence_tier": "progress_only_not_full_coverage",
        "generated_at_utc": generated_at,
        "independent_audit_path": str(args.audit.resolve()),
        "independent_audit_sha256": audit_hash,
        "configuration_count": len(paths),
        "full_audit_complete": sum(row["full_audit_complete"] for row in paths),
        "component_status_counts": {
            name: dict(Counter(row["component_status"][name] for row in paths))
            for name in component_names
        },
        "paths": paths,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "full_coverage_status.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    with (args.results / "full_coverage_status.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_id", "full_audit_complete", "issue_count", *component_names])
        for row in paths:
            writer.writerow([
                row["model_id"], row["full_audit_complete"], row["issue_count"],
                *(row["component_status"][name] for name in component_names),
            ])
    print(json.dumps({
        "full_audit_complete": aggregate["full_audit_complete"],
        "configuration_count": aggregate["configuration_count"],
        "independent_audit_sha256": audit_hash,
    }))


if __name__ == "__main__":
    main()

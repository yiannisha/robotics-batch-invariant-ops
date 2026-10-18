"""Retired pilot audit; use audit_full_cross_model_campaign.py instead."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from audit_cross_model_campaign import CONFIGURATIONS, inspect


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raise SystemExit(
        "This pilot audit was invalidated on 2026-10-05: it accepted reduced, "
        "self-declared schedules. Run scripts/audit_full_cross_model_campaign.py."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    parser.add_argument("--output", type=Path, default=Path("results/cross_model/final_audit.json"))
    args = parser.parse_args()
    root = args.results
    failures: list[str] = []
    core = [inspect(root, identifier, name) for identifier, name in CONFIGURATIONS]
    check(len(core) == 29, "expected 29 audited configurations", failures)
    for record in core:
        check(record["measurement_core_complete"], f"{record['id']}: measurement core incomplete", failures)
        check(record["fixed_end_to_end_exact"] is True, f"{record['id']}: repaired path not exact", failures)
        check(record["invariant_transformations_exact"] is True, f"{record['id']}: transformations not exact", failures)
        check(record["operator_benchmark_sessions"] >= 5, f"{record['id']}: fewer than five operator sessions", failures)
        per_model_report = root / record["id"] / "REPORT.md"
        check(
            per_model_report.is_file() and per_model_report.stat().st_size > 500,
            f"{record['id']}: per-configuration report missing or empty",
            failures,
        )

    aggregate_path = root / "aggregate_results.json"
    aggregate = json.loads(aggregate_path.read_text())
    configurations = aggregate["configurations"]
    check(len(configurations) == 29, "aggregate JSON does not contain 29 configurations", failures)
    check(aggregate["aggregates"]["measurement_core_complete"] == 29, "aggregate core count is not 29", failures)
    check(aggregate["aggregates"]["fixed_end_to_end_exact"] == 29, "aggregate exact count is not 29", failures)
    check(aggregate["aggregates"]["fixed_transformations_exact"] == 29, "aggregate transformation count is not 29", failures)
    check(aggregate["aggregates"]["operator_measurement_count"] > 0, "aggregate has no operator measurements", failures)
    check({record["id"] for record in configurations} == {item[0] for item in CONFIGURATIONS}, "aggregate configuration IDs differ from audit scope", failures)

    with (root / "aggregate_results.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    check(len(csv_rows) == 29, "aggregate CSV does not contain 29 rows", failures)
    for relative in (
        "CAMPAIGN_REPORT.md",
        "figures/policy_b1_latency_ratio.svg",
        "figures/serving_latency_ratio.svg",
    ):
        path = root / relative
        check(path.is_file() and path.stat().st_size > 500, f"missing or empty deliverable: {relative}", failures)

    closed_loop_path = root / "closed_loop" / "applicability.json"
    closed_loop = json.loads(closed_loop_path.read_text())
    check(closed_loop["configuration_count"] == 29, "closed-loop audit does not contain 29 configurations", failures)
    check(not closed_loop["missing_evidence"], "closed-loop audit has missing source evidence", failures)
    check(
        closed_loop["applicable_public_simulator"] + closed_loop["not_applicable"] == 29,
        "closed-loop applicability counts do not partition scope",
        failures,
    )
    for record in closed_loop["configurations"]:
        if record["public_simulator"]:
            satisfied = record["paired_episodes_completed"] > 0 or (
                record["status"] == "BLOCKED BEFORE EPISODE 1" and bool(record["blocker"])
            )
            check(satisfied, f"{record['id']}: applicable closed-loop path has neither episodes nor a blocker", failures)
        else:
            check(record["status"].startswith("NOT APPLICABLE"), f"{record['id']}: unmatched evaluator not marked N/A", failures)

    probe_path = root / "closed_loop" / "rdt_maniskill_probe.json"
    probe = json.loads(probe_path.read_text())
    check(probe["graphics_environment"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility", "graphics capability evidence changed", failures)
    check(probe["simulator_probe"]["started"] is False, "SAPIEN probe unexpectedly marked started", failures)
    check("ErrorIncompatibleDriver" in probe["simulator_probe"]["stderr"], "SAPIEN blocker signature missing", failures)
    check(probe["mujoco_egl_probe"]["started"] is False, "MuJoCo probe unexpectedly marked started", failures)
    check("eglQueryString" in probe["mujoco_egl_probe"]["stderr"], "MuJoCo EGL blocker signature missing", failures)

    artifact_paths = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path not in {args.output, root / "archive_manifest.json"}
        and "__pycache__" not in path.parts
    )
    manifest = {
        "schema_version": 1,
        "file_count": len(artifact_paths),
        "files": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        ],
    }
    (root / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "configuration_count": len(core),
        "measurement_core_complete": sum(record["measurement_core_complete"] for record in core),
        "aggregate_sha256": sha256(aggregate_path),
        "closed_loop_applicability_sha256": sha256(closed_loop_path),
        "runtime_probe_sha256": sha256(probe_path),
        "archive_manifest_file_count": manifest["file_count"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()

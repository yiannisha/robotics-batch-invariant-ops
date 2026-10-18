"""Generate a path report only after every non-report audit component passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_full_cross_model_campaign import (
    POLICY_MODES,
    audit_configuration,
    capacity_audit,
    large_serving_shapes,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def required_artifact_paths(directory: Path, capacity: dict) -> list[Path]:
    paths = [
        directory / "replay_manifest.json",
        directory / "numerical_freeze.json",
        directory / "diagnostic/invariance.jsonl.gz",
        directory / "diagnostic/singleton_fidelity.jsonl.gz",
        directory / "diagnostic/causal_ablation.jsonl",
        directory / "heldout/invariance.jsonl.gz",
        directory / "heldout/singleton_fidelity.jsonl.gz",
        directory / "heldout/summary.json",
        directory / "heldout/flow_summary.json",
        directory / "operator/raw.jsonl",
        directory / "performance/policy_timings.jsonl",
        directory / "performance/serving_timings.jsonl",
        directory / "serving/requests.jsonl",
        directory / "closed_loop/evaluator_attempt.json",
        directory / "closed_loop/paired/paired_summary.json",
    ]
    feasible = capacity["feasible"]
    paths.extend(
        directory / f"performance/policy_{mode}_B{size}_timings.jsonl"
        for size in (16, 32)
        for mode in POLICY_MODES
        if feasible.get(f"{mode}|B{size}")
    )
    paths.extend(
        path
        for _, fixed in large_serving_shapes(capacity)
        for path in (
            directory / f"performance/serving_fixed_B{fixed}_timings.jsonl",
            directory / f"serving/requests_fixed_B{fixed}.jsonl",
        )
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    args = parser.parse_args()
    directory = args.results / args.model_id / "full"
    initial = audit_configuration(args.results, args.model_id)
    substantive = [problem for problem in initial["issues"]
                   if problem != "report: missing full campaign report"]
    if substantive:
        raise RuntimeError("refusing report before experiment completion:\n- " + "\n- ".join(substantive))

    capacity_issues: list[str] = []
    capacity = capacity_audit(
        directory / "diagnostic/large_batch_capacity.jsonl", directory, capacity_issues,
    )
    if not capacity.get("complete"):
        raise RuntimeError(f"capacity evidence changed after the path audit: {capacity_issues}")
    artifact_paths = required_artifact_paths(directory, capacity)
    if any(not path.is_file() for path in artifact_paths):
        missing = [str(path) for path in artifact_paths if not path.is_file()]
        raise FileNotFoundError(f"audited report artifacts disappeared: {missing}")
    artifacts = [{
        "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
    } for path in artifact_paths]
    report = {
        "schema_version": 1,
        "evidence_tier": "full_pi0_equivalent_coverage",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "independent_non_report_audit_passed": True,
        "components": initial["components"],
        "heldout_summary": read(directory / "heldout/summary.json"),
        "flow_summary": read(directory / "heldout/flow_summary.json"),
        "closed_loop_summary": read(directory / "closed_loop/paired/paired_summary.json"),
        "serving_summary": read(directory / "serving/summary.json"),
        "large_serving_summaries": {
            f"logical_B{logical}|fixed_B{fixed}": read(
                directory / f"serving/summary_fixed_B{fixed}.json"
            )
            for logical, fixed in large_serving_shapes(capacity)
        },
        "artifacts": artifacts,
    }
    json_path = directory / "report.json"
    json_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    completed = [name for name, value in initial["components"].items() if value.get("complete")]
    markdown = [
        f"# {args.model_id} full π0-equivalent campaign",
        "",
        "This report was generated only after every non-report component passed the independent fail-closed audit.",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Completed audited components: {len(completed)}",
        f"- Machine-readable report: `{json_path.name}`",
        "",
        "## Audited components",
        "",
        *[f"- `{name}`: complete" for name in completed],
        "",
        "## Evidence",
        "",
        *[f"- `{item['path']}` — {item['bytes']} bytes — `{item['sha256']}`"
          for item in artifacts],
        "",
        "Numerical outcomes, flow-step statistics, serving measurements, and paired success-rate results are retained verbatim in the machine-readable report and pinned artifacts above.",
        "",
    ]
    (directory / "REPORT.md").write_text("\n".join(markdown))
    final = audit_configuration(args.results, args.model_id)
    if not final.get("complete"):
        raise RuntimeError(f"report written but independent path audit still fails: {final['issues']}")
    print(json.dumps({"model_id": args.model_id, "full_path_audit": "passed",
                      "report": str((directory / 'REPORT.md').resolve())}))


if __name__ == "__main__":
    main()

"""Fail-closed aggregate audit for the closed-loop success-rate campaign only.

This gate deliberately excludes offline numerical, timing, and serving evidence.
A path passes only with either a complete four-condition paired campaign, a
fail-closed no-matched-evaluator record, or a precise evidence-backed blocker.

The full campaign gate intentionally accepts only permanent evaluator
exceptions.  That is too strong for this narrower gate: a missing public asset
or checkpoint can make the experiment impossible in the current execution
environment without being permanent in principle.  Such a blocker is accepted
here only when a real failed preflight/probe, immutable logs, pinned official
source, pinned investigation evidence, and an explicit resolution are all
present.  Merely writing ``blocked`` never satisfies this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_cross_model_campaign import CONFIGURATIONS
from audit_full_cross_model_campaign import evaluator_audit


BLOCKER_TYPES = {
    "missing_exact_checkpoint",
    "missing_evaluator_runtime",
    "missing_evaluator_assets",
    "incompatible_evaluator_runtime",
    "resource_capacity",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        advised_through = 0
        try:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                position = stream.tell()
                if position - advised_through >= 64 * 1024 * 1024:
                    try:
                        os.posix_fadvise(
                            stream.fileno(), advised_through,
                            position - advised_through, os.POSIX_FADV_DONTNEED,
                        )
                        advised_through = position
                    except (AttributeError, OSError):
                        pass
        finally:
            # This audit can hash tens of gigabytes of resident checkpoints.
            # Release only the resulting clean page-cache pages so verification
            # cannot evict or OOM the concurrently running evaluator.
            try:
                os.posix_fadvise(
                    stream.fileno(), advised_through, 0, os.POSIX_FADV_DONTNEED
                )
            except (AttributeError, OSError):
                pass
    return digest.hexdigest()


def _pinned_file(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    path_value = item.get("path")
    expected = item.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        return False
    path = Path(path_value)
    return path.is_file() and _sha256(path) == expected


def evidence_backed_blocker_audit(path: Path, identifier: str, issues: list[str]) -> dict:
    """Validate a current-environment evaluator blocker without calling it permanent."""

    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        issues.append(f"closed loop: missing or malformed evaluator blocker {path}")
        return {"complete": False, "status": "missing"}

    closed_loop = path.parent
    full = closed_loop.parent
    raw_rollouts = [
        candidate
        for candidate in (closed_loop / "paired").glob("*/rollouts/*")
        if candidate.is_file() and candidate.suffix in {".json", ".gz"}
    ]
    if raw_rollouts:
        issues.append(
            f"closed loop: stale blocker is contradicted by raw paired rollouts for {identifier}"
        )

    if record.get("model_id") != identifier or record.get("status") != "blocked":
        issues.append(f"closed loop: evaluator blocker belongs to another model/status: {identifier}")
    if record.get("blocker_type") not in BLOCKER_TYPES:
        issues.append(f"closed loop: unsupported or missing blocker type for {identifier}")
    command = record.get("attempted_command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
        or not isinstance(record.get("exit_code"), int)
        or record.get("exit_code") == 0
        or not record.get("attempt_timestamp_utc")
    ):
        issues.append(f"closed loop: no failed safe preflight/probe recorded for {identifier}")

    logs = [record.get("stdout_path"), record.get("stderr_path")]
    log_paths = [Path(value) for value in logs if isinstance(value, str)]
    if len(log_paths) != 2 or any(not item.is_file() for item in log_paths):
        issues.append(f"closed loop: blocker lacks immutable stdout/stderr files for {identifier}")
    elif not any(item.read_text(errors="replace").strip() for item in log_paths):
        issues.append(f"closed loop: blocker preflight/probe emitted no evidence for {identifier}")

    if not record.get("failure_class") or not record.get("evaluator"):
        issues.append(f"closed loop: blocker classification/evaluator missing for {identifier}")
    if record.get("probe_episodes_completed") != 0 or record.get("paired_episodes_completed") != 0:
        issues.append(f"closed loop: blocked record has inconsistent episode counts for {identifier}")

    source = {
        "path": record.get("source_evidence_path"),
        "sha256": record.get("source_evidence_sha256"),
    }
    if not _pinned_file(source):
        issues.append(f"closed loop: blocker lacks pinned official evaluator source for {identifier}")

    investigation = record.get("blocker_investigation")
    if not isinstance(investigation, dict):
        investigation = {}
    if investigation.get("safe_probe_exhausted") is not True:
        issues.append(f"closed loop: blocker does not prove safe probing was exhausted for {identifier}")
    # Permanence is an assertion about the future and is not required.  Its
    # value must still be explicit so reports cannot silently imply it.
    if not isinstance(investigation.get("reason_permanent"), bool):
        issues.append(f"closed loop: blocker permanence is not explicitly classified for {identifier}")
    attempts = investigation.get("resolution_attempts")
    if not isinstance(attempts, list) or not attempts or not all(
        isinstance(value, str) and value.strip() for value in attempts
    ):
        issues.append(f"closed loop: blocker lacks concrete resolution attempts for {identifier}")
    if not isinstance(investigation.get("required_resolution"), str) or not investigation[
        "required_resolution"
    ].strip():
        issues.append(f"closed loop: blocker lacks an explicit required resolution for {identifier}")
    evidence = investigation.get("evidence_files")
    if not isinstance(evidence, list) or not evidence or not all(_pinned_file(item) for item in evidence):
        issues.append(f"closed loop: blocker lacks pinned investigation evidence for {identifier}")

    availability = record.get("checkpoint_availability")
    if availability == "resident":
        checkpoint_files = record.get("checkpoint_files")
        if (
            not isinstance(checkpoint_files, list)
            or not checkpoint_files
            or not all(_pinned_file(item) for item in checkpoint_files)
        ):
            issues.append(f"closed loop: resident checkpoint is not hash-pinned for {identifier}")
    elif availability == "missing":
        manifest = record.get("checkpoint_manifest")
        missing = record.get("missing_checkpoint_artifacts")
        if not _pinned_file(manifest):
            issues.append(f"closed loop: missing checkpoint lacks a pinned manifest for {identifier}")
        if not isinstance(missing, list) or not missing or not all(
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item["name"]
            and isinstance(item.get("sha256"), str)
            and len(item["sha256"]) == 64
            and isinstance(item.get("size_bytes"), int)
            and item["size_bytes"] > 0
            for item in (missing or [])
        ):
            issues.append(f"closed loop: missing checkpoint artifacts are not fully identified for {identifier}")
        for receipt_path in (
            full / "checkpoint_cache_receipt.json",
            full / "checkpoint_memfd_prefetch_receipt.json",
            full / "checkpoint_stream_receipt.json",
        ):
            if not receipt_path.is_file() or receipt_path.stat().st_mtime <= path.stat().st_mtime:
                continue
            try:
                receipt = json.loads(receipt_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if receipt.get("all_pinned_files_verified") is True:
                issues.append(
                    "closed loop: stale missing-checkpoint blocker is contradicted by "
                    f"newer verified receipt {receipt_path} for {identifier}"
                )
                break
    else:
        issues.append(f"closed loop: checkpoint availability is not explicit for {identifier}")

    return {"complete": not issues, "status": "blocked"}


def audit_campaign(results: Path) -> dict:
    records = []
    for identifier, name in CONFIGURATIONS:
        issues: list[str] = []
        attempt_path = results / identifier / "full" / "closed_loop" / "evaluator_attempt.json"
        try:
            status = json.loads(attempt_path.read_text()).get("status")
        except (OSError, json.JSONDecodeError):
            status = None
        if status == "blocked":
            component = evidence_backed_blocker_audit(attempt_path, identifier, issues)
        else:
            component = evaluator_audit(attempt_path, identifier, issues)
        records.append(
            {
                "id": identifier,
                "name": name,
                "status": component.get("status", "missing"),
                "complete": not issues,
                "evaluator_attempt_path": str(attempt_path),
                "evaluator_attempt_sha256": (
                    _sha256(attempt_path) if attempt_path.is_file() else None
                ),
                "issues": issues,
            }
        )

    status_counts = Counter(record["status"] for record in records)
    return {
        "schema_version": 2,
        "scope": "closed_loop_success_rate_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_count": len(records),
        "complete_count": sum(record["complete"] for record in records),
        "pending_count": sum(not record["complete"] for record in records),
        "status_counts": dict(sorted(status_counts.items())),
        "passed": all(record["complete"] for record in records),
        "configurations": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cross_model/closed_loop/success_rate_audit.json"),
    )
    args = parser.parse_args()

    report = audit_campaign(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "configurations"},
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

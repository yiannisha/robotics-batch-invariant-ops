"""Summarize independently accepted paired closed-loop success results.

This is a descriptive report, not a completion auditor.  It admits a model
only when the fail-closed success-rate audit marks the evaluator record both
``status=completed`` and ``complete=true`` with no issues, then verifies the
attempt-record and paired-summary hashes before deriving any statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


NATIVE_COMPARISON = "native_singleton|native_dynamic"
PATCHED_COMPARISON = "patched_singleton|patched_dynamic"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def comparison_row(summary: dict, name: str) -> dict:
    left, right = name.split("|")
    counts = summary.get("success_counts") or {}
    intervals = summary.get("paired_confidence_intervals") or {}
    disagreements = summary.get("disagreement_counts") or {}
    divergences = summary.get("action_divergence_counts") or {}
    samples = summary.get("paired_sample_counts") or {}
    default_samples = summary.get("paired_episodes", 0) // 4
    sample_count = samples.get(name, default_samples)
    interval = intervals.get(name)
    if (not isinstance(sample_count, int) or sample_count <= 0
            or not isinstance(counts.get(left), int)
            or not isinstance(counts.get(right), int)
            or not isinstance(disagreements.get(name), int)
            or not isinstance(divergences.get(name), int)
            or not isinstance(interval, dict)
            or not isinstance(interval.get("ci95"), list)
            or len(interval["ci95"]) != 2):
        raise ValueError(f"paired summary lacks complete comparison {name}")
    difference = (counts[right] - counts[left]) / sample_count
    if abs(difference - interval.get("difference", float("inf"))) > 1e-12:
        raise ValueError(f"reported difference disagrees with success counts for {name}")
    ci95 = interval["ci95"]
    if not all(isinstance(value, (int, float)) for value in ci95):
        raise ValueError(f"invalid confidence interval for {name}")
    discordant = disagreements[name]
    net_successes = counts[right] - counts[left]
    if ((discordant + net_successes) % 2
            or abs(net_successes) > discordant):
        raise ValueError(f"success counts and disagreements are inconsistent for {name}")
    right_only = (discordant + net_successes) // 2
    left_only = discordant - right_only
    if discordant:
        lower_tail = sum(
            math.comb(discordant, value)
            for value in range(min(left_only, right_only) + 1)
        ) / (2 ** discordant)
        mcnemar_p = min(1.0, 2 * lower_tail)
    else:
        mcnemar_p = 1.0
    return {
        "comparison": name,
        "sample_count": sample_count,
        "left_successes": counts[left],
        "right_successes": counts[right],
        "left_success_rate": counts[left] / sample_count,
        "right_success_rate": counts[right] / sample_count,
        "right_minus_left": difference,
        "ci95": ci95,
        "ci95_excludes_zero": ci95[0] > 0 or ci95[1] < 0,
        "outcome_disagreements": disagreements[name],
        "outcome_disagreement_rate": disagreements[name] / sample_count,
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "action_divergences": divergences[name],
        "action_divergence_rate": divergences[name] / sample_count,
    }


def build_report(results: Path) -> dict:
    audit_path = results / "closed_loop/success_rate_audit.json"
    audit = read_json(audit_path)
    if audit.get("schema_version") != 2:
        raise ValueError("success-rate audit does not pin evaluator-attempt hashes")
    audit_rows = audit.get("configurations", [])
    if not isinstance(audit_rows, list):
        raise ValueError("success-rate audit configurations are malformed")
    audited_attempts: dict[str, tuple[Path, dict | None]] = {}
    for audit_row in audit_rows:
        model_id = audit_row.get("id")
        attempt_path = Path(audit_row.get("evaluator_attempt_path", ""))
        if not attempt_path.is_absolute():
            attempt_path = Path.cwd() / attempt_path
        expected_hash = audit_row.get("evaluator_attempt_sha256")
        if attempt_path.is_file():
            if not isinstance(expected_hash, str) or sha256(attempt_path) != expected_hash:
                raise ValueError(
                    f"evaluator attempt changed after success-rate audit: {model_id}"
                )
            attempt = read_json(attempt_path)
        else:
            if expected_hash is not None:
                raise ValueError(
                    f"evaluator attempt disappeared after success-rate audit: {model_id}"
                )
            attempt = None
        audited_attempts[model_id] = (attempt_path, attempt)
    accepted = [
        row for row in audit_rows
        if row.get("status") == "completed"
        and row.get("complete") is True
        and row.get("issues") == []
    ]
    models = []
    for audit_row in accepted:
        model_id = audit_row.get("id")
        attempt_path, attempt = audited_attempts[model_id]
        if attempt is None:
            raise ValueError(f"completed evaluator attempt is missing: {model_id}")
        if attempt.get("model_id") != model_id or attempt.get("status") != "completed":
            raise ValueError(f"accepted audit row has invalid attempt record: {model_id}")
        summary_path = Path(attempt.get("paired_summary_path", ""))
        if (not summary_path.is_file()
                or sha256(summary_path) != attempt.get("paired_summary_sha256")):
            raise ValueError(f"paired summary is missing or hash-changed: {model_id}")
        summary = read_json(summary_path)
        if summary.get("model_id") != model_id or summary.get("evidence_tier") != "full":
            raise ValueError(f"paired summary is not full evidence: {model_id}")
        episodes_path = Path(summary.get("episodes_path", ""))
        if (not episodes_path.is_file()
                or sha256(episodes_path) != summary.get("episodes_sha256")):
            raise ValueError(f"paired episodes are missing or hash-changed: {model_id}")
        models.append({
            "model_id": model_id,
            "attempt_record_path": str(attempt_path.resolve()),
            "attempt_record_sha256": sha256(attempt_path),
            "paired_summary_path": str(summary_path.resolve()),
            "paired_summary_sha256": sha256(summary_path),
            "episodes_path": str(episodes_path.resolve()),
            "episodes_sha256": sha256(episodes_path),
            "native_batch_comparison": comparison_row(summary, NATIVE_COMPARISON),
            "patched_batch_comparison": comparison_row(summary, PATCHED_COMPARISON),
        })
    models.sort(key=lambda row: row["model_id"])

    native = [row["native_batch_comparison"] for row in models]
    patched = [row["patched_batch_comparison"] for row in models]
    native_n = sum(row["sample_count"] for row in native)
    patched_n = sum(row["sample_count"] for row in patched)
    aggregate = {
        "completed_model_count": len(models),
        "native_paired_initializations": native_n,
        "native_singleton_successes": sum(row["left_successes"] for row in native),
        "native_dynamic_successes": sum(row["right_successes"] for row in native),
        "native_dynamic_minus_singleton": (
            sum(row["right_successes"] - row["left_successes"] for row in native) / native_n
            if native_n else None
        ),
        "native_outcome_disagreements": sum(row["outcome_disagreements"] for row in native),
        "native_outcome_disagreement_rate": (
            sum(row["outcome_disagreements"] for row in native) / native_n
            if native_n else None
        ),
        "native_action_divergences": sum(row["action_divergences"] for row in native),
        "native_action_divergence_rate": (
            sum(row["action_divergences"] for row in native) / native_n
            if native_n else None
        ),
        "patched_paired_initializations": patched_n,
        "patched_outcome_disagreements": sum(row["outcome_disagreements"] for row in patched),
        "patched_outcome_disagreement_rate": (
            sum(row["outcome_disagreements"] for row in patched) / patched_n
            if patched_n else None
        ),
        "patched_action_divergences": sum(row["action_divergences"] for row in patched),
        "patched_action_divergence_rate": (
            sum(row["action_divergences"] for row in patched) / patched_n
            if patched_n else None
        ),
        "models_with_native_ci95_excluding_zero": [
            row["model_id"] for row in models
            if row["native_batch_comparison"]["ci95_excludes_zero"]
        ],
    }
    aggregate["closed_loop_behavior_change_observed"] = (
        aggregate["native_action_divergences"] > 0
        and aggregate["native_outcome_disagreements"] > 0
    )
    aggregate["systematic_accuracy_shift_supported_at_model_level_95pct"] = bool(
        aggregate["models_with_native_ci95_excluding_zero"]
    )
    return {
        "schema_version": 1,
        "scope": "independently_completed_paired_closed_loop_results",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_success_rate_audit_path": str(audit_path.resolve()),
        "source_success_rate_audit_sha256": sha256(audit_path),
        "simulation_only_not_physical_robot_evidence": True,
        "models": models,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "closed_loop/completed_success_results.json"
    report = build_report(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "model_id", "paired_initializations", "singleton_successes",
            "dynamic_successes", "dynamic_minus_singleton",
            "ci95_low", "ci95_high", "outcome_disagreements",
            "singleton_only_successes", "dynamic_only_successes",
            "mcnemar_exact_two_sided_p", "action_divergences", "summary_sha256",
        ])
        for model in report["models"]:
            row = model["native_batch_comparison"]
            writer.writerow([
                model["model_id"], row["sample_count"], row["left_successes"],
                row["right_successes"], row["right_minus_left"],
                row["ci95"][0], row["ci95"][1], row["outcome_disagreements"],
                row["left_only_successes"], row["right_only_successes"],
                row["mcnemar_exact_two_sided_p"], row["action_divergences"],
                model["paired_summary_sha256"],
            ])
    print(json.dumps({
        "completed_model_count": report["aggregate"]["completed_model_count"],
        "native_paired_initializations": report["aggregate"]["native_paired_initializations"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

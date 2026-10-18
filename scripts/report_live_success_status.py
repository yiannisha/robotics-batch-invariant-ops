"""Build a progress-only ledger from raw paired closed-loop episodes.

This report never promotes cardinality to an integrity claim. Dedicated model
auditors remain authoritative; this file exists to expose exact live counts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_cross_model_campaign import CONFIGURATIONS


CONDITIONS = (
    "native_singleton",
    "native_dynamic",
    "patched_singleton",
    "patched_dynamic",
)
PLANNED_PER_CONDITION = {
    "pi_fast": 500,
    "internvla_a15_standard": 500,
    "internvla_a15_optimized": 500,
    "molmoact2": 500,
    "groot_n17": 500,
    "xvla": 500,
    "uva": 500,
    "uva_joint": 500,
    "smolvla": 500,
    "openvla_oft": 500,
    "univla_baai": 500,
    "spatialvla": 496,
    "rdt_1b": 250,
    "cogact_small": 496,
    "wsa_base": 500,
    # The official CALVIN metric is one ordered five-subtask sequence.  Keep
    # those sequence records as the closed-loop denominator; do not inflate
    # them into five independent episodes.
    "openhelix": 1000,
}

# These evaluators execute dynamic requests in physical B=4 groups.  The final
# group for each 50-reset task therefore contains reset IDs 50 and 51 solely as
# companion slots.  They are useful batch-invariance evidence, but they are not
# part of the paired success-rate denominator.
LEROBOT_PHYSICAL_COMPANION_MODELS = {
    "pi_fast", "smolvla", "xvla", "univla_baai", "molmoact2",
}
VALIDATED_PREFIX_MODELS = {"openhelix", "univla_baai", "molmoact2"}


def lerobot_protocol_eligible(root: Path, condition: str) -> bool:
    manifest_path = root / "condition_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    expected_policy_batch = 1 if condition.endswith("_singleton") else 4
    expected_mode = "serial_singleton" if expected_policy_batch == 1 else "dynamic_batch"
    return (
        manifest.get("paired_protocol_version") == 2
        and manifest.get("condition") == condition
        and manifest.get("policy_batch_size") == expected_policy_batch
        and manifest.get("environment_batch_size") == 4
        and manifest.get("execution_mode") == expected_mode
        and manifest.get("smoke_test") is False
    )


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def certified_success_models(results: Path) -> set[str]:
    """Accept completed aggregate certifications only while their source is unchanged."""

    audit_path = results / "success_rate_audit.json"
    if not audit_path.is_file():
        return set()
    try:
        report = json.loads(audit_path.read_text())
    except Exception:
        return set()
    repository = results.parents[1]
    certified = set()
    for row in report.get("configurations", []):
        attempt = row.get("evaluator_attempt_path")
        attempt_path = Path(attempt) if isinstance(attempt, str) else None
        if attempt_path is not None and not attempt_path.is_absolute():
            attempt_path = repository / attempt_path
        if (
            row.get("complete") is True
            and row.get("status") == "completed"
            and row.get("issues") == []
            and attempt_path is not None
            and attempt_path.is_file()
            and valid_sha256(row.get("evaluator_attempt_sha256"))
            and file_sha256(attempt_path) == row["evaluator_attempt_sha256"]
        ):
            certified.add(row["id"])
    return certified


def condition_counts(root: Path, logical_initialization_limit: int | None = None) -> dict:
    seen: dict[tuple[object, object], bool] = {}
    duplicates = []
    conflicting_duplicates = []
    malformed_files = []
    physical_companion_slots = 0
    files = sorted((root / "rollouts").glob("*.json"))
    files += sorted((root / "rollouts").glob("*.json.gz"))
    for path in files:
        try:
            rows = read_json(path).get("episodes")
        except Exception as error:
            malformed_files.append({"path": str(path), "error": str(error)})
            continue
        if not isinstance(rows, list):
            malformed_files.append({"path": str(path), "error": "episodes is not a list"})
            continue
        for row in rows:
            if not isinstance(row, dict):
                malformed_files.append({"path": str(path), "error": "episode is not an object"})
                continue
            key = (row.get("task"), row.get("initialization_id"))
            if None in key:
                malformed_files.append({"path": str(path), "error": "episode key is incomplete"})
                continue
            if (logical_initialization_limit is not None
                    and isinstance(key[1], int)
                    and key[1] >= logical_initialization_limit):
                physical_companion_slots += 1
                continue
            success = row.get("success") is True
            if key in seen:
                duplicates.append({"task": key[0], "initialization_id": key[1]})
                if seen[key] != success:
                    conflicting_duplicates.append({
                        "task": key[0], "initialization_id": key[1],
                    })
                    # A progress ledger must not award a success when duplicate
                    # evidence disagrees; the dedicated auditor will reject the
                    # condition independently.
                    seen[key] = False
                continue
            seen[key] = success
    return {
        "episodes": len(seen),
        "successes": sum(seen.values()),
        "rollout_files": len(files),
        "physical_companion_slots": physical_companion_slots,
        "duplicate_episode_keys": duplicates,
        "conflicting_duplicate_episode_keys": conflicting_duplicates,
        "malformed_files": malformed_files,
        "official_summary_present": (root / "official/eval_info.json").is_file(),
    }


def openhelix_sequence_counts(root: Path) -> dict:
    """Count raw official CALVIN sequence records without weakening validation."""

    seen: dict[int, bool] = {}
    duplicates = []
    conflicting_duplicates = []
    malformed_files = []
    files = sorted((root / "rollouts").glob("sequence_*.json.gz"))
    for path in files:
        try:
            row = read_json(path)
        except Exception as error:
            malformed_files.append({"path": str(path), "error": str(error)})
            continue
        sequence_index = row.get("sequence_index") if isinstance(row, dict) else None
        if (
            not isinstance(sequence_index, int)
            or sequence_index < 0
            or row.get("condition") != root.name
            or not isinstance(row.get("success"), bool)
            or row.get("initialization_id") != sequence_index
            or row.get("explicit_init_state_id") != sequence_index
        ):
            malformed_files.append({
                "path": str(path), "error": "incomplete or inconsistent sequence identity",
            })
            continue
        success = row["success"]
        if sequence_index in seen:
            duplicates.append({"sequence_index": sequence_index})
            if seen[sequence_index] != success:
                conflicting_duplicates.append({"sequence_index": sequence_index})
                seen[sequence_index] = False
            continue
        seen[sequence_index] = success
    return {
        "episodes": len(seen),
        "successes": sum(seen.values()),
        "rollout_files": len(files),
        "physical_companion_slots": 0,
        "duplicate_episode_keys": duplicates,
        "conflicting_duplicate_episode_keys": conflicting_duplicates,
        "malformed_files": malformed_files,
        "official_summary_present": (root.parent.parent / "paired_summary.json").is_file(),
        "denominator_unit": "official_five_subtask_sequence",
    }


def validated_prefix_counts(root: Path, model_id: str, raw: dict) -> dict:
    """Expose only records accepted by a dedicated independent prefix audit."""

    empty = {
        "validated_episodes": 0,
        "validated_successes": 0,
        "validation_integrity_passed": False,
        "validation_provenance_passed": False,
        "validation_audit_path": None,
        "validation_source": None,
        "validation_error": None,
    }
    if model_id not in VALIDATED_PREFIX_MODELS:
        return empty
    path = root / "prefix_audit.json"
    if not path.is_file():
        return empty
    try:
        audit = json.loads(path.read_text())
    except Exception as error:
        return {**empty, "validation_audit_path": str(path), "validation_error": str(error)}
    integrity = (
        audit.get("model_id") == model_id
        and audit.get("condition") == root.name
        and audit.get("integrity_passed") is True
    )
    provenance = valid_sha256(audit.get("condition_manifest_sha256"))
    if model_id in {"univla_baai", "molmoact2"}:
        provenance = provenance and audit.get("provenance_integrity_passed") is True
        episodes = audit.get("planned_episodes_cached")
        successes = audit.get("planned_successes_cached")
    else:
        provenance = (
            provenance
            and valid_sha256(audit.get("adapter_sha256"))
        )
        episodes = audit.get("sequences_validated")
        successes = audit.get("full_sequence_successes")
    cardinality_valid = (
        isinstance(episodes, int)
        and isinstance(successes, int)
        and 0 <= successes <= episodes <= raw["episodes"]
        and successes <= raw["successes"]
    )
    accepted = integrity and provenance and cardinality_valid
    return {
        "validated_episodes": episodes if accepted else 0,
        "validated_successes": successes if accepted else 0,
        "validation_integrity_passed": integrity,
        "validation_provenance_passed": provenance,
        "validation_audit_path": str(path),
        "validation_source": "dedicated_independent_prefix_audit",
        "validation_error": None if accepted else "prefix audit/raw ledger mismatch",
    }


def history_snapshot(report: dict) -> dict:
    return {
        "schema_version": 2,
        "generated_at_utc": report["generated_at_utc"],
        "planned_paired_episodes": report["planned_paired_episodes"],
        "raw_paired_episodes": report["raw_paired_episodes"],
        "raw_paired_successes": report["raw_paired_successes"],
        "validated_paired_episodes": report["validated_paired_episodes"],
        "validated_paired_successes": report["validated_paired_successes"],
        "paths": [
            {
                "model_id": row["model_id"],
                "live_status": row["live_status"],
                "planned_paired_episodes": row["planned_paired_episodes"],
                "raw_paired_episodes": row["raw_paired_episodes"],
                "raw_paired_successes": row["raw_paired_successes"],
                "validated_paired_episodes": row["validated_paired_episodes"],
                "validated_paired_successes": row["validated_paired_successes"],
                "condition_episodes": {
                    condition: row["conditions"][condition]["episodes"]
                    for condition in CONDITIONS
                },
            }
            for row in report["paths"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    args = parser.parse_args()
    applicability_path = args.results / "closed_loop/applicability.json"
    applicability = json.loads(applicability_path.read_text())
    applicable = {row["id"]: row for row in applicability["configurations"]}
    certified_models = certified_success_models(args.results)

    rows = []
    for identifier, name in CONFIGURATIONS:
        classification = applicable[identifier]["status"]
        planned = PLANNED_PER_CONDITION.get(identifier)
        conditions = {}
        for condition in CONDITIONS:
            condition_root = args.results / identifier / "full/closed_loop/paired" / condition
            logical_initialization_limit = (
                50
                if identifier in LEROBOT_PHYSICAL_COMPANION_MODELS
                else None
            )
            counts = (
                openhelix_sequence_counts(condition_root)
                if identifier == "openhelix"
                else condition_counts(
                    condition_root,
                    logical_initialization_limit=logical_initialization_limit,
                )
            )
            protocol_eligible = (
                lerobot_protocol_eligible(condition_root, condition)
                if identifier in LEROBOT_PHYSICAL_COMPANION_MODELS else True
            )
            counts["protocol_eligible_for_progress"] = protocol_eligible
            counts["excluded_protocol_episodes"] = 0
            counts["excluded_protocol_successes"] = 0
            if not protocol_eligible and counts["episodes"]:
                counts["excluded_protocol_episodes"] = counts["episodes"]
                counts["excluded_protocol_successes"] = counts["successes"]
                counts["episodes"] = 0
                counts["successes"] = 0
            counts["planned_episodes"] = planned
            counts.update(validated_prefix_counts(condition_root, identifier, counts))
            counts["raw_cardinality_complete"] = (
                planned is not None
                and counts["episodes"] == planned
                and not counts["duplicate_episode_keys"]
                and not counts["malformed_files"]
                and counts["official_summary_present"]
            )
            conditions[condition] = counts
        observed = sum(item["episodes"] for item in conditions.values())
        successes = sum(item["successes"] for item in conditions.values())
        certified_complete = (
            identifier in certified_models
            and all(item["raw_cardinality_complete"] for item in conditions.values())
        )
        if certified_complete:
            for item in conditions.values():
                item.update({
                    "validated_episodes": item["episodes"],
                    "validated_successes": item["successes"],
                    "validation_integrity_passed": True,
                    "validation_provenance_passed": True,
                    "validation_audit_path": str(
                        args.results / "success_rate_audit.json"
                    ),
                    "validation_source": "completed_independent_success_rate_audit",
                    "validation_error": None,
                })
        validated = sum(item["validated_episodes"] for item in conditions.values())
        validated_successes = sum(item["validated_successes"] for item in conditions.values())
        if certified_complete:
            live_status = "completed_independently_audited"
        elif observed:
            # Raw completed rollouts supersede a stale preflight-only
            # applicability label, but remain only partial until audited.
            live_status = (
                "raw_cardinality_complete_requires_independent_audit"
                if all(item["raw_cardinality_complete"] for item in conditions.values())
                else "paired_campaign_partial"
            )
        elif classification.startswith("NOT APPLICABLE"):
            live_status = "not_applicable"
        elif classification == "BLOCKED BEFORE EPISODE 1":
            live_status = "blocked_before_episode_1"
        elif planned is None:
            live_status = "unclassified_runnable_path"
        else:
            live_status = "probe_only_paired_campaign_pending"
        rows.append({
            "model_id": identifier,
            "name": name,
            "applicability_status": classification,
            "live_status": live_status,
            "planned_episodes_per_condition": planned,
            "planned_paired_episodes": None if planned is None else planned * len(CONDITIONS),
            "raw_paired_episodes": observed,
            "raw_paired_successes": successes,
            "validated_paired_episodes": validated,
            "validated_paired_successes": validated_successes,
            "conditions": conditions,
        })

    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 2,
        "evidence_tier": "progress_ledger_with_raw_and_independently_validated_prefix_counts",
        "generated_at_utc": generated,
        "configuration_count": len(rows),
        "runnable_path_count": len(PLANNED_PER_CONDITION),
        "planned_paired_episodes": sum(
            value * len(CONDITIONS) for value in PLANNED_PER_CONDITION.values()
        ),
        "raw_paired_episodes": sum(row["raw_paired_episodes"] for row in rows),
        "raw_paired_successes": sum(row["raw_paired_successes"] for row in rows),
        "validated_paired_episodes": sum(row["validated_paired_episodes"] for row in rows),
        "validated_paired_successes": sum(row["validated_paired_successes"] for row in rows),
        "paths": rows,
    }
    output = args.results / "closed_loop/live_success_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    history_path = output.with_name("live_success_status_history.jsonl")
    with history_path.open("a") as stream:
        stream.write(json.dumps(history_snapshot(report), sort_keys=True) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "model_id", "live_status", "planned_paired_episodes",
            "raw_paired_episodes", "raw_paired_successes",
            "validated_paired_episodes", "validated_paired_successes",
            *[f"{condition}_episodes" for condition in CONDITIONS],
        ])
        for row in rows:
            writer.writerow([
                row["model_id"], row["live_status"], row["planned_paired_episodes"],
                row["raw_paired_episodes"], row["raw_paired_successes"],
                row["validated_paired_episodes"], row["validated_paired_successes"],
                *[row["conditions"][condition]["episodes"] for condition in CONDITIONS],
            ])
    print(json.dumps({
        "runnable_path_count": report["runnable_path_count"],
        "planned_paired_episodes": report["planned_paired_episodes"],
        "raw_paired_episodes": report["raw_paired_episodes"],
        "raw_paired_successes": report["raw_paired_successes"],
        "validated_paired_episodes": report["validated_paired_episodes"],
        "validated_paired_successes": report["validated_paired_successes"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

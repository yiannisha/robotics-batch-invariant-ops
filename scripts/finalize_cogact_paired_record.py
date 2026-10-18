"""Promote completed CogACT paired evidence after independent validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_full_cross_model_campaign import (
    COGACT_CHECKPOINT_BYTES,
    COGACT_CHECKPOINT_SHA256,
    COGACT_TASK_MANIFEST_SHA256,
    evaluator_audit,
    sha256,
)


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--attempt-record", type=Path, required=True)
    parser.add_argument("--checkpoint-weight", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--official-evaluator-source", type=Path, required=True)
    parser.add_argument("--evaluator-task-manifest", type=Path, required=True)
    args = parser.parse_args()

    paired_root = args.paired_root.resolve()
    summary_path = require_file(paired_root / "paired_summary.json", "paired summary")
    pairing_path = require_file(paired_root / "reset_pairing_audit.json", "pairing audit")
    checkpoint = require_file(args.checkpoint_weight, "CogACT checkpoint")
    source_evidence = require_file(args.source_evidence, "CogACT source evidence")
    evaluator_source = require_file(
        args.official_evaluator_source, "official CogACT evaluator source"
    )
    task_manifest = require_file(args.evaluator_task_manifest, "CogACT task manifest")
    summary = json.loads(summary_path.read_text())
    pairing = json.loads(pairing_path.read_text())
    episodes_path = require_file(Path(summary.get("episodes_path", "")), "paired episodes")
    if (
        summary.get("schema_version") != 1
        or summary.get("evidence_tier") != "full_paired_closed_loop"
        or summary.get("model_id") != "cogact_small"
        or summary.get("task_count") != 12
        or summary.get("initializations_per_condition") != 496
        or summary.get("paired_episode_rows") != 1984
        or summary.get("episodes_sha256") != sha256(episodes_path)
        or summary.get("task_manifest_sha256") != COGACT_TASK_MANIFEST_SHA256
        or sha256(task_manifest) != COGACT_TASK_MANIFEST_SHA256
    ):
        raise ValueError("CogACT summary lacks exact 12-task/496-reset/four-condition evidence")
    if pairing.get("fully_paired") is not True:
        raise ValueError("CogACT independent reset/noise pairing did not pass")
    if checkpoint.stat().st_size != COGACT_CHECKPOINT_BYTES:
        raise ValueError("CogACT checkpoint byte size differs")

    previous = (
        json.loads(args.attempt_record.read_text()) if args.attempt_record.is_file() else {}
    )
    record = dict(previous)
    record.update(
        {
            "schema_version": 1,
            "model_id": "cogact_small",
            "status": "completed",
            "completion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_weight_path": str(checkpoint),
            "checkpoint_weight_size_bytes": COGACT_CHECKPOINT_BYTES,
            "checkpoint_weight_sha256": COGACT_CHECKPOINT_SHA256,
            "exact_checkpoint_files": [
                {"path": str(checkpoint), "sha256": COGACT_CHECKPOINT_SHA256}
            ],
            "source_evidence_path": str(source_evidence),
            "source_evidence_sha256": sha256(source_evidence),
            "official_evaluator_source_path": str(evaluator_source),
            "official_evaluator_source_sha256": sha256(evaluator_source),
            "paired_start_seed": 20261005,
            "paired_episodes_completed": 1984,
            "episodes_path": str(episodes_path),
            "episodes_sha256": sha256(episodes_path),
            "paired_summary_path": str(summary_path),
            "paired_summary_sha256": sha256(summary_path),
            "reset_pairing_audit_path": str(pairing_path),
            "reset_pairing_audit_sha256": sha256(pairing_path),
            "condition_eval_info": summary["condition_eval_info"],
            "evaluator_task_manifest_path": str(task_manifest),
            "evaluator_task_manifest_sha256": COGACT_TASK_MANIFEST_SHA256,
        }
    )
    args.attempt_record.parent.mkdir(parents=True, exist_ok=True)
    staging = args.attempt_record.with_name(args.attempt_record.name + ".staging")
    staging.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    issues: list[str] = []
    result = evaluator_audit(staging, "cogact_small", issues)
    if not result.get("complete") or issues:
        staging.unlink(missing_ok=True)
        raise RuntimeError(
            "independent completed CogACT evaluator audit failed:\n- " + "\n- ".join(issues)
        )
    staging.replace(args.attempt_record)
    print(
        json.dumps(
            {
                "model_id": "cogact_small",
                "status": "completed",
                "paired_episodes": 1984,
                "independent_evaluator_audit": "passed",
                "attempt_record": str(args.attempt_record.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()

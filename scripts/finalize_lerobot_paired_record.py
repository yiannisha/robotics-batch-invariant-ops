"""Promote a completed paired LeRobot campaign only after independent validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_full_cross_model_campaign import (
    SPATIALVLA_ARTIFACT_HASHES, SPATIALVLA_CHECKPOINT, evaluator_audit,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        choices=(
            "smolvla", "xvla", "pi_fast",
            "uva", "uva_joint",
            "internvla_a15_standard", "internvla_a15_optimized",
            "openvla_oft", "wsa_base", "rdt_1b",
            "spatialvla", "groot_n17",
        ),
        required=True,
    )
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--attempt-record", type=Path, required=True)
    parser.add_argument("--checkpoint-weight", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--official-evaluator-source", type=Path, required=True)
    parser.add_argument("--evaluator-task-manifest", type=Path)
    args = parser.parse_args()

    paired_root = args.paired_root.resolve()
    summary_path = require_file(paired_root / "paired_summary.json", "paired summary")
    pairing_path = require_file(paired_root / "reset_pairing_audit.json", "reset-pairing audit")
    checkpoint = require_file(args.checkpoint_weight, "checkpoint weight")
    source_evidence = require_file(args.source_evidence, "checkpoint-matched evaluator evidence")
    evaluator_source = require_file(args.official_evaluator_source, "official evaluator source")
    summary = json.loads(summary_path.read_text())
    pairing = json.loads(pairing_path.read_text())
    episodes_path = require_file(Path(summary.get("episodes_path", "")), "paired episode stream")
    task_manifest = None
    task_record = None
    requires_task_manifest = (
        args.model_id.startswith("internvla_a15_")
        or args.model_id in (
            "uva", "uva_joint", "openvla_oft", "wsa_base", "rdt_1b",
            "spatialvla", "groot_n17",
        )
    )
    if requires_task_manifest:
        if args.evaluator_task_manifest is None:
            raise ValueError(f"{args.model_id} completion requires --evaluator-task-manifest")
        task_manifest = require_file(args.evaluator_task_manifest, "evaluator task manifest")
        task_record = json.loads(task_manifest.read_text())
    expected_tasks = (
        task_record.get("task_count", 0) if args.model_id == "spatialvla"
        else len(task_record.get("task_ids", [])) if task_record else 10
    )
    if expected_tasks < 1:
        raise ValueError("evaluator task manifest has no tasks")

    if args.model_id == "spatialvla":
        if (summary.get("evidence_tier") != "full_paired_closed_loop"
                or summary.get("model_id") != "spatialvla"
                or summary.get("paired_episode_rows") != 1_984
                or summary.get("task_count") != 12
                or summary.get("initializations_per_condition") != 496):
            raise ValueError("SpatialVLA paired summary lacks 12-task/496-reset/four-condition evidence")
    else:
        if summary.get("evidence_tier") != "full" or summary.get("model_id") != args.model_id:
            raise ValueError("paired summary is not full evidence for the requested model")
        if (summary.get("paired_episodes") != 4 * expected_tasks * 50
                or summary.get("tasks") != expected_tasks
                or summary.get("initializations_per_task") != 50):
            raise ValueError(
                f"paired summary does not contain {expected_tasks} tasks x 50 resets x 4 conditions"
            )
    if summary.get("episodes_sha256") != sha256(episodes_path):
        raise ValueError("paired episode stream hash differs from summary")
    if pairing.get("fully_paired") is not True:
        raise ValueError("independent reset/noise pairing audit did not pass")
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("resident checkpoint weight differs from pinned hash")

    previous = {}
    if args.attempt_record.is_file():
        previous = json.loads(args.attempt_record.read_text())
    record = dict(previous)
    record.update({
        "schema_version": 1,
        "model_id": args.model_id,
        "status": "completed",
        "completion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_weight_path": str(checkpoint),
        "checkpoint_weight_sha256": args.checkpoint_sha256,
        "source_evidence_path": str(source_evidence),
        "source_evidence_sha256": sha256(source_evidence),
        "official_evaluator_source_path": str(evaluator_source),
        "official_evaluator_source_sha256": sha256(evaluator_source),
        "paired_start_seed": (
            task_record["base_seed"] if args.model_id == "spatialvla"
            else summary["paired_start_seed"]
        ),
        "paired_episodes_completed": (
            summary["paired_episode_rows"] if args.model_id == "spatialvla"
            else summary["paired_episodes"]
        ),
        "episodes_path": str(episodes_path),
        "episodes_sha256": sha256(episodes_path),
        "paired_summary_path": str(summary_path),
        "paired_summary_sha256": sha256(summary_path),
        "reset_pairing_audit_path": str(pairing_path),
        "reset_pairing_audit_sha256": sha256(pairing_path),
        "condition_eval_info": summary["condition_eval_info"],
    })
    if requires_task_manifest:
        record.update({
            "evaluator_task_manifest_path": str(task_manifest),
            "evaluator_task_manifest_sha256": sha256(task_manifest),
        })
        if task_record.get("checkpoint_files"):
            record["exact_checkpoint_files"] = task_record["checkpoint_files"]
        elif args.model_id == "spatialvla":
            record["exact_checkpoint_files"] = [
                {"path": str((SPATIALVLA_CHECKPOINT / name).resolve()), "sha256": digest}
                for name, digest in SPATIALVLA_ARTIFACT_HASHES.items()
            ]

    args.attempt_record.parent.mkdir(parents=True, exist_ok=True)
    staging = args.attempt_record.with_name(args.attempt_record.name + ".staging")
    staging.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    issues: list[str] = []
    result = evaluator_audit(staging, args.model_id, issues)
    if not result.get("complete") or issues:
        staging.unlink(missing_ok=True)
        raise RuntimeError("independent completed-evaluator audit failed:\n- " + "\n- ".join(issues))
    staging.replace(args.attempt_record)
    print(json.dumps({
        "model_id": args.model_id,
        "status": "completed",
        "paired_episodes": record["paired_episodes_completed"],
        "independent_evaluator_audit": "passed",
        "attempt_record": str(args.attempt_record.resolve()),
    }))


if __name__ == "__main__":
    main()

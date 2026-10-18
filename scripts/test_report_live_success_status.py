import gzip
import json
from pathlib import Path

from scripts.report_live_success_status import (
    certified_success_models,
    condition_counts,
    file_sha256,
    history_snapshot,
    lerobot_protocol_eligible,
    openhelix_sequence_counts,
    validated_prefix_counts,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_condition_counts_deduplicates_json_and_gzip_and_reports_malformed(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts"
    write_json(rollouts / "one.json", {
        "episodes": [{"task": 0, "initialization_id": 0, "success": True}],
    })
    compressed = gzip.compress(json.dumps({
        "episodes": [{"task": 0, "initialization_id": 1, "success": False}],
    }).encode(), mtime=0)
    (rollouts / "two.json.gz").write_bytes(compressed)
    write_json(rollouts / "duplicate.json", {
        "episodes": [{"task": 0, "initialization_id": 0, "success": False}],
    })
    (rollouts / "malformed.json").write_text("not-json")

    result = condition_counts(tmp_path)

    assert result["episodes"] == 2
    assert result["successes"] == 0
    assert result["rollout_files"] == 4
    assert result["duplicate_episode_keys"] == [{"task": 0, "initialization_id": 0}]
    assert result["conflicting_duplicate_episode_keys"] == [
        {"task": 0, "initialization_id": 0},
    ]
    assert len(result["malformed_files"]) == 1
    assert result["official_summary_present"] is False


def test_condition_counts_rejects_incomplete_episode_identity(tmp_path: Path) -> None:
    write_json(tmp_path / "rollouts/incomplete.json", {
        "episodes": [{"task": 0, "success": True}],
    })

    result = condition_counts(tmp_path)

    assert result["episodes"] == 0
    assert result["successes"] == 0
    assert len(result["malformed_files"]) == 1


def test_condition_counts_excludes_physical_companion_slots(tmp_path: Path) -> None:
    write_json(tmp_path / "rollouts/final_batch.json", {
        "episodes": [
            {"task": 0, "initialization_id": 48, "success": True},
            {"task": 0, "initialization_id": 49, "success": False},
            {"task": 0, "initialization_id": 50, "success": True},
            {"task": 0, "initialization_id": 51, "success": True},
        ],
    })

    result = condition_counts(tmp_path, logical_initialization_limit=50)

    assert result["episodes"] == 2
    assert result["successes"] == 1
    assert result["physical_companion_slots"] == 2


def test_openhelix_sequence_counts_uses_sequence_denominator(tmp_path: Path) -> None:
    root = tmp_path / "native_singleton"
    for sequence_index, success in ((0, True), (1, False)):
        payload = {
            "condition": root.name,
            "sequence_index": sequence_index,
            "initialization_id": sequence_index,
            "explicit_init_state_id": sequence_index,
            "success": success,
        }
        path = root / "rollouts" / f"sequence_{sequence_index:04d}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))

    result = openhelix_sequence_counts(root)

    assert result["episodes"] == 2
    assert result["successes"] == 1
    assert result["denominator_unit"] == "official_five_subtask_sequence"
    assert result["malformed_files"] == []


def test_lerobot_progress_requires_fixed_environment_protocol(tmp_path: Path) -> None:
    manifest = {
        "paired_protocol_version": 2,
        "condition": "native_singleton",
        "policy_batch_size": 1,
        "environment_batch_size": 4,
        "execution_mode": "serial_singleton",
        "smoke_test": False,
    }
    write_json(tmp_path / "condition_manifest.json", manifest)
    assert lerobot_protocol_eligible(tmp_path, "native_singleton")
    manifest["environment_batch_size"] = 1
    write_json(tmp_path / "condition_manifest.json", manifest)
    assert not lerobot_protocol_eligible(tmp_path, "native_singleton")
    manifest["environment_batch_size"] = 4
    manifest["smoke_test"] = True
    write_json(tmp_path / "condition_manifest.json", manifest)
    assert not lerobot_protocol_eligible(tmp_path, "native_singleton")


def test_validated_prefix_counts_never_promotes_beyond_raw_openhelix(tmp_path: Path) -> None:
    root = tmp_path / "native_singleton"
    write_json(root / "prefix_audit.json", {
        "model_id": "openhelix", "condition": "native_singleton",
        "integrity_passed": True, "sequences_validated": 7,
        "full_sequence_successes": 3,
        "condition_manifest_sha256": "a" * 64, "adapter_sha256": "b" * 64,
    })
    accepted = validated_prefix_counts(
        root, "openhelix", {"episodes": 8, "successes": 4},
    )
    assert accepted["validated_episodes"] == 7
    assert accepted["validated_successes"] == 3
    rejected = validated_prefix_counts(
        root, "openhelix", {"episodes": 6, "successes": 4},
    )
    assert rejected["validated_episodes"] == 0
    assert rejected["validation_error"] == "prefix audit/raw ledger mismatch"


def test_validated_prefix_counts_requires_exact_model_provenance(tmp_path: Path) -> None:
    root = tmp_path / "native_singleton"
    audit = {
        "model_id": "univla_baai", "condition": "native_singleton",
        "integrity_passed": True, "provenance_integrity_passed": True,
        "planned_episodes_cached": 28, "planned_successes_cached": 5,
        "condition_manifest_sha256": "a" * 64,
    }
    write_json(root / "prefix_audit.json", audit)
    accepted = validated_prefix_counts(
        root, "univla_baai", {"episodes": 32, "successes": 6},
    )
    assert accepted["validated_episodes"] == 28
    audit["provenance_integrity_passed"] = False
    write_json(root / "prefix_audit.json", audit)
    rejected = validated_prefix_counts(
        root, "univla_baai", {"episodes": 32, "successes": 6},
    )
    assert rejected["validated_episodes"] == 0


def test_completed_certification_requires_unchanged_evaluator_attempt(tmp_path: Path) -> None:
    results = tmp_path / "results/cross_model"
    attempt = results / "model/full/closed_loop/evaluator_attempt.json"
    write_json(attempt, {"status": "completed"})
    audit = {
        "configurations": [{
            "id": "model", "status": "completed", "complete": True, "issues": [],
            "evaluator_attempt_path": str(attempt),
            "evaluator_attempt_sha256": file_sha256(attempt),
        }],
    }
    write_json(results / "success_rate_audit.json", audit)
    assert certified_success_models(results) == {"model"}
    write_json(attempt, {"status": "changed"})
    assert certified_success_models(results) == set()


def test_history_snapshot_retains_counts_without_heavy_integrity_details() -> None:
    conditions = {
        condition: {"episodes": index, "malformed_files": ["large detail"]}
        for index, condition in enumerate((
            "native_singleton", "native_dynamic",
            "patched_singleton", "patched_dynamic",
        ), start=1)
    }
    report = {
        "generated_at_utc": "2026-10-09T00:00:00+00:00",
        "planned_paired_episodes": 2000,
        "raw_paired_episodes": 10,
        "raw_paired_successes": 7,
        "validated_paired_episodes": 8,
        "validated_paired_successes": 6,
        "paths": [{
            "model_id": "model", "live_status": "paired_campaign_partial",
            "planned_paired_episodes": 2000, "raw_paired_episodes": 10,
            "raw_paired_successes": 7, "conditions": conditions,
            "validated_paired_episodes": 8, "validated_paired_successes": 6,
        }],
    }

    snapshot = history_snapshot(report)

    assert snapshot["raw_paired_episodes"] == 10
    assert snapshot["validated_paired_episodes"] == 8
    assert snapshot["paths"][0]["condition_episodes"] == {
        "native_singleton": 1, "native_dynamic": 2,
        "patched_singleton": 3, "patched_dynamic": 4,
    }
    assert "malformed_files" not in json.dumps(snapshot)

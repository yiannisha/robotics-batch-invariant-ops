import hashlib
import json
from pathlib import Path

import pytest

from scripts.report_completed_success_results import build_report
from scripts.plot_completed_success_results import render_svg


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_completed_result(root: Path) -> tuple[Path, Path]:
    model_root = root / "model/full/closed_loop"
    episodes = model_root / "paired/episodes.jsonl.gz"
    episodes.parent.mkdir(parents=True, exist_ok=True)
    episodes.write_bytes(b"pinned episodes")
    comparisons = {
        "native_singleton|native_dynamic": {
            "difference": 0.25, "ci95": [-0.25, 0.75],
            "method": "paired_percentile_bootstrap", "resamples": 10_000,
            "seed": 1,
        },
        "patched_singleton|patched_dynamic": {
            "difference": 0.0, "ci95": [0.0, 0.0],
            "method": "paired_percentile_bootstrap", "resamples": 10_000,
            "seed": 2,
        },
    }
    summary = model_root / "paired/paired_summary.json"
    write_json(summary, {
        "model_id": "model", "evidence_tier": "full", "paired_episodes": 16,
        "episodes_path": str(episodes), "episodes_sha256": digest(episodes),
        "success_counts": {
            "native_singleton": 2, "native_dynamic": 3,
            "patched_singleton": 3, "patched_dynamic": 3,
        },
        "paired_sample_counts": {
            "native_singleton|native_dynamic": 4,
            "patched_singleton|patched_dynamic": 4,
        },
        "disagreement_counts": {
            "native_singleton|native_dynamic": 1,
            "patched_singleton|patched_dynamic": 0,
        },
        "action_divergence_counts": {
            "native_singleton|native_dynamic": 4,
            "patched_singleton|patched_dynamic": 0,
        },
        "paired_confidence_intervals": comparisons,
    })
    attempt = model_root / "evaluator_attempt.json"
    write_json(attempt, {
        "model_id": "model", "status": "completed",
        "paired_summary_path": str(summary),
        "paired_summary_sha256": digest(summary),
    })
    write_json(root / "closed_loop/success_rate_audit.json", {
        "schema_version": 2,
        "configurations": [{
            "id": "model", "status": "completed", "complete": True,
            "issues": [], "evaluator_attempt_path": str(attempt),
            "evaluator_attempt_sha256": digest(attempt),
        }],
    })
    return summary, attempt


def test_report_uses_only_hash_verified_independently_completed_results(
    tmp_path: Path,
) -> None:
    make_completed_result(tmp_path)

    report = build_report(tmp_path)

    assert report["aggregate"]["completed_model_count"] == 1
    assert report["aggregate"]["native_paired_initializations"] == 4
    assert report["aggregate"]["native_outcome_disagreements"] == 1
    assert report["aggregate"]["native_action_divergences"] == 4
    assert report["aggregate"]["closed_loop_behavior_change_observed"] is True
    assert report["aggregate"]["systematic_accuracy_shift_supported_at_model_level_95pct"] is False
    comparison = report["models"][0]["native_batch_comparison"]
    assert comparison["left_only_successes"] == 0
    assert comparison["right_only_successes"] == 1
    assert comparison["mcnemar_exact_two_sided_p"] == 1.0


def test_report_rejects_summary_changed_after_completed_audit(tmp_path: Path) -> None:
    summary, _ = make_completed_result(tmp_path)
    summary.write_text(summary.read_text() + "\n")

    with pytest.raises(ValueError, match="hash-changed"):
        build_report(tmp_path)


def test_report_rejects_attempt_changed_after_completed_audit(tmp_path: Path) -> None:
    _, attempt = make_completed_result(tmp_path)
    attempt.write_text(attempt.read_text() + "\n")

    with pytest.raises(ValueError, match="evaluator attempt changed after"):
        build_report(tmp_path)


def test_report_rejects_pending_attempt_status_transition(tmp_path: Path) -> None:
    make_completed_result(tmp_path)
    pending = tmp_path / "pending/full/closed_loop/evaluator_attempt.json"
    write_json(pending, {"model_id": "pending", "status": "probe_succeeded"})
    audit_path = tmp_path / "closed_loop/success_rate_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["configurations"].append({
        "id": "pending", "status": "probe_succeeded_requires_paired_campaign",
        "complete": False, "issues": ["paired campaign pending"],
        "evaluator_attempt_path": str(pending),
        "evaluator_attempt_sha256": digest(pending),
    })
    write_json(audit_path, audit)
    write_json(pending, {"model_id": "pending", "status": "completed"})

    with pytest.raises(ValueError, match="pending"):
        build_report(tmp_path)


def test_svg_pins_audit_and_renders_completed_model(tmp_path: Path) -> None:
    make_completed_result(tmp_path)
    report = build_report(tmp_path)

    svg = render_svg(report)

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "source_success_rate_audit_sha256=" in svg
    assert ">model</text>" in svg
    assert "native 25.0%" in svg
    assert "repaired 0.0%" in svg

"""Completion-contract checks for the independent GR00T pairing audit."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from scripts.audit_groot_reset_pairing import CONDITIONS, audit
from scripts.audit_groot_paired_prefix import audit as audit_prefix


PROBE = Path(
    "/workspace/robotics-batch-invariant-ops/results/cross_model/groot_n17/"
    "full/closed_loop/paired_probe/native_singleton"
)


def write_condition(root, condition: str, task_count: int, resets: int) -> None:
    rows = []
    for task in range(task_count):
        for initialization in range(resets):
            rows.append({
                "task": task,
                "initialization_id": initialization,
                "reset_seed": 20261005 + initialization,
                "initial_observation_sha256": f"{task:02x}{initialization:04x}".ljust(64, "0"),
                "first_noise_sha256": f"{task:02x}{initialization:04x}".ljust(64, "1"),
                "success": False,
                "action_sequence": [[0.0] * 7],
            })
    directory = root / condition / "rollouts"
    directory.mkdir(parents=True)
    (directory / "all.json").write_text(json.dumps({"episodes": rows}))


def test_partial_pairing_never_claims_completion(tmp_path) -> None:
    for condition in CONDITIONS:
        write_condition(tmp_path, condition, task_count=1, resets=1)
    report = audit(tmp_path, require_complete=False)
    assert report["pairing_integrity_passed"] is True
    assert report["fully_paired"] is False
    with pytest.raises(ValueError, match="exact 500-key schedule"):
        audit(tmp_path, require_complete=True)


def test_exact_full_schedule_sets_shared_completion_contract(tmp_path) -> None:
    for condition in CONDITIONS:
        write_condition(tmp_path, condition, task_count=10, resets=50)
    report = audit(tmp_path, require_complete=True)
    assert report["fully_paired"] is True
    assert report["condition_episode_counts"] == {
        condition: 500 for condition in CONDITIONS
    }
    assert report["pairing_checks"] == 1500


def test_losslessly_compressed_prefix_remains_auditable(tmp_path) -> None:
    root = tmp_path / "probe"
    shutil.copytree(PROBE, root)
    source = next((root / "rollouts").glob("*.json"))
    compressed = source.with_suffix(".json.gz")
    compressed.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    source.unlink()
    report = audit_prefix(root)
    assert report["integrity_passed"] is True
    assert report["episodes_cached"] == 1

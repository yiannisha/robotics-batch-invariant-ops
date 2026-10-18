"""Tests for the queued exact-checkpoint LIBERO prefix monitor."""

from __future__ import annotations

import json
from pathlib import Path

from monitor_exact_lerobot_paired_prefix import validate_due_prefixes


def test_monitor_waits_for_step_then_persists_independent_audit(tmp_path: Path) -> None:
    root = tmp_path / "native_singleton"
    rollouts = root / "rollouts"
    rollouts.mkdir(parents=True)
    (root / "condition_manifest.json").write_text("{}")
    calls = []

    def fake_audit(condition_root: Path) -> dict:
        calls.append(condition_root)
        count = len(list((condition_root / "rollouts").glob("*.json")))
        return {
            "rollout_files": count,
            "planned_episodes_cached": count * 4,
            "integrity_passed": True,
            "provenance_integrity_passed": True,
        }

    for index in range(6):
        (rollouts / f"rollout_{index}.json").write_text(json.dumps({}))
    state = validate_due_prefixes(tmp_path, 7, fake_audit)
    assert calls == []
    assert state["conditions"]["native_singleton"]["validated_rollout_files"] == 0

    (rollouts / "rollout_6.json").write_text(json.dumps({}))
    state = validate_due_prefixes(tmp_path, 7, fake_audit)
    assert calls == [root]
    assert state["conditions"]["native_singleton"]["validated_rollout_files"] == 7
    persisted = json.loads((root / "prefix_audit.json").read_text())
    assert persisted["provenance_integrity_passed"] is True
    assert "validated_at_utc" in persisted


def test_monitor_treats_only_500_validated_planned_episodes_as_complete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "native_singleton"
    rollouts = root / "rollouts"
    rollouts.mkdir(parents=True)
    (root / "condition_manifest.json").write_text("{}")
    for index in range(130):
        (rollouts / f"rollout_{index}.json").write_text("{}")

    def fake_audit(_condition_root: Path) -> dict:
        return {
            "rollout_files": 130,
            "planned_episodes_cached": 500,
            "integrity_passed": True,
            "provenance_integrity_passed": True,
        }

    state = validate_due_prefixes(tmp_path, 7, fake_audit)
    assert state["conditions"]["native_singleton"]["complete"] is True
    assert state["all_conditions_complete"] is False

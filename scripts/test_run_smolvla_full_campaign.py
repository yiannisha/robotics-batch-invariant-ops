import json
from pathlib import Path

import pytest

import run_smolvla_full_campaign as campaign


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    full = root / "results/cross_model/smolvla/full"
    pilot = root / "results/cross_model/smolvla/pilots/non_success_pre_full_coverage"
    adapter = root / "scripts/model_invariance/adapters/lerobot_smolvla.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("adapter-v1\n")
    full.mkdir(parents=True)
    (full / "replay_manifest.json").write_text('{"observations": []}\n')
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "FULL", full)
    monkeypatch.setattr(campaign, "MANIFEST", full / "replay_manifest.json")
    monkeypatch.setattr(campaign, "PILOT", pilot)
    monkeypatch.setattr(campaign, "START_MARKER", full / "non_success_full_campaign_started.json")
    return full, pilot


def test_archive_preserves_pilots_and_closed_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full, pilot = configure(tmp_path, monkeypatch)
    (full / "diagnostic").mkdir()
    (full / "diagnostic/invariance.jsonl.gz").write_bytes(b"pilot")
    (full / "performance").mkdir()
    (full / "performance/policy_timings.jsonl").write_text("pilot\n")
    (full / "closed_loop").mkdir()
    (full / "closed_loop/evaluator_attempt.json").write_text("{}\n")
    (full / "numerical_freeze.json").write_text("{}\n")

    campaign.archive_pre_full_coverage_pilots()

    assert (pilot / "diagnostic/invariance.jsonl.gz").read_bytes() == b"pilot"
    assert (pilot / "performance/policy_timings.jsonl").read_text() == "pilot\n"
    assert (pilot / "numerical_freeze.json").is_file()
    assert (full / "closed_loop/evaluator_attempt.json").is_file()
    assert (full / "replay_manifest.json").is_file()
    marker = json.loads(campaign.START_MARKER.read_text())
    assert marker["archived"] == ["diagnostic", "performance", "numerical_freeze.json"]
    campaign.archive_pre_full_coverage_pilots()


def test_archive_rejects_adapter_change_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(tmp_path, monkeypatch)
    campaign.archive_pre_full_coverage_pilots()
    adapter = campaign.ROOT / "scripts/model_invariance/adapters/lerobot_smolvla.py"
    adapter.write_text("adapter-v2\n")
    with pytest.raises(RuntimeError, match="adapter changed"):
        campaign.archive_pre_full_coverage_pilots()

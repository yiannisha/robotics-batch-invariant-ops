import json
from pathlib import Path

import pytest

import run_uva_action_full_campaign as campaign


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    model_root = root / "results/cross_model/uva"
    full = model_root / "full"
    pilot = model_root / "pilots/non_success_pre_full_coverage"
    adapters = root / "scripts/model_invariance/adapters"
    adapters.mkdir(parents=True)
    (adapters / "uva_libero10.py").write_text("base-v1\n")
    (adapters / "uva_libero10_full.py").write_text("full-v1\n")
    full.mkdir(parents=True)
    (full / "replay_manifest.json").write_text('{"observations": []}\n')
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "MODEL_ROOT", model_root)
    monkeypatch.setattr(campaign, "FULL", full)
    monkeypatch.setattr(campaign, "MANIFEST", full / "replay_manifest.json")
    monkeypatch.setattr(campaign, "PILOT", pilot)
    monkeypatch.setattr(campaign, "START_MARKER", full / "non_success_full_campaign_started.json")
    monkeypatch.setattr(campaign, "ADAPTER_FILES", {
        "base_adapter": adapters / "uva_libero10.py",
        "full_adapter": adapters / "uva_libero10_full.py",
    })
    return full


def test_pilot_archive_preserves_closed_loop_and_labels_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = configure(tmp_path, monkeypatch)
    (full / "diagnostic").mkdir()
    (full / "diagnostic/invariance.jsonl.gz").write_bytes(b"pilot")
    (full / "closed_loop").mkdir()
    (full / "closed_loop/evaluator_attempt.json").write_text("{}\n")
    (campaign.MODEL_ROOT / "baseline.json").write_text("{}\n")
    campaign.archive_pilots()
    assert (campaign.PILOT / "diagnostic/invariance.jsonl.gz").read_bytes() == b"pilot"
    assert (full / "closed_loop/evaluator_attempt.json").is_file()
    marker = json.loads(campaign.START_MARKER.read_text())
    assert marker["archived"] == ["diagnostic"]
    assert marker["legacy_pilot_paths"] == ["results/cross_model/uva/baseline.json"]
    campaign.archive_pilots()


def test_pilot_archive_rejects_full_adapter_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(tmp_path, monkeypatch)
    campaign.archive_pilots()
    (campaign.ROOT / "scripts/model_invariance/adapters/uva_libero10_full.py").write_text(
        "full-v2\n"
    )
    with pytest.raises(RuntimeError, match="adapter changed"):
        campaign.archive_pilots()


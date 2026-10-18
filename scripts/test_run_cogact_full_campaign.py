import json
from pathlib import Path

import pytest

import run_cogact_full_campaign as campaign


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    model_root = root / "results/cross_model/cogact_small"
    full = model_root / "full"
    pilot = model_root / "pilots/non_success_pre_full_coverage"
    adapter = root / "scripts/model_invariance/adapters/cogact_small.py"
    invariant = root / "batch_invariant_ops/batch_invariant_ops.py"
    adapter.parent.mkdir(parents=True)
    invariant.parent.mkdir(parents=True)
    adapter.write_text("adapter-v1\n")
    invariant.write_text("ops-v1\n")
    full.mkdir(parents=True)
    (full / "replay_manifest.json").write_text('{"observations": []}\n')
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "MODEL_ROOT", model_root)
    monkeypatch.setattr(campaign, "FULL", full)
    monkeypatch.setattr(campaign, "MANIFEST", full / "replay_manifest.json")
    monkeypatch.setattr(campaign, "PILOT", pilot)
    monkeypatch.setattr(campaign, "START_MARKER", full / "non_success_full_campaign_started.json")
    return full


def test_archive_preserves_replay_and_success_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = configure(tmp_path, monkeypatch)
    (full / "diagnostic").mkdir()
    (full / "diagnostic/invariance.jsonl.gz").write_bytes(b"pilot")
    (full / "closed_loop").mkdir()
    (full / "closed_loop/paired_summary.json").write_text("{}\n")
    (full / "episodes").mkdir()
    (full / "episodes/task_00.json").write_text("{}\n")
    (full / "frames").mkdir()
    (full / "frames/frame.png").write_bytes(b"png")
    campaign.archive_pilots()
    assert (campaign.PILOT / "full_diagnostic/invariance.jsonl.gz").read_bytes() == b"pilot"
    assert (full / "closed_loop/paired_summary.json").is_file()
    assert (full / "episodes/task_00.json").is_file()
    assert (full / "frames/frame.png").is_file()
    assert json.loads(campaign.START_MARKER.read_text())["archived"] == ["full/diagnostic"]
    campaign.archive_pilots()


def test_archive_rejects_adapter_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(tmp_path, monkeypatch)
    campaign.archive_pilots()
    (campaign.ROOT / "scripts/model_invariance/adapters/cogact_small.py").write_text(
        "adapter-v2\n"
    )
    with pytest.raises(RuntimeError, match="sources changed"):
        campaign.archive_pilots()

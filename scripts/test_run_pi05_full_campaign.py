import hashlib
import json
from pathlib import Path

import pytest

import run_pi05_full_campaign as campaign


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    model_root = root / "results/cross_model/pi05"
    full = model_root / "full"
    full.mkdir(parents=True)
    manifest = full / "replay_manifest.json"
    manifest.write_text('{"observations": []}\n')
    adapter = root / "scripts/model_invariance/adapters/lerobot_pi05.py"
    invariant = root / "batch_invariant_ops/batch_invariant_ops.py"
    adapter.parent.mkdir(parents=True)
    invariant.parent.mkdir(parents=True)
    adapter.write_text("adapter-v1\n")
    invariant.write_text("ops-v1\n")
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "MODEL_ROOT", model_root)
    monkeypatch.setattr(campaign, "FULL", full)
    monkeypatch.setattr(campaign, "MANIFEST", manifest)
    monkeypatch.setattr(campaign, "PILOT", model_root / "pilots/non_success_pre_full_coverage")
    monkeypatch.setattr(campaign, "START_MARKER", full / "non_success_full_campaign_started.json")
    return full


def test_pi05_archive_preserves_replay_and_closed_loop(tmp_path: Path, monkeypatch) -> None:
    full = configure(tmp_path, monkeypatch)
    (full / "diagnostic").mkdir()
    (full / "diagnostic/pilot.json").write_text("{}\n")
    (full / "closed_loop").mkdir()
    (full / "closed_loop/evaluator_attempt.json").write_text("{}\n")
    (campaign.MODEL_ROOT / "baseline.json").write_text("{}\n")
    campaign.archive_pilots()
    assert (campaign.PILOT / "full_diagnostic/pilot.json").is_file()
    assert (campaign.PILOT / "baseline.json").is_file()
    assert (full / "closed_loop/evaluator_attempt.json").is_file()
    assert (full / "replay_manifest.json").is_file()
    marker = json.loads(campaign.START_MARKER.read_text())
    assert marker["checkpoint_sha256"] == campaign.CHECKPOINT_SHA256
    campaign.archive_pilots()


def test_pi05_checkpoint_preflight_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"exact-weight")
    monkeypatch.setattr(campaign, "CHECKPOINT_WEIGHT", weight)
    monkeypatch.setattr(campaign, "CHECKPOINT_BYTES", len(b"exact-weight"))
    monkeypatch.setattr(
        campaign, "CHECKPOINT_SHA256", hashlib.sha256(b"exact-weight").hexdigest()
    )
    campaign.verify_checkpoint()
    weight.write_bytes(b"wrong-weight")
    with pytest.raises(ValueError, match="differs"):
        campaign.verify_checkpoint()

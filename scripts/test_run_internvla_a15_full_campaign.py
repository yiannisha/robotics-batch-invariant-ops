import json
from pathlib import Path

import pytest

import run_internvla_a15_full_campaign as campaign


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str) -> Path:
    root = tmp_path / "repo"
    results = root / "results/cross_model"
    directory = results / model_id / "full"
    adapter = root / "scripts/model_invariance/adapters/internvla_a15.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("adapter-v1\n")
    directory.mkdir(parents=True)
    (directory / "replay_manifest.json").write_text('{"observations": []}\n')
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "RESULTS", results)
    return directory


@pytest.mark.parametrize("model_id,backend", campaign.PATHS)
def test_backend_pilot_archive_is_separate_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, backend: str,
) -> None:
    directory = configure(tmp_path, monkeypatch, model_id)
    (directory / "diagnostic").mkdir()
    (directory / "diagnostic/invariance.jsonl.gz").write_bytes(b"pilot")
    (directory / "closed_loop").mkdir()
    (directory / "closed_loop/evaluator_attempt.json").write_text("{}\n")

    campaign.archive_pilots(model_id, backend)

    pilot = campaign.RESULTS / model_id / "pilots/non_success_pre_full_coverage"
    assert (pilot / "diagnostic/invariance.jsonl.gz").read_bytes() == b"pilot"
    assert (directory / "closed_loop/evaluator_attempt.json").is_file()
    marker = json.loads((directory / "non_success_full_campaign_started.json").read_text())
    assert marker["backend"] == backend
    assert marker["adapter_execution_configuration"] == campaign.execution_configuration(backend)
    campaign.archive_pilots(model_id, backend)


def test_backend_archive_rejects_configuration_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "internvla_a15_optimized"
    configure(tmp_path, monkeypatch, model_id)
    campaign.archive_pilots(model_id, "optimized")
    with pytest.raises(RuntimeError, match="changed after campaign start"):
        campaign.archive_pilots(model_id, "standard")

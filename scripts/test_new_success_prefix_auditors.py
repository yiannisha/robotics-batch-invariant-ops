"""Mutation checks for the WSA, RDT, and SpatialVLA prefix auditors."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from scripts.audit_rdt_paired_prefix import audit as audit_rdt
from scripts.audit_spatialvla_paired_prefix import (
    audit as audit_spatialvla,
    valid_state as valid_spatialvla_state,
)
from scripts.audit_wsa_paired_prefix import audit as audit_wsa


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "results/cross_model"


def copied_probe(tmp_path: Path, model: str) -> Path:
    source = PROBES / model / "full/closed_loop/probe/native_singleton"
    assert source.is_dir(), f"missing checked-in probe evidence: {source}"
    destination = tmp_path / model
    shutil.copytree(source, destination)
    if model == "spatialvla":
        manifest_path = destination / "condition_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["server_metadata_path"] = str((destination / "server_metadata.json").resolve())
        manifest_path.write_text(json.dumps(manifest))
    return destination


@pytest.mark.parametrize(
    ("model", "auditor"),
    (
        ("wsa_base", audit_wsa),
        ("rdt_1b", audit_rdt),
        ("spatialvla", audit_spatialvla),
    ),
)
def test_real_probe_evidence_passes(tmp_path: Path, model: str, auditor) -> None:
    result = auditor(copied_probe(tmp_path, model))
    assert result["integrity_passed"] is True
    assert result["complete"] is True
    assert result["episodes_cached"] == 1


@pytest.mark.parametrize(
    ("model", "auditor", "rollout_name"),
    (
        (
            "wsa_base",
            audit_wsa,
            "libero_10_0_initializations_0_0.json",
        ),
        (
            "rdt_1b",
            audit_rdt,
            "maniskill_0_initializations_0_0.json",
        ),
    ),
)
def test_stochastic_auditors_reject_wrong_request_noise(
    tmp_path: Path, model: str, auditor, rollout_name: str,
) -> None:
    root = copied_probe(tmp_path, model)
    path = root / "rollouts" / rollout_name
    payload = json.loads(path.read_text())
    payload["episodes"][0]["first_noise_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        auditor(root)


def test_spatialvla_rejects_malformed_raw_action_chunk(tmp_path: Path) -> None:
    root = copied_probe(tmp_path, "spatialvla")
    path = root / "rollouts/task_0_initializations_0_0.json"
    payload = json.loads(path.read_text())
    payload["episodes"][0]["raw_action_chunks"][0][0].append(0.0)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        audit_spatialvla(root)


def test_spatialvla_state_schema_covers_both_official_robot_families() -> None:
    joint_state = {"qpos": [0.0, 1.0], "qvel": [0.0, 0.0]}
    assert valid_spatialvla_state({**joint_state, "tcp_pose": [0.0] * 7}, "tcp_pose")
    assert valid_spatialvla_state({**joint_state, "base_pose": [0.0] * 7}, "base_pose")
    assert not valid_spatialvla_state({**joint_state, "tcp_pose": [0.0] * 7}, "base_pose")
    assert not valid_spatialvla_state({**joint_state, "base_pose": [0.0] * 7}, "tcp_pose")


def test_wsa_rejects_duplicate_physical_rollout(tmp_path: Path) -> None:
    root = copied_probe(tmp_path, "wsa_base")
    source = root / "rollouts/libero_10_0_initializations_0_0.json"
    shutil.copy2(source, root / "rollouts/duplicate.json")
    with pytest.raises(ValueError, match="duplicate WSA task/reset"):
        audit_wsa(root)


def test_wsa_accepts_losslessly_compressed_rollout(tmp_path: Path) -> None:
    root = copied_probe(tmp_path, "wsa_base")
    source = root / "rollouts/libero_10_0_initializations_0_0.json"
    compressed = source.with_suffix(source.suffix + ".gz")
    with source.open("rb") as input_stream, gzip.open(compressed, "wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    source.unlink()
    result = audit_wsa(root)
    assert result["episodes_cached"] == 1
    assert result["integrity_passed"] is True


def test_wsa_rejects_duplicate_compressed_representation(tmp_path: Path) -> None:
    root = copied_probe(tmp_path, "wsa_base")
    source = root / "rollouts/libero_10_0_initializations_0_0.json"
    compressed = source.with_suffix(source.suffix + ".gz")
    with source.open("rb") as input_stream, gzip.open(compressed, "wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    with pytest.raises(ValueError, match="duplicate compressed and uncompressed"):
        audit_wsa(root)

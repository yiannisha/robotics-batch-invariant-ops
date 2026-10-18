from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_openhelix_paired_prefix import (
    array_bundle_sha256,
    audit,
    file_sha256,
)
from scripts.monitor_openhelix_paired_prefix import validate_due_prefixes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cross_model/openhelix/full/closed_loop/paired/native_singleton"


def copy_one_sequence(tmp_path: Path) -> Path:
    target = tmp_path / "native_singleton"
    (target / "rollouts").mkdir(parents=True)
    shutil.copy2(SOURCE / "condition_manifest.json", target / "condition_manifest.json")
    for suffix in ("json.gz", "npz"):
        shutil.copy2(
            SOURCE / "rollouts" / f"sequence_0000.{suffix}",
            target / "rollouts" / f"sequence_0000.{suffix}",
        )
    metadata = target / "rollouts/sequence_0000.json.gz"
    with gzip.open(metadata, "rt") as stream:
        row = json.load(stream)
    row["trajectory_npz_path"] = str(
        (target / "rollouts/sequence_0000.npz").resolve()
    )
    with gzip.open(metadata, "wt") as stream:
        json.dump(row, stream)
    return target


def test_real_openhelix_sequence_passes(tmp_path: Path) -> None:
    result = audit(copy_one_sequence(tmp_path))
    assert result["integrity_passed"] is True
    assert result["complete"] is False
    assert result["sequences_validated"] == 1


def test_openhelix_rejects_reset_identity_mutation(tmp_path: Path) -> None:
    root = copy_one_sequence(tmp_path)
    path = root / "rollouts/sequence_0000.json.gz"
    with gzip.open(path, "rt") as stream:
        row = json.load(stream)
    row["explicit_init_state_id"] = 7
    with gzip.open(path, "wt") as stream:
        json.dump(row, stream)
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        audit(root)


def test_openhelix_rejects_missing_trajectory_pair(tmp_path: Path) -> None:
    root = copy_one_sequence(tmp_path)
    (root / "rollouts/sequence_0000.npz").unlink()
    with pytest.raises(ValueError, match="prefix differs"):
        audit(root)


def test_monitor_validates_due_prefix_and_records_other_conditions(tmp_path: Path) -> None:
    copy_one_sequence(tmp_path)

    state = validate_due_prefixes(tmp_path, step=1)

    assert state["all_conditions_complete"] is False
    assert state["conditions"]["native_singleton"] == {
        "raw_sequences": 1,
        "validated_sequences": 1,
        "complete": False,
    }
    assert state["conditions"]["native_dynamic"]["raw_sequences"] == 0
    result = json.loads(
        (tmp_path / "native_singleton/prefix_audit.json").read_text()
    )
    assert result["integrity_passed"] is True
    assert result["sequences_validated"] == 1
    assert "validated_at_utc" in result


def test_openhelix_rejects_self_consistent_trajectory_request_mutation(
    tmp_path: Path,
) -> None:
    root = copy_one_sequence(tmp_path)
    arrays_path = root / "rollouts/sequence_0000.npz"
    with np.load(arrays_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["batch_sizes"][0] = 2
    with arrays_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata_path = root / "rollouts/sequence_0000.json.gz"
    with gzip.open(metadata_path, "rt") as stream:
        row = json.load(stream)
    row["trajectory_npz_sha256"] = file_sha256(arrays_path)
    row["trajectory_arrays_sha256"] = array_bundle_sha256(arrays)
    with gzip.open(metadata_path, "wt") as stream:
        json.dump(row, stream)

    with pytest.raises(ValueError, match="trajectory/request mapping"):
        audit(root)

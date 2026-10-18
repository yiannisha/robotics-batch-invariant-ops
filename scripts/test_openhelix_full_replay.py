from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_full_cross_model_campaign import openhelix_source_audit


ROOT = Path("/workspace/robotics-batch-invariant-ops")
MANIFEST_PATH = ROOT / "results/cross_model/openhelix/full/replay_manifest.json"


def test_openhelix_manifest_has_exact_full_cardinality_and_noise() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = manifest["observations"]
    assert len(rows) == 1000
    assert len({row["id"] for row in rows}) == 1000
    assert len({row["physical_observation_sha256"] for row in rows}) == 1000
    assert sum(row["split"] == "diagnostic" for row in rows) == 200
    assert sum(row["split"] == "heldout" for row in rows) == 800
    noise = np.load(manifest["noise_file"], mmap_mode="r", allow_pickle=False)
    assert noise.shape == (1000, 3, 4446)
    assert noise.dtype == np.float32
    for index in (0, 199, 200, 999):
        for noise_id in range(3):
            digest = hashlib.sha256(np.asarray(noise[index, noise_id]).tobytes()).hexdigest()
            assert rows[index]["noise"][noise_id]["sha256"] == digest


def test_openhelix_independent_source_audit_passes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    issues: list[str] = []
    openhelix_source_audit(manifest, manifest["observations"], issues)
    assert issues == []


def test_openhelix_adapter_reconstructs_frozen_request(monkeypatch) -> None:
    monkeypatch.setenv("OPENHELIX_SOURCE", "/workspace/models/openhelix")
    monkeypatch.setenv(
        "OPENHELIX_CHECKPOINT",
        "/workspace/assets/openhelix/checkpoint/prompt_tuning_aux",
    )
    monkeypatch.setenv("OPENHELIX_BASE", "/workspace/assets/openhelix/llava_metadata")
    monkeypatch.setenv("OPENHELIX_VISION", "/workspace/assets/openhelix/clip_metadata")
    monkeypatch.setenv("OPENHELIX_REPLAY_MANIFEST", str(MANIFEST_PATH))
    from scripts.model_invariance.adapters.openhelix_calvin import OpenHelixCalvinAdapter

    adapter = OpenHelixCalvinAdapter()
    adapter._add_source_path()
    request = adapter.load_replay_input(999, 2)
    assert request["planner_image"].shape == (200, 200, 3)
    assert request["rgb"].shape == (2, 3, 200, 200)
    assert request["pcd"].shape == (2, 3, 200, 200)
    assert request["proprio"].shape == (8,)
    assert request["noise"].shape == (19, 9)
    assert request["position_step_noise"].shape == (25, 19, 3)
    assert request["rotation_step_noise"].shape == (25, 19, 6)


def test_openhelix_remote_shard_uses_sealed_memfd(tmp_path, monkeypatch) -> None:
    from scripts.model_invariance.adapters import openhelix_calvin as module

    payload = b"exact-test-shard" * 257
    digest = hashlib.sha256(payload).hexdigest()
    name = "test.safetensors"
    monkeypatch.setattr(module, "CHECKPOINT_SHARDS", [(name, len(payload), digest)])
    monkeypatch.setattr(module, "POLICY_CHECKPOINT", ("policy.pth", 1, "0" * 64))
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda _request, timeout: io.BytesIO(payload),
    )
    adapter = module.OpenHelixCalvinAdapter.__new__(module.OpenHelixCalvinAdapter)
    adapter.checkpoint = tmp_path / "checkpoint"
    adapter.stream_checkpoint = True
    adapter.stream_cache = tmp_path / "must-remain-empty"
    adapter.checkpoint_receipt = tmp_path / "receipt.json"

    path, descriptor = adapter._checkpoint_shard(name, len(payload), digest)
    assert descriptor is not None
    try:
        assert path.read_bytes() == payload
        assert not adapter.stream_cache.exists()
        record = json.loads(adapter.checkpoint_receipt.read_text())
        assert record["files"][name]["source"] == "pinned_remote_memfd_stream"
        seals = module.fcntl.fcntl(descriptor, module.fcntl.F_GET_SEALS)
        assert seals & module.fcntl.F_SEAL_WRITE
    finally:
        os.close(descriptor)
    assert not path.exists()


def test_openhelix_live_planner_image_cast_is_lossless_and_fail_closed() -> None:
    from scripts.model_invariance.adapters.openhelix_calvin import _planner_image_uint8

    values = (np.arange(6 * 7 * 3, dtype=np.int64) % 256).reshape(6, 7, 3)
    converted = _planner_image_uint8(values)
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted.astype(np.int64), values)
    with pytest.raises(ValueError, match="not integral"):
        _planner_image_uint8(values.astype(np.float32))
    with pytest.raises(ValueError, match="exceed uint8"):
        _planner_image_uint8(np.full((2, 2, 3), 256, dtype=np.int64))
    with pytest.raises(ValueError, match="shape differs"):
        _planner_image_uint8(np.zeros((2, 2), dtype=np.uint8))

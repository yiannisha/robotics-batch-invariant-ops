from __future__ import annotations

import hashlib

import numpy as np

from scripts.run_openhelix_paired_calvin import (
    DYNAMIC_BATCH_SCHEDULE,
    array_bundle_sha256,
    request_seed,
    target_position,
)
from scripts.summarize_openhelix_paired_calvin import first_divergence, paired_bootstrap


def test_request_seed_is_condition_independent_and_request_specific() -> None:
    keys = [
        request_seed(sequence, subtask, policy_call)
        for sequence in range(3)
        for subtask in range(5)
        for policy_call in range(7)
    ]
    assert len(keys) == len(set(keys))
    assert keys == [
        request_seed(sequence, subtask, policy_call)
        for sequence in range(3)
        for subtask in range(5)
        for policy_call in range(7)
    ]


def test_dynamic_schedule_and_target_positions_cover_declared_contract() -> None:
    assert DYNAMIC_BATCH_SCHEDULE == (1, 2, 4, 8)
    for batch_size in DYNAMIC_BATCH_SCHEDULE:
        positions = [target_position(batch_size, call) for call in range(3)]
        assert positions == (
            [0, 0, 0] if batch_size == 1
            else [0, batch_size // 2, batch_size - 1]
        )


def test_array_bundle_hash_binds_names_shapes_dtypes_and_values() -> None:
    base = {"noise": np.arange(12, dtype=np.float32).reshape(3, 4)}
    digest = array_bundle_sha256(base)
    assert len(digest) == 64
    assert digest == array_bundle_sha256({"noise": base["noise"].copy()})
    assert digest != array_bundle_sha256({"other": base["noise"]})
    assert digest != array_bundle_sha256({"noise": base["noise"].astype(np.float64)})
    changed = base["noise"].copy()
    changed[0, 0] = 1
    assert digest != array_bundle_sha256({"noise": changed})
    assert all(character in "0123456789abcdef" for character in digest)


def test_adapter_request_noise_repeats_exactly_and_changes_by_seed(monkeypatch) -> None:
    monkeypatch.setenv("OPENHELIX_SOURCE", "/workspace/models/openhelix")
    from scripts.model_invariance.adapters.openhelix_calvin import OpenHelixCalvinAdapter

    adapter = OpenHelixCalvinAdapter()
    left = adapter.request_noise(91234)
    repeated = adapter.request_noise(91234)
    right = adapter.request_noise(91235)
    assert set(left) == {"noise", "position_step_noise", "rotation_step_noise"}
    assert left["noise"].shape == (19, 9)
    assert left["position_step_noise"].shape == (25, 19, 3)
    assert left["rotation_step_noise"].shape == (25, 19, 6)
    for key in left:
        assert left[key].dtype.is_floating_point
        assert left[key].equal(repeated[key])
    assert any(not left[key].equal(right[key]) for key in left)
    payload = b"".join(left[key].numpy().tobytes() for key in sorted(left))
    assert len(hashlib.sha256(payload).hexdigest()) == 64


def test_openhelix_paired_bootstrap_is_seeded_and_exact_for_constant_difference() -> None:
    result = paired_bootstrap([1] * 50, 123)
    assert result == {
        "method": "paired_percentile_bootstrap",
        "resamples": 10_000,
        "seed": 123,
        "difference": 1.0,
        "ci95": [1.0, 1.0],
    }


def test_first_divergence_checks_values_and_trajectory_length() -> None:
    left = np.zeros((3, 7), dtype=np.float32)
    right = left.copy()
    assert first_divergence(left, right) is None
    right[2, 4] = 1
    assert first_divergence(left, right) == 2
    assert first_divergence(left, left[:2]) == 2

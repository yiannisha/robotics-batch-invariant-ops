import gzip
import json
from pathlib import Path

import pytest

from scripts.audit_uva_paired_prefix import EXPECTED, PACKAGES, audit


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_probe(root: Path) -> None:
    manifest = {
        **EXPECTED,
        "schema_version": 1,
        "model": "uva",
        "path": "action",
        "condition": "native_singleton",
        "dependency_versions": PACKAGES,
        "episodes_per_task": 1,
        "task_ids": [0],
        "seed": 20261005,
        "batch_size": 1,
        "invariant_ops": False,
        "task_suite": "libero_10",
        "episode_length": 500,
        "n_obs_steps": 16,
        "n_action_steps": 8,
        "control_mode": "absolute",
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "genuine_batched_policy_call": True,
        "evidence_tier": "checkpoint_matched_probe",
    }
    episode = {
        "task": 0,
        "task_group": "libero_10",
        "task_description": "test task",
        "initialization_id": 0,
        "reset_seed": 20261005,
        "initial_observation_sha256": "a" * 64,
        "first_noise_sha256": "b" * 64,
        "success": True,
        "trajectory": [[0.0] * 8],
        "action_sequence": [[0.0] * 7],
        "rewards": [1.0],
        "done": [True],
        "policy_batch_sizes": [1],
        "inference_s": [0.5],
    }
    write(root / "condition_manifest.json", manifest)
    write(root / "rollouts/libero_10_0_initializations_0_0.json", {
        "task_id": 0,
        "task_group": "libero_10",
        "seeds": [20261005],
        "explicit_init_state_ids": [0],
        "episodes": [episode],
    })
    write(root / "official/eval_info.json", {
        "overall": {"n_episodes": 1, "n_success": 1},
    })


def test_accepts_complete_checkpoint_matched_probe(tmp_path: Path) -> None:
    make_probe(tmp_path)
    result = audit(tmp_path)
    assert result["integrity_passed"] is True
    assert result["complete"] is True
    assert result["episodes_cached"] == 1
    assert result["successes_cached"] == 1


def test_accepts_losslessly_compressed_rollout(tmp_path: Path) -> None:
    make_probe(tmp_path)
    source = tmp_path / "rollouts/libero_10_0_initializations_0_0.json"
    compressed = source.with_suffix(".json.gz")
    compressed.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    source.unlink()
    result = audit(tmp_path)
    assert result["integrity_passed"] is True
    assert result["complete"] is True
    assert result["rollout_files"] == 1


def test_rejects_malformed_action_trajectory(tmp_path: Path) -> None:
    make_probe(tmp_path)
    path = tmp_path / "rollouts/libero_10_0_initializations_0_0.json"
    value = json.loads(path.read_text())
    value["episodes"][0]["action_sequence"][0].append(0.0)
    write(path, value)
    with pytest.raises(ValueError, match="trajectory is malformed"):
        audit(tmp_path)


def test_rejects_summary_success_not_proven_by_rewards(tmp_path: Path) -> None:
    make_probe(tmp_path)
    path = tmp_path / "rollouts/libero_10_0_initializations_0_0.json"
    value = json.loads(path.read_text())
    value["episodes"][0]["rewards"] = [0.0]
    write(path, value)
    with pytest.raises(ValueError, match="success does not match rewards"):
        audit(tmp_path)

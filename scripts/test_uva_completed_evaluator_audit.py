import json
from pathlib import Path

from scripts.audit_full_cross_model_campaign import (
    UVA_EVALUATOR_SHA256,
    UVA_LIBERO10_CHECKPOINT_SHA256,
    UVA_LIBERO_ENVIRONMENT_SHA256,
    UVA_LIBERO_RUNNER_SHA256,
    UVA_PACKAGES,
    UVA_POLICY_SHA256,
    UVA_REPOSITORY_COMMIT,
    UVA_ROTATION_CONVERSIONS_SHA256,
    UVA_TASK_DESCRIPTIONS,
    official_success_row_matches_raw,
    sha256,
    uva_evaluator_evidence_audit,
)


CONDITIONS = (
    "native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic",
)
START_SEED = 20261005


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def build_campaign(root: Path, identifier: str = "uva") -> tuple[dict, dict, dict]:
    path_kind = "joint" if identifier == "uva_joint" else "action"
    checkpoint = Path("/workspace/assets/uva/libero10.ckpt")
    wrapper = Path(__file__).with_name("run_uva_paired_libero.py")
    adapter = Path(__file__).parent / "model_invariance/adapters/uva_libero10.py"
    task_manifest = root / "evaluator_task_manifest.json"
    write_json(task_manifest, {
        "schema_version": 1,
        "task_suite": "libero_10",
        "task_ids": list(range(10)),
        "task_descriptions": UVA_TASK_DESCRIPTIONS,
        "initializations_per_task": 50,
        "checkpoint_weight_path": str(checkpoint.resolve()),
        "checkpoint_weight_sha256": UVA_LIBERO10_CHECKPOINT_SHA256,
        "checkpoint_repository_revision": UVA_REPOSITORY_COMMIT,
    })
    record = {
        "evaluator_task_manifest_path": str(task_manifest),
        "official_evaluator_source_path": "/workspace/models/uva/eval_sim.py",
    }
    groups: dict[tuple[int, int], dict[str, dict]] = {
        (task, initialization): {}
        for task in range(10) for initialization in range(50)
    }
    manifest_sources = {}
    for condition in CONDITIONS:
        batch_size = 1 if condition.endswith("singleton") else 4
        condition_root = root / condition
        manifest_path = condition_root / "condition_manifest.json"
        write_json(manifest_path, {
            "schema_version": 1,
            "model": identifier,
            "path": path_kind,
            "condition": condition,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": UVA_LIBERO10_CHECKPOINT_SHA256,
            "official_repo_commit": UVA_REPOSITORY_COMMIT,
            "official_evaluator_sha256": UVA_EVALUATOR_SHA256,
            "official_policy_sha256": UVA_POLICY_SHA256,
            "official_libero_runner_sha256": UVA_LIBERO_RUNNER_SHA256,
            "official_libero_environment_sha256": UVA_LIBERO_ENVIRONMENT_SHA256,
            "rotation_conversions_sha256": UVA_ROTATION_CONVERSIONS_SHA256,
            "paired_adapter_sha256": sha256(adapter),
            "paired_wrapper_sha256": sha256(wrapper),
            "dependency_versions": UVA_PACKAGES,
            "python": "3.12.3",
            "torch": "2.8.0+cu128",
            "episodes_per_task": 50,
            "task_ids": list(range(10)),
            "seed": START_SEED,
            "batch_size": batch_size,
            "invariant_ops": condition.startswith("patched"),
            "task_suite": "libero_10",
            "episode_length": 500,
            "n_obs_steps": 16,
            "n_action_steps": 8,
            "control_mode": "absolute",
            "explicit_init_state_by_initialization_id": True,
            "request_local_noise": True,
            "genuine_batched_policy_call": True,
            "evidence_tier": "full_paired_condition",
        })
        manifest_sources[condition] = {
            "path": str(manifest_path), "sha256": sha256(manifest_path),
        }
        for task in range(10):
            for begin in range(0, 50, batch_size):
                ids = list(range(begin, min(begin + batch_size, 50)))
                episodes = []
                for initialization in ids:
                    episode = {
                        "task": task,
                        "task_group": "libero_10",
                        "task_description": UVA_TASK_DESCRIPTIONS[task],
                        "initialization_id": initialization,
                        "reset_seed": START_SEED + initialization,
                        "policy_batch_sizes": [len(ids)],
                        "inference_s": [0.1],
                    }
                    if identifier == "uva_joint":
                        episode["prediction_video_sha256"] = ["a" * 64]
                    episodes.append(episode)
                rollout = condition_root / "rollouts" / f"{task}_{begin}.json"
                write_json(rollout, {
                    "task_id": task,
                    "task_group": "libero_10",
                    "explicit_init_state_ids": ids,
                    "seeds": [START_SEED + value for value in ids],
                    "episodes": episodes,
                })
                for episode in episodes:
                    key = (task, episode["initialization_id"])
                    groups[key][condition] = {
                        **episode,
                        "condition": condition,
                        "source_rollout_path": str(rollout.resolve()),
                        "explicit_init_state_id": episode["initialization_id"],
                    }
    return record, manifest_sources, groups


def run_audit(
    root: Path, record: dict, manifests: dict, groups: dict, identifier: str = "uva",
) -> list[str]:
    issues: list[str] = []
    pairs = {(task, initialization) for task in range(10) for initialization in range(50)}
    uva_evaluator_evidence_audit(
        record, identifier, Path("/workspace/assets/uva/libero10.ckpt"), manifests,
        set(CONDITIONS), set(range(10)), pairs, groups, START_SEED, issues,
    )
    return issues


def test_uva_completed_evidence_accepts_exact_physical_groups(tmp_path: Path) -> None:
    record, manifests, groups = build_campaign(tmp_path)
    assert run_audit(tmp_path, record, manifests, groups) == []


def test_uva_aggregate_only_official_successes_are_fail_closed() -> None:
    raw = [True, False, True, True]
    assert official_success_row_matches_raw(
        {"n_episodes": 4, "n_success": 3}, raw, allow_aggregate_only=True,
    ) == (True, "aggregate_only")
    assert official_success_row_matches_raw(
        {"n_episodes": 4, "n_success": 2}, raw, allow_aggregate_only=True,
    ) == (False, None)
    assert official_success_row_matches_raw(
        {"n_episodes": 4, "n_success": 3}, raw, allow_aggregate_only=False,
    ) == (False, None)


def test_uva_completed_evidence_rejects_wrong_physical_occupancy(tmp_path: Path) -> None:
    record, manifests, groups = build_campaign(tmp_path)
    rollout = tmp_path / "native_dynamic/rollouts/0_0.json"
    value = json.loads(rollout.read_text())
    value["episodes"][0]["policy_batch_sizes"] = [1]
    write_json(rollout, value)
    issues = run_audit(tmp_path, record, manifests, groups)
    assert any("raw batched request differs" in issue for issue in issues)


def test_uva_joint_completed_evidence_requires_each_video_hash(tmp_path: Path) -> None:
    record, manifests, groups = build_campaign(tmp_path, "uva_joint")
    assert run_audit(tmp_path, record, manifests, groups, "uva_joint") == []
    rollout = tmp_path / "native_singleton/rollouts/0_0.json"
    value = json.loads(rollout.read_text())
    value["episodes"][0]["prediction_video_sha256"] = ["not-a-hash"]
    write_json(rollout, value)
    issues = run_audit(tmp_path, record, manifests, groups, "uva_joint")
    assert any("raw batched request differs" in issue for issue in issues)

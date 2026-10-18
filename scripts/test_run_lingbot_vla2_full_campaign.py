from __future__ import annotations

import json

import run_lingbot_vla2_full_campaign as campaign


def test_lingbot_runner_uses_every_source_limited_observation():
    manifest = json.loads(campaign.MANIFEST.read_text())
    assert len(manifest["observations"]) == 396
    assert manifest["split_counts"] == {"diagnostic": 200, "heldout": 196}
    indexes = [int(value) for value in campaign.replay_indices(32).split(",")]
    assert len(indexes) == len(set(indexes)) == 32
    assert all(
        manifest["observations"][index]["split"] == "diagnostic" for index in indexes
    )


def test_lingbot_runner_contains_every_required_non_success_stage():
    source = campaign.Path(campaign.__file__).read_text()
    required = (
        "probe_model_large_batches.py",
        "run_model_replay_invariance.py",
        "trace_model_operator_shapes.py",
        "build_lingbot_vla2_operator_config.py",
        "run_model_causal_ablation.py",
        "freeze_model_numerics.py",
        "qualify_real_shape_matrix_ops.py",
        "benchmark_model_policy.py",
        "benchmark_model_serving.py",
        "profile_model_kernel_launches.py",
        "benchmark_model_request_queue.py",
        "audit_full_cross_model_campaign.py",
    )
    assert all(name in source for name in required)
    assert '"--sessions", "5"' in source
    assert '"--warmup", "20"' in source
    assert '"--repetitions", "200"' in source
    assert "serving_fixed_B{fixed}_timings.jsonl" in source
    assert "requests_fixed_B{fixed}.jsonl" in source


def test_lingbot_runner_checks_exact_checkpoint_and_waits_for_pi05():
    source = campaign.Path(campaign.__file__).read_text()
    assert len(campaign.CHECKPOINT_SHARD_HASHES) == 6
    assert "run_pi05_full_campaign.py" in source
    assert '"--batch-sizes", "16", "32"' in source
    assert '"--batch-sizes", "1", "2", "4", "8"' in source

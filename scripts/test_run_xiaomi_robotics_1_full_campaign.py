from __future__ import annotations

import json
import importlib

import run_lingbot_vla2_full_campaign as shared
import run_xiaomi_robotics_1_full_campaign as campaign


def test_xiaomi_runner_uses_complete_real_replay():
    manifest = json.loads(campaign.MANIFEST.read_text())
    assert len(manifest["observations"]) == 1000
    assert manifest["split_counts"] == {"diagnostic": 200, "heldout": 800}
    assert manifest["noise_shape"] == [1000, 3, 10, 60]
    assert manifest["state_contract"].startswith("complete real 25-D dataset state")


def test_xiaomi_runner_configures_every_full_campaign_stage():
    try:
        campaign.configure_shared_driver()
        source = campaign.Path(shared.__file__).read_text()
        required = (
            "probe_model_large_batches.py", "run_model_replay_invariance.py",
            "trace_model_operator_shapes.py", "run_model_causal_ablation.py",
            "freeze_model_numerics.py", "qualify_real_shape_matrix_ops.py",
            "benchmark_model_policy.py", "benchmark_model_serving.py",
            "profile_model_kernel_launches.py", "benchmark_model_request_queue.py",
            "audit_full_cross_model_campaign.py",
        )
        assert all(name in source for name in required)
        assert shared.MODEL_ID == campaign.MODEL_ID
        assert shared.OPERATOR_BUILDER == "scripts/build_xiaomi_robotics_1_operator_config.py"
        assert shared.FLOW_STEPS == 5
        assert len(shared.FROZEN_INPUT_FILES) == 104
        assert '"--sessions", "5"' in source
        assert '"--warmup", "20"' in source
        assert '"--repetitions", "200"' in source
    finally:
        importlib.reload(shared)


def test_xiaomi_runner_preflight_accepts_pinned_assets():
    campaign.verify_assets()

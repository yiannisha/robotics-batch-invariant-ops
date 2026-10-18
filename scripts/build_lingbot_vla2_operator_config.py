"""Build LingBot-VLA2's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256
from build_rdt_operator_config import matrix_configurations, reduction_configurations
from build_spatialvla_operator_config import convolution_configurations


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.lingbot_vla2"
EXPECTED_EXECUTION_CONFIGURATION = {
    "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "image_shape_hwc": [480, 640, 3],
    "state_dim": 14,
    "flow_steps": 10,
    "action_chunk_size": 50,
    "model_action_dim": 55,
    "source_limit_observations": 396,
    "precision": "float32",
    "deterministic_moe": "deterministic_token_choice_moe",
}


def validate_trace(path: Path, batch_size: int, implementation: str) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("adapter") != EXPECTED_ADAPTER
        or report.get("adapter_execution_configuration")
        != EXPECTED_EXECUTION_CONFIGURATION
        or report.get("implementation") != implementation
        or report.get("batch_size") != batch_size
        or len(report.get("request_ids") or []) != batch_size
        or not report.get("replay_manifest_sha256")
        or not report.get("gpu")
        or not report.get("torch")
    ):
        raise ValueError(
            f"malformed or foreign LingBot-VLA2 {implementation} B{batch_size} trace: {path}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-b1", type=Path, required=True)
    parser.add_argument("--trace-b8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trace_b1 = validate_trace(args.trace_b1, 1, "native")
    trace_b8 = validate_trace(args.trace_b8, 8, "batch_invariant")
    if trace_b1["replay_manifest_sha256"] != trace_b8["replay_manifest_sha256"]:
        raise ValueError("LingBot-VLA2 traces use different frozen replay manifests")
    shards = sorted(args.checkpoint.glob("model-*-of-*.safetensors"))
    if len(shards) != 6:
        raise FileNotFoundError("LingBot-VLA2 exact six checkpoint shards are missing")

    configurations = convolution_configurations(
        trace_b8, args.trace_b8, id_prefix="lingbot_vla2", source_name="LingBot-VLA2"
    )
    configurations.extend(matrix_configurations(
        trace_b8, id_prefix="lingbot_vla2", source_name="LingBot-VLA2"
    ))
    configurations.extend(reduction_configurations(
        trace_b8, args.trace_b8, id_prefix="lingbot_vla2", source_name="LingBot-VLA2"
    ))
    if any(
        row.get("aten_operation") == "aten::scaled_dot_product_attention"
        for row in trace_b8.get("operations", [])
    ):
        configurations.extend(sdpa_configurations(
            trace_b8, args.trace_b8,
            id_prefix="lingbot_vla2", source_name="LingBot-VLA2",
        ))
    if not configurations:
        raise ValueError("LingBot-VLA2 trace yielded no qualifying operators")
    if len({row["id"] for row in configurations}) != len(configurations):
        raise ValueError("LingBot-VLA2 operator configuration IDs are not unique")
    evidence = [
        repository_path(args.trace_b1),
        repository_path(args.trace_b8),
        "/workspace/models/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py",
        "/workspace/models/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py",
        "/workspace/robotics-batch-invariant-ops/scripts/model_invariance/adapters/lingbot_vla2.py",
        "/workspace/robotics-batch-invariant-ops/batch_invariant_ops/batch_invariant_ops.py",
        "/workspace/assets/lingbot_vla2/lingbotvla_cli.yaml",
        "/workspace/batch-invariant-pizero-exp/PLAN.md",
    ]
    missing = [path for path in evidence if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"LingBot-VLA2 operator source evidence missing: {missing}")
    report = {
        "schema_version": 1,
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "checkpoint_weight_shards": [str(path.resolve()) for path in shards],
        "checkpoint_weight_shards_sha256": [sha256(path) for path in shards],
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": evidence,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"configurations": len(configurations), "output": str(args.output)}))


if __name__ == "__main__":
    main()

"""Build UniVLA's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256
from build_rdt_operator_config import matrix_configurations, reduction_configurations
from build_spatialvla_operator_config import convolution_configurations


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.univla_baai"


def validate_trace(path: Path, batch_size: int, implementation: str) -> dict:
    report = json.loads(path.read_text())
    configuration = report.get("adapter_execution_configuration") or {}
    if (
        report.get("schema_version") != 1
        or report.get("adapter") != EXPECTED_ADAPTER
        or report.get("implementation") != implementation
        or report.get("batch_size") != batch_size
        or len(report.get("request_ids") or []) != batch_size
        or not report.get("replay_manifest_sha256")
        or not report.get("gpu")
        or not report.get("torch")
        or configuration.get("native_language_padding") != "longest_in_batch_right"
        or configuration.get("invariant_language_padding") != "canonical_right"
        or configuration.get("language_max_length") != 1293
        or configuration.get("language_pad_token_id") != 151643
        or configuration.get("maximum_generated_action_tokens") != 80
        or configuration.get("decoding") != "greedy_constrained_fast"
        or configuration.get("camera_resize_hw") != [200, 200]
    ):
        raise ValueError(
            f"malformed or foreign UniVLA {implementation} B{batch_size} trace: {path}"
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
        raise ValueError("UniVLA traces use different frozen replay manifests")
    if (trace_b1["adapter_execution_configuration"]
            != trace_b8["adapter_execution_configuration"]):
        raise ValueError("UniVLA traces use different execution configurations")

    configurations = matrix_configurations(
        trace_b8, id_prefix="univla", source_name="UniVLA"
    )
    configurations.extend(reduction_configurations(
        trace_b8, args.trace_b8, id_prefix="univla", source_name="UniVLA"
    ))
    if any(
        row.get("aten_operation") == "aten::convolution"
        for row in trace_b8.get("operations", [])
    ):
        configurations.extend(convolution_configurations(
            trace_b8, args.trace_b8, id_prefix="univla", source_name="UniVLA"
        ))
    if any(
        row.get("aten_operation") == "aten::scaled_dot_product_attention"
        for row in trace_b8.get("operations", [])
    ):
        configurations.extend(sdpa_configurations(
            trace_b8, args.trace_b8, id_prefix="univla", source_name="UniVLA"
        ))
    if not configurations:
        raise ValueError("UniVLA B8 trace yielded no qualifying operators")
    if len({row["id"] for row in configurations}) != len(configurations):
        raise ValueError("UniVLA operator configuration IDs are not unique")

    shards = sorted(args.checkpoint.glob("model-*-of-00004.safetensors"))
    if len(shards) != 4:
        raise FileNotFoundError("UniVLA exact four checkpoint shards are missing")
    evidence = [
        repository_path(args.trace_b1),
        repository_path(args.trace_b8),
        "/workspace/models/univla/reference/RoboVLMs/eval/libero/model_wrapper_emu.py",
        "/workspace/models/univla/reference/Emu3/emu3/mllm/modeling_emu3.py",
        "/workspace/robotics-batch-invariant-ops/scripts/model_invariance/adapters/univla_baai.py",
        "/workspace/robotics-batch-invariant-ops/batch_invariant_ops/batch_invariant_ops.py",
        "/workspace/batch-invariant-pizero-exp/PLAN.md",
    ]
    missing = [path for path in evidence if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"UniVLA operator source evidence missing: {missing}")
    report = {
        "schema_version": 1,
        "model_id": "univla_baai",
        "adapter_execution_configuration": trace_b8["adapter_execution_configuration"],
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

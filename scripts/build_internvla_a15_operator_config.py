"""Build either InternVLA-A1.5 backend's real-shape operator manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.internvla_a15"
BACKENDS = {
    "internvla_a15_standard": ("standard", None),
    "internvla_a15_optimized": (
        "optimized",
        {"backend": "optimized", "optimized_graph_cache_policy": "recapture_per_policy_call"},
    ),
}


def validate_trace(path: Path, batch_size: int, execution_configuration: dict | None) -> dict:
    report = json.loads(path.read_text())
    if (report.get("schema_version") != 1
            or report.get("adapter") != EXPECTED_ADAPTER
            or report.get("adapter_execution_configuration") != execution_configuration
            or report.get("batch_size") != batch_size
            or len(report.get("request_ids") or []) != batch_size
            or not report.get("replay_manifest_sha256")
            or not report.get("gpu") or not report.get("torch")):
        raise ValueError(f"malformed or foreign InternVLA-A1.5 B{batch_size} trace: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--trace-b1", type=Path, required=True)
    parser.add_argument("--trace-b8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    backend, execution_configuration = BACKENDS[args.model_id]
    trace_b1 = validate_trace(args.trace_b1, 1, execution_configuration)
    trace_b8 = validate_trace(args.trace_b8, 8, execution_configuration)
    if trace_b1["replay_manifest_sha256"] != trace_b8["replay_manifest_sha256"]:
        raise ValueError("B1/B8 traces use different frozen replay manifests")

    configurations = [
        {
            "id": "qwen35_vision_patch_conv3d",
            "source_operation": "InternVLA Qwen3.5-VL visual patch Conv3d",
            "source_weight_key": (
                "model.qwen3_5_with_expert.qwen3_5.model.visual.patch_embed.proj.weight"
            ),
            "operation": "conv3d", "m_per_request": 512,
            "input_per_unit": [3, 2, 16, 16],
            "weight_shape": [1024, 3, 2, 16, 16],
            "stride": [2, 16, 16], "padding": [0, 0, 0],
            "dilation": [1, 1, 1], "groups": 1, "layout": "contiguous",
        },
        {
            "id": "qwen35_depthwise_causal_conv1d",
            "source_operation": "InternVLA Qwen3.5 action-expert depthwise causal Conv1d",
            "source_weight_key": (
                "model.qwen3_5_with_expert.action_expert.layers.0.linear_attn.conv1d.weight"
            ),
            "operation": "conv1d", "m_per_request": 1,
            "input_per_unit": [6144, 650], "weight_shape": [6144, 1, 4],
            "stride": [1], "padding": [3], "dilation": [1],
            "groups": 6144, "layout": "contiguous",
        },
        {
            "id": "vision_block0_attention_projection_addmm",
            "source_operation": (
                "InternVLA first divergent leaf: qwen3_5 visual block 0 attention projection"
            ),
            "source_weight_key": (
                "model.qwen3_5_with_expert.qwen3_5.model.visual.blocks.0.attn.proj.weight"
            ),
            "operation": "addmm", "m_per_request": 512,
            "k": 1024, "n": 1024, "layout": "right_transposed",
        },
        {
            "id": "action_attention_value_bmm",
            "source_operation": (
                "InternVLA action_expert layer 19 attention-probability/value product"
            ),
            "operation": "bmm", "m_per_request": 8,
            "m": 58, "k": 708, "n": 256, "layout": "contiguous",
        },
    ]
    attention = sdpa_configurations(
        trace_b8, args.trace_b8,
        id_prefix=f"internvla_{backend}", source_name=f"InternVLA-A1.5 {backend}",
    )
    configurations.extend(attention)
    largest = max(
        attention,
        key=lambda item: item["m_per_request"] * item["query_heads"]
        * item["query_length"] * item["key_length"],
    )
    configurations.extend([
        {
            "id": "largest_attention_qk_transposed",
            "source_operation": f"QK^T decomposition of {largest['id']}",
            "operation": "bmm",
            "m_per_request": largest["m_per_request"] * largest["query_heads"],
            "m": largest["query_length"], "k": largest["head_dim"],
            "n": largest["key_length"], "layout": "right_transposed",
            "source_qkv_stride": largest["source_q_stride"],
        },
        {
            "id": "largest_attention_pv_contiguous",
            "source_operation": f"PV decomposition of {largest['id']}",
            "operation": "bmm",
            "m_per_request": largest["m_per_request"] * largest["query_heads"],
            "m": largest["query_length"], "k": largest["key_length"],
            "n": largest["value_dim"], "layout": "contiguous",
            "source_qkv_stride": largest["source_q_stride"],
        },
    ])
    model_root = Path(f"results/cross_model/{args.model_id}")
    report = {
        "schema_version": 1,
        "model_id": args.model_id,
        "backend": backend,
        "adapter_execution_configuration": execution_configuration,
        "checkpoint_weights": str(args.checkpoint.resolve()),
        "checkpoint_weights_sha256": sha256(args.checkpoint),
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            str(model_root / "first_divergence.json"),
            "research/internvla_a15/first_divergence.json",
            "research/internvla_a15/investigation.md",
            repository_path(args.trace_b1), repository_path(args.trace_b8),
            "/workspace/assets/internvla_a15/checkpoint/config.json",
            "/workspace/batch-invariant-pizero-exp/PLAN.md",
        ],
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "model_id": args.model_id,
        "configurations": len(configurations),
        "sdpa_configurations": len(attention),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()

"""Build OpenVLA-OFT's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.openvla_oft"
EXPECTED_EXECUTION_CONFIGURATION = {
    "native_language_padding": "longest_in_batch_right",
    "invariant_language_padding": "canonical_right",
    "language_max_length": 37,
    "language_pad_token_id": 32000,
}


def validate_trace(path: Path, batch_size: int) -> dict:
    report = json.loads(path.read_text())
    if (report.get("schema_version") != 1
            or report.get("adapter") != EXPECTED_ADAPTER
            or report.get("adapter_execution_configuration") != EXPECTED_EXECUTION_CONFIGURATION
            or report.get("batch_size") != batch_size
            or len(report.get("request_ids") or []) != batch_size
            or not report.get("replay_manifest_sha256")
            or not report.get("gpu") or not report.get("torch")):
        raise ValueError(f"malformed or foreign OpenVLA-OFT B{batch_size} trace: {path}")
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
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    trace_b1 = validate_trace(args.trace_b1, 1)
    trace_b8 = validate_trace(args.trace_b8, 8)
    if trace_b1["replay_manifest_sha256"] != trace_b8["replay_manifest_sha256"]:
        raise ValueError("B1/B8 traces use different frozen replay manifests")

    configurations = [
        {
            "id": "dinov2_patch_conv_contiguous",
            "source_operation": "OpenVLA-OFT DINOv2 patch projection at two images/request",
            "operation": "conv2d", "m_per_request": 2,
            "input_per_image": [3, 224, 224], "weight_shape": [1024, 3, 14, 14],
            "stride": [14, 14], "padding": [0, 0], "dilation": [1, 1],
            "groups": 1, "layout": "contiguous",
        },
        {
            "id": "dinov2_patch_conv_channels_last",
            "source_operation": "OpenVLA-OFT DINOv2 patch projection channels-last diagnostic",
            "operation": "conv2d", "m_per_request": 2,
            "input_per_image": [3, 224, 224], "weight_shape": [1024, 3, 14, 14],
            "stride": [14, 14], "padding": [0, 0], "dilation": [1, 1],
            "groups": 1, "layout": "channels_last",
        },
        {
            "id": "siglip_patch_conv_contiguous",
            "source_operation": "OpenVLA-OFT fused SigLIP patch projection at two images/request",
            "operation": "conv2d", "m_per_request": 2,
            "input_per_image": [3, 224, 224], "weight_shape": [1152, 3, 14, 14],
            "stride": [14, 14], "padding": [0, 0], "dilation": [1, 1],
            "groups": 1, "layout": "contiguous",
        },
        {
            "id": "siglip_patch_conv_channels_last",
            "source_operation": "OpenVLA-OFT fused SigLIP patch projection channels-last diagnostic",
            "operation": "conv2d", "m_per_request": 2,
            "input_per_image": [3, 224, 224], "weight_shape": [1152, 3, 14, 14],
            "stride": [14, 14], "padding": [0, 0], "dilation": [1, 1],
            "groups": 1, "layout": "channels_last",
        },
        {
            "id": "dinov2_block0_attention_projection_addmm",
            "source_operation": (
                "OpenVLA-OFT first divergent leaf: "
                "vision_backbone.featurizer.blocks.0.attn.proj"
            ),
            "operation": "addmm", "m_per_request": 2 * 261,
            "k": 1024, "n": 1024, "layout": "right_transposed",
        },
        {
            "id": "action_head_fc1_addmm",
            "source_operation": "OpenVLA-OFT released L1 action head fc1 projection",
            "operation": "addmm", "m_per_request": 8,
            "k": 28672, "n": 4096, "layout": "right_transposed",
        },
    ]
    attention = sdpa_configurations(
        trace_b8, args.trace_b8,
        id_prefix="openvla_oft", source_name="OpenVLA-OFT",
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
    shards = sorted(args.checkpoint.glob("model-*.safetensors"))
    if len(shards) != 4:
        raise ValueError("expected the exact four-shard OpenVLA-OFT checkpoint")
    report = {
        "schema_version": 1,
        "checkpoint_weight_shards": [str(path.resolve()) for path in shards],
        "checkpoint_weight_shards_sha256": [sha256(path) for path in shards],
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            "results/cross_model/openvla_oft/first_divergence.json",
            "results/cross_model/openvla_oft/performance/operator_session_1.json",
            repository_path(args.trace_b1), repository_path(args.trace_b8),
            str((args.checkpoint / "config.json").resolve()),
            "/workspace/batch-invariant-pizero-exp/PLAN.md",
        ],
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "configurations": len(configurations),
        "sdpa_configurations": len(attention),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()

"""Build SmolVLA's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.lerobot_smolvla"
EXPECTED_EXECUTION_CONFIGURATION = {
    "native_language_padding": "longest",
    "invariant_language_padding": "max_length",
    "language_max_length": 48,
    "language_pad_token_id": 2,
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
        raise ValueError(f"malformed or foreign SmolVLA B{batch_size} trace: {path}")
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
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    trace_b1 = validate_trace(args.trace_b1, 1)
    trace_b8 = validate_trace(args.trace_b8, 8)
    if trace_b1["replay_manifest_sha256"] != trace_b8["replay_manifest_sha256"]:
        raise ValueError("B1/B8 traces use different frozen replay manifests")

    configurations = [
        {
            "id": "connector_mm_contiguous",
            "source_operation": "SmolVLM modality connector projection",
            "operation": "mm", "m_per_request": 64,
            "k": 12288, "n": 960, "layout": "contiguous",
        },
        {
            "id": "connector_mm_right_transposed",
            "source_operation": "SmolVLM modality connector projection",
            "operation": "mm", "m_per_request": 64,
            "k": 12288, "n": 960, "layout": "right_transposed",
        },
        {
            "id": "connector_mm_left_strided",
            "source_operation": "SmolVLM modality connector non-contiguous diagnostic",
            "operation": "mm", "m_per_request": 64,
            "k": 12288, "n": 960, "layout": "left_strided",
        },
        {
            "id": "state_projection_addmm",
            "source_operation": "SmolVLA state projection linear",
            "operation": "addmm", "m_per_request": 1,
            "k": 32, "n": 960, "layout": "right_transposed",
        },
        {
            "id": "vision_q_projection_addmm",
            "source_operation": "SmolVLM vision block 0 Q projection linear",
            "operation": "addmm", "m_per_request": 1024,
            "k": 768, "n": 768, "layout": "right_transposed",
        },
        {
            "id": "vision_q_projection_left_strided",
            "source_operation": "SmolVLM vision block 0 Q projection non-contiguous diagnostic",
            "operation": "addmm", "m_per_request": 1024,
            "k": 768, "n": 768, "layout": "left_strided",
        },
        {
            "id": "siglip_patch_conv_contiguous",
            "source_operation": "SmolVLM SigLIP patch projection at two images/request",
            "source_weight_key": (
                "model.vlm_with_expert.vlm.model.vision_model.embeddings."
                "patch_embedding.weight"
            ),
            "operation": "conv2d", "m_per_request": 2,
            "input_per_image": [3, 512, 512], "weight_shape": [768, 3, 16, 16],
            "stride": [16, 16], "padding": [0, 0], "dilation": [1, 1],
            "groups": 1, "layout": "contiguous",
        },
        {
            "id": "siglip_patch_conv_channels_last",
            "source_operation": "SmolVLM SigLIP patch projection channels-last diagnostic",
            "source_weight_key": (
                "model.vlm_with_expert.vlm.model.vision_model.embeddings."
                "patch_embedding.weight"
            ),
            "operation": "conv2d", "m_per_request": 2,
            "input_per_image": [3, 512, 512], "weight_shape": [768, 3, 16, 16],
            "stride": [16, 16], "padding": [0, 0], "dilation": [1, 1],
            "groups": 1, "layout": "channels_last",
        },
    ]
    attention = sdpa_configurations(
        trace_b8, args.trace_b8, id_prefix="smolvla", source_name="SmolVLA",
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
    report = {
        "schema_version": 1,
        "checkpoint_weights": str(args.checkpoint.resolve()),
        "checkpoint_weights_sha256": sha256(args.checkpoint),
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            "results/cross_model/smolvla/first_divergence.json",
            "research/smolvla/investigation.md",
            repository_path(args.trace_b1), repository_path(args.trace_b8),
            "/workspace/assets/smolvla/checkpoint/config.json",
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

"""Build PI0.5's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.lerobot_pi05"
EXPECTED_EXECUTION_CONFIGURATION = {
    "language_padding": "max_length_right",
    "language_max_length": 200,
    "camera_features": [
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    ],
    "missing_camera_contract": "official_zero_image_false_mask",
    "flow_steps": 10,
    "chunk_size": 50,
    "max_action_dim": 32,
}


def validate_trace(path: Path, batch_size: int) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("adapter") != EXPECTED_ADAPTER
        or report.get("adapter_execution_configuration")
        != EXPECTED_EXECUTION_CONFIGURATION
        or report.get("batch_size") != batch_size
        or len(report.get("request_ids") or []) != batch_size
        or not report.get("replay_manifest_sha256")
        or not report.get("gpu")
        or not report.get("torch")
    ):
        raise ValueError(f"malformed or foreign PI0.5 B{batch_size} trace: {path}")
    return report


def linear(
    identifier: str,
    source: str,
    operation: str,
    rows: int,
    inner: int,
    output: int,
    layout: str = "right_transposed",
) -> dict:
    return {
        "id": identifier,
        "source_operation": source,
        "operation": operation,
        "m_per_request": rows,
        "k": inner,
        "n": output,
        "layout": layout,
    }


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

    # Shapes come from the exact LeRobot PI0.5 architecture: three official
    # camera slots at 224px, SigLIP 1152-wide patch tokens, a 2048-wide
    # PaliGemma prefix, and a 1024-wide action expert over 50x32 chunks.
    configurations = [
        linear(
            "vision_q_projection_addmm",
            "PI0.5 SigLIP block-0 Q projection for three 256-token camera streams",
            "addmm", 3 * 256, 1152, 1152,
        ),
        linear(
            "vision_q_projection_left_strided",
            "PI0.5 SigLIP block-0 Q projection non-contiguous diagnostic",
            "addmm", 3 * 256, 1152, 1152, "left_strided",
        ),
        linear(
            "vision_multimodal_projection_addmm",
            "PI0.5 PaliGemma multimodal projector",
            "addmm", 3 * 256, 1152, 2048,
        ),
        linear(
            "prefix_q_projection_mm",
            "PI0.5 PaliGemma prefix Q projection over image and 200 language tokens",
            "mm", 3 * 256 + 200, 2048, 2048, "right_transposed",
        ),
        linear(
            "action_input_projection_addmm",
            "PI0.5 50-step noisy-action input projection",
            "addmm", 50, 32, 1024,
        ),
        linear(
            "expert_q_projection_mm",
            "PI0.5 Gemma action-expert Q projection",
            "mm", 50, 1024, 2048, "right_transposed",
        ),
        linear(
            "action_output_projection_addmm",
            "PI0.5 50-step velocity output projection",
            "addmm", 50, 1024, 32,
        ),
        {
            "id": "siglip_patch_conv_contiguous",
            "source_operation": "PI0.5 SigLIP patch projection at three camera slots/request",
            "operation": "conv2d",
            "m_per_request": 3,
            "input_per_image": [3, 224, 224],
            "weight_shape": [1152, 3, 14, 14],
            "stride": [14, 14],
            "padding": [0, 0],
            "dilation": [1, 1],
            "groups": 1,
            "layout": "contiguous",
        },
        {
            "id": "siglip_patch_conv_channels_last",
            "source_operation": "PI0.5 SigLIP patch projection channels-last diagnostic",
            "operation": "conv2d",
            "m_per_request": 3,
            "input_per_image": [3, 224, 224],
            "weight_shape": [1152, 3, 14, 14],
            "stride": [14, 14],
            "padding": [0, 0],
            "dilation": [1, 1],
            "groups": 1,
            "layout": "channels_last",
        },
    ]
    attention = sdpa_configurations(
        trace_b8, args.trace_b8, id_prefix="pi05", source_name="PI0.5"
    )
    if not attention:
        raise ValueError("PI0.5 trace contains no exercised attention shape")
    configurations.extend(attention)
    largest = max(
        attention,
        key=lambda item: item["m_per_request"]
        * item["query_heads"]
        * item["query_length"]
        * item["key_length"],
    )
    configurations.extend(
        [
            {
                "id": "largest_attention_qk_transposed",
                "source_operation": f"QK^T decomposition of {largest['id']}",
                "operation": "bmm",
                "m_per_request": largest["m_per_request"] * largest["query_heads"],
                "m": largest["query_length"],
                "k": largest["head_dim"],
                "n": largest["key_length"],
                "layout": "right_transposed",
                "source_qkv_stride": largest["source_q_stride"],
            },
            {
                "id": "largest_attention_pv_contiguous",
                "source_operation": f"PV decomposition of {largest['id']}",
                "operation": "bmm",
                "m_per_request": largest["m_per_request"] * largest["query_heads"],
                "m": largest["query_length"],
                "k": largest["key_length"],
                "n": largest["value_dim"],
                "layout": "contiguous",
                "source_qkv_stride": largest["source_q_stride"],
            },
        ]
    )
    report = {
        "schema_version": 1,
        "checkpoint_weights": str(args.checkpoint.resolve()),
        "checkpoint_weights_sha256": sha256(args.checkpoint),
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            repository_path(args.trace_b1),
            repository_path(args.trace_b8),
            "/workspace/models/lerobot/src/lerobot/policies/pi05/modeling_pi05.py",
            "/workspace/assets/pi05/checkpoint/config.json",
            "/workspace/batch-invariant-pizero-exp/PLAN.md",
        ],
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "configurations": len(configurations),
                "sdpa_configurations": len(attention),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()

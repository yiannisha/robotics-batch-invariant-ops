"""Build DexVLA's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256
from build_rdt_operator_config import matrix_configurations, reduction_configurations
from build_spatialvla_operator_config import convolution_configurations


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.dexvla"
EXPECTED_EXECUTION_CONFIGURATION = {
    "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "image_shape_hwc": [480, 640, 3],
    "state_dim": 14,
    "reasoning_generation": "greedy_max_new_tokens_256",
    "diffusion_steps": 10,
    "action_chunk_size": 50,
    "action_dim": 14,
    "source_limit_observations": 396,
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
            f"malformed or foreign DexVLA {implementation} B{batch_size} trace: {path}"
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
        raise ValueError("DexVLA B1/B8 traces use different frozen replay manifests")
    shards = [
        args.checkpoint / "model-00001-of-00002.safetensors",
        args.checkpoint / "model-00002-of-00002.safetensors",
    ]
    if any(not path.is_file() for path in shards):
        raise FileNotFoundError("DexVLA checkpoint shards are missing")

    configurations = convolution_configurations(
        trace_b8, args.trace_b8, id_prefix="dexvla", source_name="DexVLA"
    )
    configurations.extend(matrix_configurations(
        trace_b8, id_prefix="dexvla", source_name="DexVLA"
    ))
    configurations.extend(reduction_configurations(
        trace_b8, args.trace_b8, id_prefix="dexvla", source_name="DexVLA"
    ))
    attention = sdpa_configurations(
        trace_b8, args.trace_b8, id_prefix="dexvla", source_name="DexVLA"
    )
    configurations.extend(attention)

    # Qwen2-VL packs three 1,564-token images per request.  The repaired path
    # splits those exact cu_seqlens segments, then executes the established
    # invariant BMM/softmax/BMM attention pipeline.  Qualify both actual
    # attention matrix products explicitly, not a smaller surrogate.
    configurations.extend([
        {
            "id": "dexvla_vision_segment_qk_transposed",
            "source_operation": (
                "DexVLA Qwen2-VL varlen vision attention QK^T for three "
                "1,564-token images/request"
            ),
            "operation": "bmm",
            "m_per_request": 3 * 16,
            "m": 1564,
            "k": 80,
            "n": 1564,
            "layout": "right_transposed",
        },
        {
            "id": "dexvla_vision_segment_pv_contiguous",
            "source_operation": (
                "DexVLA Qwen2-VL varlen vision attention PV for three "
                "1,564-token images/request"
            ),
            "operation": "bmm",
            "m_per_request": 3 * 16,
            "m": 1564,
            "k": 1564,
            "n": 80,
            "layout": "contiguous",
        },
        {
            "id": "dexvla_scaledp_conditioning_addmm",
            "source_operation": (
                "DexVLA ScaleDP first conditioning projection: 1,536D FiLM + 14D state"
            ),
            "operation": "addmm",
            "m_per_request": 1,
            "k": 1550,
            "n": 1024,
            "layout": "right_transposed",
        },
    ])
    if len({row["id"] for row in configurations}) != len(configurations):
        raise ValueError("DexVLA operator configuration IDs are not unique")
    evidence = [
        repository_path(args.trace_b1),
        repository_path(args.trace_b8),
        "/workspace/models/dexvla/qwen2_vla/models/modeling_qwen2_vla.py",
        "/workspace/models/dexvla/policy_heads/models/transformer_diffusion/modeling_scaledp.py",
        "/workspace/robotics-batch-invariant-ops/scripts/model_invariance/adapters/dexvla.py",
        "/workspace/robotics-batch-invariant-ops/batch_invariant_ops/batch_invariant_ops.py",
        "/workspace/assets/dexvla/repo/checkpoint-60000/config2.json",
        "/workspace/batch-invariant-pizero-exp/PLAN.md",
    ]
    if any(not Path(path).is_file() for path in evidence):
        missing = [path for path in evidence if not Path(path).is_file()]
        raise FileNotFoundError(f"DexVLA operator source evidence missing: {missing}")
    report = {
        "schema_version": 1,
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "checkpoint_weight_shards": [str(path.resolve()) for path in shards],
        "checkpoint_weight_shards_sha256": [sha256(path) for path in shards],
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": evidence,
        "packed_vision_attention_contract": {
            "images_per_request": 3,
            "tokens_per_image": 1564,
            "heads": 16,
            "head_dim": 80,
        },
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

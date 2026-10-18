"""Build CogACT-Small's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256
from build_rdt_operator_config import matrix_configurations, reduction_configurations
from build_spatialvla_operator_config import convolution_configurations


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.cogact_small"
EXPECTED_EXECUTION_CONFIGURATION = {
    "native_language_padding": "longest_in_batch_right",
    "invariant_language_padding": "canonical_right",
    "canonical_language_tokens": 27,
    "prompt_builder": "PurePromptBuilder",
    "cfg_scale": 1.5,
    "ddim_steps": 10,
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
        or not report.get("gpu") or not report.get("torch")
    ):
        raise ValueError(f"malformed or foreign CogACT B{batch_size} trace: {path}")
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
        raise ValueError("CogACT B1/B8 traces use different frozen replay manifests")

    configurations = convolution_configurations(
        trace_b8, args.trace_b8, id_prefix="cogact", source_name="CogACT-Small"
    )
    configurations.extend(matrix_configurations(
        trace_b8, id_prefix="cogact", source_name="CogACT-Small"
    ))
    configurations.extend(reduction_configurations(
        trace_b8, args.trace_b8, id_prefix="cogact", source_name="CogACT-Small"
    ))
    attention = sdpa_configurations(
        trace_b8, args.trace_b8, id_prefix="cogact", source_name="CogACT-Small"
    )
    configurations.extend(attention)
    largest = max(
        attention,
        key=lambda item: item["m_per_request"] * item["query_heads"]
        * item["query_length"] * item["key_length"],
    )
    configurations.extend([
        {
            "id": "cogact_largest_attention_qk_transposed",
            "source_operation": f"QK^T decomposition of {largest['id']}",
            "operation": "bmm",
            "m_per_request": largest["m_per_request"] * largest["query_heads"],
            "m": largest["query_length"], "k": largest["head_dim"],
            "n": largest["key_length"], "layout": "right_transposed",
            "source_qkv_stride": largest["source_q_stride"],
        },
        {
            "id": "cogact_largest_attention_pv_contiguous",
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
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "checkpoint_weights": str(args.checkpoint.resolve()),
        "checkpoint_weights_sha256": sha256(args.checkpoint),
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            repository_path(args.trace_b1), repository_path(args.trace_b8),
            "/workspace/models/cogact/vla/cogactvla.py",
            "/workspace/models/cogact-openvla/prismatic/models/vlms/prismatic.py",
            "/workspace/models/cogact-openvla/prismatic/models/backbones/vision/dinosiglip_vit.py",
        ],
        "configurations": configurations,
    }
    if any(not Path(path).is_file() for path in report["source_evidence"]):
        missing = [path for path in report["source_evidence"] if not Path(path).is_file()]
        raise FileNotFoundError(f"CogACT operator source evidence missing: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "configurations": len(configurations), "sdpa_configurations": len(attention),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()

"""Build VITRA-VLA-3B's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256
from build_rdt_operator_config import matrix_configurations, reduction_configurations
from build_spatialvla_operator_config import convolution_configurations


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.vitra_vla_3b"
EXPECTED_EXECUTION_CONFIGURATION = {
    "source": "microsoft/VITRA-TeleData",
    "source_episodes": 50,
    "source_image_shape_hwc": [720, 1280, 3],
    "model_image_shape_hwc": [224, 398, 3],
    "real_xhand_state_dim": 18,
    "human_statistics_state_dim": 122,
    "padded_model_state_dim": 212,
    "canonical_language_tokens": 276,
    "ddim_steps": 10,
    "action_chunk_size": 16,
    "model_action_dim": 192,
    "robot_output_dim": 102,
    "precision": "float32",
}
CHECKPOINT_BYTES = 15_067_578_877
CHECKPOINT_SHA256 = "7f57dfd4451d68f41accab8a06bb388d6fdc2cce87ee8b31fafa7c8f01d83796"


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
            f"malformed or foreign VITRA {implementation} B{batch_size} trace: {path}"
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
        raise ValueError("VITRA traces use different frozen replay manifests")
    checkpoint = args.checkpoint / "vitra-vla-3b.pt"
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != CHECKPOINT_BYTES
        or sha256(checkpoint) != CHECKPOINT_SHA256
    ):
        raise FileNotFoundError("VITRA exact checkpoint is missing or changed")

    configurations = convolution_configurations(
        trace_b8, args.trace_b8, id_prefix="vitra", source_name="VITRA-VLA-3B"
    )
    configurations.extend(matrix_configurations(
        trace_b8, id_prefix="vitra", source_name="VITRA-VLA-3B"
    ))
    configurations.extend(reduction_configurations(
        trace_b8, args.trace_b8, id_prefix="vitra", source_name="VITRA-VLA-3B"
    ))
    if any(
        row.get("aten_operation") == "aten::scaled_dot_product_attention"
        for row in trace_b8.get("operations", [])
    ):
        configurations.extend(sdpa_configurations(
            trace_b8, args.trace_b8,
            id_prefix="vitra", source_name="VITRA-VLA-3B",
        ))
    if not configurations:
        raise ValueError("VITRA trace yielded no qualifying operators")
    if len({row["id"] for row in configurations}) != len(configurations):
        raise ValueError("VITRA operator configuration IDs are not unique")
    evidence = [
        repository_path(args.trace_b1),
        repository_path(args.trace_b8),
        "/workspace/models/vitra/vitra/models/vla/vitra_paligemma.py",
        "/workspace/models/vitra/vitra/models/action_model/dit.py",
        "/workspace/models/vitra/vitra/models/action_model/gaussian_diffusion.py",
        "/workspace/robotics-batch-invariant-ops/scripts/model_invariance/adapters/vitra_vla_3b.py",
        "/workspace/robotics-batch-invariant-ops/batch_invariant_ops/batch_invariant_ops.py",
        "/workspace/batch-invariant-pizero-exp/open-pi-zero/PLAN.md",
    ]
    missing = [path for path in evidence if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"VITRA operator source evidence missing: {missing}")
    report = {
        "schema_version": 1,
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "checkpoint_weight_shards": [str(checkpoint.resolve())],
        "checkpoint_weight_shards_sha256": [CHECKPOINT_SHA256],
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": evidence,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"configurations": len(configurations), "output": str(args.output)}))


if __name__ == "__main__":
    main()

"""Build MolmoAct2's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256
from build_rdt_operator_config import matrix_configurations, reduction_configurations
from build_spatialvla_operator_config import convolution_configurations


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.molmoact2"
EXPECTED_EXECUTION_CONFIGURATION = {
    "camera_features": ["observation.images.image", "observation.images.wrist_image"],
    "source_image_shape_hwc": [256, 256, 3],
    "processed_pixel_values_shape_per_request": [2, 729, 588],
    "processed_image_token_pooling_shape_per_request": [392, 4],
    "state_dim": 8,
    "native_language_padding": "longest_in_batch_right",
    "invariant_language_padding": "max_length_right",
    "canonical_language_tokens": 497,
    "language_pad_token_id": 151643,
    "flow_steps": 10,
    "max_action_horizon": 10,
    "max_action_dim": 32,
    "robot_output_dim": 7,
    "precision": "bfloat16",
}
SHARD_HASHES = {
    "model-00001-of-00005.safetensors": "5aa56113a3c1c95f0c03b90bada5b7ac2babf59c752a3d6201b8362a530c6809",
    "model-00002-of-00005.safetensors": "1fd95aec9ad144d3f4eac42db88a000321e016beeae606ecfcb84fcbdb13ecf4",
    "model-00003-of-00005.safetensors": "e6c455434915e74a8d750bedd0fc4c23b5a8ce37954aaff9c3e3f22085a26890",
    "model-00004-of-00005.safetensors": "4e5bcf5bad81573153d6deaa3d25c3e53dc697658488b1cdbab4c8a3dffba1b6",
    "model-00005-of-00005.safetensors": "76140ad00c225f95b80eb6b6a5fea2f985e4fdde3356a3c67c93d87d083d4ee2",
}


def validate_trace(path: Path, batch_size: int, implementation: str) -> dict:
    report = json.loads(path.read_text())
    if (report.get("schema_version") != 1
            or report.get("adapter") != EXPECTED_ADAPTER
            or report.get("adapter_execution_configuration")
                != EXPECTED_EXECUTION_CONFIGURATION
            or report.get("implementation") != implementation
            or report.get("batch_size") != batch_size
            or len(report.get("request_ids") or []) != batch_size
            or not report.get("replay_manifest_sha256")
            or not report.get("gpu") or not report.get("torch")):
        raise ValueError(f"malformed or foreign MolmoAct2 trace: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-b1", type=Path, required=True)
    parser.add_argument("--trace-b8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-receipt", type=Path,
        default=Path("results/cross_model/molmoact2/full/checkpoint_stream_receipt.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trace_b1 = validate_trace(args.trace_b1, 1, "native")
    trace_b8 = validate_trace(args.trace_b8, 8, "batch_invariant")
    if trace_b1["replay_manifest_sha256"] != trace_b8["replay_manifest_sha256"]:
        raise ValueError("MolmoAct2 traces use different frozen replay manifests")
    receipt = json.loads(args.checkpoint_receipt.read_text())
    index = args.checkpoint / "model.safetensors.index.json"
    if (not index.is_file()
            or sha256(index) != "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06"):
        raise ValueError("MolmoAct2 checkpoint index differs")
    if (receipt.get("all_pinned_files_verified") is not True
            or {name: row.get("sha256") for name, row in receipt.get("files", {}).items()}
                != SHARD_HASHES
            or (receipt.get("strict_model_load") or {}).get("loaded_state_tensors") != 1295
            or (receipt.get("strict_model_load") or {}).get("missing_state_tensors") != []):
        raise ValueError("MolmoAct2 exact streamed checkpoint receipt is incomplete")
    configurations = convolution_configurations(
        trace_b8, args.trace_b8, id_prefix="molmoact2", source_name="MolmoAct2"
    )
    configurations.extend(matrix_configurations(
        trace_b8, id_prefix="molmoact2", source_name="MolmoAct2"
    ))
    configurations.extend(reduction_configurations(
        trace_b8, args.trace_b8, id_prefix="molmoact2", source_name="MolmoAct2"
    ))
    if any(row.get("aten_operation") == "aten::scaled_dot_product_attention"
           for row in trace_b8.get("operations", [])):
        configurations.extend(sdpa_configurations(
            trace_b8, args.trace_b8, id_prefix="molmoact2", source_name="MolmoAct2"
        ))
    if not configurations or len({row["id"] for row in configurations}) != len(configurations):
        raise ValueError("MolmoAct2 trace yielded no unique qualifying operators")
    evidence = [
        repository_path(args.trace_b1), repository_path(args.trace_b8),
        "/workspace/assets/molmoact2/checkpoint/modeling_molmoact2.py",
        "/workspace/assets/molmoact2/checkpoint/inference.py",
        "/workspace/robotics-batch-invariant-ops/scripts/model_invariance/adapters/molmoact2.py",
        "/workspace/robotics-batch-invariant-ops/batch_invariant_ops/batch_invariant_ops.py",
        str(args.checkpoint_receipt.resolve()),
        "/workspace/batch-invariant-pizero-exp/PLAN.md",
    ]
    if any(not Path(path).is_file() for path in evidence):
        raise FileNotFoundError("MolmoAct2 operator source evidence is missing")
    report = {
        "schema_version": 1,
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "checkpoint_stream_receipt": str(args.checkpoint_receipt.resolve()),
        "checkpoint_stream_receipt_sha256": sha256(args.checkpoint_receipt),
        "checkpoint_weight_shards_sha256": SHARD_HASHES,
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": evidence,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"configurations": len(configurations), "output": str(args.output)}))


if __name__ == "__main__":
    main()

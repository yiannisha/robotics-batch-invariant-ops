"""Build OpenHelix's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256
from build_rdt_operator_config import matrix_configurations, reduction_configurations
from build_spatialvla_operator_config import convolution_configurations


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.openhelix_calvin"
EXPECTED_EXECUTION_CONFIGURATION = {
    "source": "KyujinL/CALVIN_ABC",
    "planner_image_shape_hwc": [200, 200, 3],
    "policy_cameras": ["static", "gripper"],
    "policy_rgb_shape_bchw": [2, 3, 200, 200],
    "policy_point_cloud_shape_bchw": [2, 3, 200, 200],
    "proprio_dim": 8,
    "planner_precision": "bfloat16",
    "policy_precision": "float32",
    "ddpm_steps": 25,
    "action_chunk_size": 19,
    "robot_output_dim": 7,
    "instruction": "push the sliding door to the right side",
}
EXPECTED_FILES = {
    "model-00000.safetensors": (2_105_645_528, "7bc815c581781b9d864ff22fbfd87f84d45adda0b283609d42f5716f6b933ac7"),
    "model-00001.safetensors": (2_114_018_192, "6f015b79cea364ea356c0e11414e9dbe71f4736b2b5b8523119a34a087e0e758"),
    "model-00002.safetensors": (2_114_034_864, "2ec780a890f2554385765fe3f4beedeb6b953d7eaf6f3887ad1535e56976433f"),
    "model-00003.safetensors": (2_124_504_248, "6d552bc92746187c7d1614186e924dee8c3b456c06be03868da9d2ac07410128"),
    "model-00004.safetensors": (2_057_395_512, "247bc9b65065446a596a3c431077bdbe4126681821334c2019369aaa65f49620"),
    "model-00005.safetensors": (2_114_018_256, "4172759f8b2e57b1e0d3d8daaddea4b5489adfa546f07b4a7ce1b53db55f2e62"),
    "model-00006.safetensors": (1_504_179_160, "ee2df39c2fc516148d2d82e108f66c189f2cd769fb4e42cc333e6dca23d87810"),
    "policy.pth": (445_004_793, "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638"),
}


def validate_trace(path: Path, batch_size: int, implementation: str) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("adapter") != EXPECTED_ADAPTER
        or report.get("adapter_execution_configuration") != EXPECTED_EXECUTION_CONFIGURATION
        or report.get("implementation") != implementation
        or report.get("batch_size") != batch_size
        or len(report.get("request_ids") or []) != batch_size
        or not report.get("replay_manifest_sha256")
        or not report.get("gpu")
        or not report.get("torch")
    ):
        raise ValueError(f"malformed or foreign OpenHelix {implementation} B{batch_size} trace")
    return report


def validate_receipt(path: Path) -> dict:
    receipt = json.loads(path.read_text())
    files = receipt.get("files") or {}
    actual = {
        name: (record.get("bytes"), record.get("sha256"))
        for name, record in files.items()
    }
    if (
        receipt.get("repository") != "OpenHelix/openhelix"
        or receipt.get("revision") != "1a6fabbe148af045ed0710d007ed24c36fa83a53"
        or receipt.get("all_pinned_files_verified") is not True
        or actual != EXPECTED_FILES
        or (receipt.get("strict_planner_load") or {}).get("missing_named_tensors") != []
        or (receipt.get("strict_planner_load") or {}).get("meta_parameters_remaining") is not False
        or (receipt.get("strict_policy_load") or {}).get("missing_keys") != []
        or (receipt.get("strict_policy_load") or {}).get("unexpected_keys") != []
    ):
        raise ValueError("OpenHelix checkpoint streaming receipt is incomplete")
    return receipt


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
        raise ValueError("OpenHelix traces use different frozen replay manifests")
    receipt_path = args.checkpoint / "checkpoint_stream_receipt.json"
    validate_receipt(receipt_path)

    configurations = convolution_configurations(
        trace_b8, args.trace_b8, id_prefix="openhelix", source_name="OpenHelix"
    )
    configurations.extend(matrix_configurations(
        trace_b8, id_prefix="openhelix", source_name="OpenHelix"
    ))
    configurations.extend(reduction_configurations(
        trace_b8, args.trace_b8, id_prefix="openhelix", source_name="OpenHelix"
    ))
    if any(row.get("aten_operation") == "aten::scaled_dot_product_attention"
           for row in trace_b8.get("operations", [])):
        configurations.extend(sdpa_configurations(
            trace_b8, args.trace_b8, id_prefix="openhelix", source_name="OpenHelix"
        ))
    if not configurations or len({row["id"] for row in configurations}) != len(configurations):
        raise ValueError("OpenHelix trace yielded no unique qualifying operators")
    evidence = [
        repository_path(args.trace_b1),
        repository_path(args.trace_b8),
        repository_path(receipt_path),
        "/workspace/models/openhelix/planer.py",
        "/workspace/models/openhelix/diffuser_actor/trajectory_optimization/diffuser_actor_act_simple.py",
        "/workspace/models/openhelix/diffuser_actor/utils/multihead_custom_attention.py",
        "/workspace/robotics-batch-invariant-ops/scripts/model_invariance/adapters/openhelix_calvin.py",
        "/workspace/robotics-batch-invariant-ops/batch_invariant_ops/batch_invariant_ops.py",
        "/workspace/batch-invariant-pizero-exp/open-pi-zero/PLAN.md",
    ]
    missing = [path for path in evidence if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"OpenHelix operator source evidence missing: {missing}")
    report = {
        "schema_version": 1,
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "checkpoint_stream_receipt": str(receipt_path.resolve()),
        "checkpoint_stream_receipt_sha256": sha256(receipt_path),
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": evidence,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"configurations": len(configurations), "output": str(args.output)}))


if __name__ == "__main__":
    main()

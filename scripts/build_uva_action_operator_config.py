"""Build UVA action's pinned real-shape operator qualification manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_pi0fast_operator_config import (
    repository_path,
    sdpa_configurations,
    sha256,
    tensor,
)


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.uva_libero10_full"


def validate_trace(path: Path, batch_size: int, expected_adapter: str = EXPECTED_ADAPTER) -> dict:
    report = json.loads(path.read_text())
    if (report.get("schema_version") != 1
            or report.get("adapter") != expected_adapter
            or report.get("adapter_execution_configuration") is not None
            or report.get("batch_size") != batch_size
            or len(report.get("request_ids") or []) != batch_size
            or not report.get("replay_manifest_sha256")
            or not report.get("gpu") or not report.get("torch")):
        raise ValueError(f"malformed or foreign UVA action B{batch_size} trace: {path}")
    return report


def contiguous_stride(shape: list[int]) -> list[int]:
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * shape[index + 1]
    return stride


def convolution_layout(argument: dict) -> str:
    shape, stride = argument["shape"], argument["stride"]
    if stride == contiguous_stride(shape):
        return "contiguous"
    if len(shape) == 4:
        batch, channels, height, width = shape
        if stride == [height * width * channels, 1, width * channels, channels]:
            return "channels_last"
    if len(shape) == 5:
        batch, channels, depth, height, width = shape
        if stride == [
            depth * height * width * channels, 1,
            height * width * channels, width * channels, channels,
        ]:
            return "channels_last_3d"
    raise ValueError(f"UVA convolution uses an unsupported exact input stride: {argument}")


def convolution_configurations(trace: dict, trace_path: Path) -> list[dict]:
    configurations = []
    identities = set()
    for row in trace.get("operations", []):
        if row.get("aten_operation") != "aten::convolution":
            continue
        arguments = row.get("arguments") or []
        if len(arguments) != 9:
            raise ValueError("traced aten::convolution does not have its exact nine arguments")
        source = tensor(arguments[0], "convolution input")
        weight = tensor(arguments[1], "convolution weight")
        bias = arguments[2]
        if bias is not None:
            tensor(bias, "convolution bias")
        stride, padding, dilation = arguments[3:6]
        transposed, output_padding, groups = arguments[6:9]
        if (not all(isinstance(value, list) for value in (stride, padding, dilation, output_padding))
                or not isinstance(transposed, bool) or not isinstance(groups, int)):
            raise ValueError("traced convolution attributes are malformed")
        rank = len(source["shape"])
        if rank not in (3, 4, 5) or len(weight["shape"]) != rank:
            raise ValueError("traced convolution rank is unsupported")
        if source["shape"][0] % 8:
            raise ValueError("UVA B8 convolution does not preserve a request-leading dimension")
        if transposed and rank != 5:
            raise ValueError("UVA only has an audited 3-D transposed-convolution kernel")
        operation = "conv_transpose3d" if transposed else f"conv{rank - 2}d"
        if transposed and (
            stride != weight["shape"][-3:] or padding != [0, 0, 0]
            or output_padding != [0, 0, 0] or dilation != [1, 1, 1]
        ):
            raise ValueError("UVA transposed convolution exceeds the invariant kernel contract")
        identity = (
            operation, tuple(source["shape"]), tuple(source["stride"]),
            tuple(weight["shape"]), tuple(stride), tuple(padding), tuple(dilation),
            tuple(output_padding), groups, bias is not None,
        )
        if identity in identities:
            raise ValueError("duplicate convolution shape/layout in de-duplicated UVA trace")
        identities.add(identity)
        signature = hashlib.sha256(repr(identity).encode()).hexdigest()[:12]
        configuration = {
            "id": f"uva_{operation}_{signature}",
            "source_operation": (
                f"UVA action traced aten::convolution ({operation}) from the pinned B8 replay"
            ),
            "operation": operation,
            "m_per_request": source["shape"][0] // 8,
            "input_per_unit": source["shape"][1:],
            "weight_shape": weight["shape"],
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "has_bias": bias is not None,
            "layout": convolution_layout(source),
            "source_trace_path": repository_path(trace_path),
            "source_input_stride_B8": source["stride"],
        }
        if transposed:
            configuration["output_padding"] = output_padding
            configuration["source_state_key"] = "model.diffactloss.conv_transpose3d.weight"
        configurations.append(configuration)
    if not configurations:
        raise ValueError("UVA B8 trace contains no convolution calls")
    return configurations


def require_first_divergent_addmm(trace: dict) -> None:
    for row in trace.get("operations", []):
        if row.get("aten_operation") != "aten::addmm":
            continue
        arguments = row.get("arguments") or []
        if len(arguments) >= 3:
            left = tensor(arguments[1], "addmm input")
            right = tensor(arguments[2], "addmm right operand")
            if left["shape"] == [240, 512] and right["shape"] == [512, 2048]:
                return
    raise ValueError("UVA B8 trace does not contain the checkpoint's first-divergent CLIP fc1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-b1", type=Path, required=True)
    parser.add_argument("--trace-b8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    expected_adapter = (
        "scripts.model_invariance.adapters.uva_libero10_joint_full"
        if args.joint else EXPECTED_ADAPTER
    )
    model_id = "uva_joint" if args.joint else "uva"
    trace_b1 = validate_trace(args.trace_b1, 1, expected_adapter)
    trace_b8 = validate_trace(args.trace_b8, 8, expected_adapter)
    if trace_b1["replay_manifest_sha256"] != trace_b8["replay_manifest_sha256"]:
        raise ValueError("B1/B8 UVA traces use different frozen replay manifests")
    require_first_divergent_addmm(trace_b8)

    configurations = [{
        "id": "clip_text_layer0_fc1_addmm",
        "source_operation": "UVA first divergent leaf: CLIP text layer 0 MLP fc1",
        "source_state_key": "text_model.text_model.encoder.layers.0.mlp.fc1.weight",
        "operation": "addmm", "m_per_request": 30,
        "k": 512, "n": 2048, "layout": "right_transposed",
    }]
    convolutions = convolution_configurations(trace_b8, args.trace_b8)
    configurations.extend(convolutions)
    has_sdpa = any(
        row.get("aten_operation") == "aten::scaled_dot_product_attention"
        for row in trace_b8.get("operations", [])
    )
    attention = (
        sdpa_configurations(
            trace_b8, args.trace_b8, id_prefix="uva", source_name="UVA action",
        ) if has_sdpa else []
    )
    configurations.extend(attention)
    if attention:
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
        "model_id": model_id,
        "adapter_execution_configuration": None,
        "checkpoint_weights": str(args.checkpoint.resolve()),
        "checkpoint_weights_sha256": sha256(args.checkpoint),
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            "results/cross_model/uva/first_divergence.json",
            "research/uva/first_divergence.json",
            "research/uva/investigation.md",
            repository_path(args.trace_b1), repository_path(args.trace_b8),
            "/workspace/models/uva/unified_video_action/model/autoregressive/diffusion_action_loss.py",
            "/workspace/batch-invariant-pizero-exp/open-pi-zero/PLAN.md",
        ],
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "configurations": len(configurations),
        "convolution_configurations": len(convolutions),
        "sdpa_configurations": len(attention),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()

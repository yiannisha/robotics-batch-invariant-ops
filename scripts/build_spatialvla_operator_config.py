"""Build SpatialVLA's pinned real-shape operator qualification manifest.

The B=8 trace is authoritative for every exercised convolution, mean,
softmax, and fused-attention shape.  The manifest also includes the traced
first-divergent vision projection and the largest representable BMM attention
surface so the model-level causal frontier is covered explicitly.
"""

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


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.spatialvla"
EXPECTED_EXECUTION_CONFIGURATION = {
    "native_language_padding": "longest_in_batch_right",
    "invariant_language_padding": "canonical_right",
    "canonical_text_length": 64,
    "unnorm_key": "bridge_orig/1.0.0",
    "do_normalize": False,
    "generation": "greedy_max_new_tokens_256",
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
        raise ValueError(f"malformed or foreign SpatialVLA B{batch_size} trace: {path}")
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
        _, channels, height, width = shape
        if stride == [height * width * channels, 1, width * channels, channels]:
            return "channels_last"
    if len(shape) == 5:
        _, channels, depth, height, width = shape
        if stride == [
            depth * height * width * channels,
            1,
            height * width * channels,
            width * channels,
            channels,
        ]:
            return "channels_last_3d"
    raise ValueError(f"unsupported traced convolution input layout: {argument}")


def signature(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()[:12]


def convolution_configurations(
    trace: dict, trace_path: Path, *,
    id_prefix: str = "spatialvla", source_name: str = "SpatialVLA",
) -> list[dict]:
    configurations = []
    identities = set()
    for row in trace.get("operations", []):
        if row.get("aten_operation") != "aten::convolution":
            continue
        arguments = row.get("arguments") or []
        if len(arguments) != 9:
            raise ValueError("traced aten::convolution lacks its exact nine arguments")
        source = tensor(arguments[0], "convolution input")
        weight = tensor(arguments[1], "convolution weight")
        bias = arguments[2]
        if bias is not None:
            tensor(bias, "convolution bias")
        stride, padding, dilation = arguments[3:6]
        transposed, output_padding, groups = arguments[6:9]
        if (
            not all(
                isinstance(value, list)
                for value in (stride, padding, dilation, output_padding)
            )
            or not isinstance(transposed, bool)
            or not isinstance(groups, int)
        ):
            raise ValueError("traced convolution attributes are malformed")
        rank = len(source["shape"])
        if rank not in (3, 4, 5) or len(weight["shape"]) != rank:
            raise ValueError("traced convolution rank is unsupported")
        if source["shape"][0] < 8 or source["shape"][0] % 8:
            raise ValueError("SpatialVLA B8 convolution lost its request-leading axis")
        if transposed:
            if rank != 4:
                raise ValueError("SpatialVLA only has a qualified ConvTranspose2D kernel")
            if (
                stride != weight["shape"][-2:]
                or padding != [0, 0]
                or output_padding != [0, 0]
                or dilation != [1, 1]
            ):
                raise ValueError(
                    "SpatialVLA transposed convolution exceeds the non-overlap kernel contract"
                )
            operation = "conv_transpose2d"
        else:
            operation = f"conv{rank - 2}d"
        identity = (
            operation,
            tuple(source["shape"]),
            tuple(source["stride"]),
            tuple(weight["shape"]),
            tuple(stride),
            tuple(padding),
            tuple(dilation),
            tuple(output_padding),
            groups,
            bias is not None,
        )
        if identity in identities:
            raise ValueError("duplicate convolution signature in de-duplicated trace")
        identities.add(identity)
        configuration = {
            "id": f"{id_prefix}_{operation}_{signature(identity)}",
            "source_operation": (
                f"{source_name} traced aten::convolution ({operation}) from pinned B8 replay"
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
            "trace_call_count": row.get("count"),
        }
        if transposed:
            configuration["output_padding"] = output_padding
        configurations.append(configuration)
    if not configurations:
        raise ValueError("SpatialVLA B8 trace contains no convolution calls")
    return configurations


def reduction_configurations(
    trace: dict, trace_path: Path, *,
    id_prefix: str = "spatialvla", source_name: str = "SpatialVLA",
) -> list[dict]:
    configurations = []
    identities = set()
    for row in trace.get("operations", []):
        operation_name = row.get("aten_operation")
        if operation_name not in ("aten::mean", "aten::_softmax"):
            continue
        arguments = row.get("arguments") or []
        if len(arguments) < 2:
            raise ValueError(f"traced {operation_name} lacks input/dimension")
        source = tensor(arguments[0], f"{operation_name} input")
        if source["shape"][0] < 8 or source["shape"][0] % 8:
            raise ValueError(f"{operation_name} lost its request-leading B8 axis")
        if source["stride"] != contiguous_stride(source["shape"]):
            raise ValueError(f"{operation_name} exact traced layout is not contiguous")
        raw_dim = arguments[1]
        if operation_name == "aten::mean":
            if not isinstance(raw_dim, list) or len(raw_dim) != 1:
                raise ValueError("SpatialVLA mean is not a single-dimension reduction")
            dim = raw_dim[0]
            keepdim = arguments[2] if len(arguments) > 2 else False
            if not isinstance(keepdim, bool):
                raise ValueError("SpatialVLA mean keepdim is malformed")
            operation = "mean"
        else:
            if not isinstance(raw_dim, int):
                raise ValueError("SpatialVLA softmax dimension is malformed")
            if len(arguments) > 2 and arguments[2] is not False:
                raise ValueError("SpatialVLA softmax half_to_float contract changed")
            dim = raw_dim
            keepdim = None
            operation = "softmax"
        if dim == 0 or dim == -len(source["shape"]):
            raise ValueError(f"{operation} reduces across the request-leading axis")
        identity = (operation, tuple(source["shape"]), tuple(source["stride"]), dim, keepdim)
        if identity in identities:
            raise ValueError("duplicate reduction signature in de-duplicated trace")
        identities.add(identity)
        configuration = {
            "id": f"{id_prefix}_{operation}_{signature(identity)}",
            "source_operation": f"{source_name} traced {operation_name} from pinned B8 replay",
            "operation": operation,
            "m_per_request": source["shape"][0] // 8,
            "input_per_unit": source["shape"][1:],
            "dim": dim,
            "layout": "contiguous",
            "source_trace_path": repository_path(trace_path),
            "source_input_stride_B8": source["stride"],
            "trace_call_count": row.get("count"),
        }
        if operation == "mean":
            configuration["keepdim"] = keepdim
        configurations.append(configuration)
    operations = {configuration["operation"] for configuration in configurations}
    if operations != {"mean", "softmax"}:
        raise ValueError(f"SpatialVLA trace lacks mean/softmax coverage: {sorted(operations)}")
    return configurations


def matrix_configurations(trace: dict) -> list[dict]:
    first_projection = None
    bmm_candidates = []
    for row in trace.get("operations", []):
        arguments = row.get("arguments") or []
        if row.get("aten_operation") == "aten::addmm" and len(arguments) >= 3:
            left = tensor(arguments[1], "addmm left")
            right = tensor(arguments[2], "addmm right")
            if (
                left["shape"] == [8 * 256, 1152]
                and right["shape"] == [1152, 1152]
                and left["stride"] == contiguous_stride(left["shape"])
                and right["stride"] == [1, 1152]
            ):
                first_projection = {
                    "id": "vision_layer0_q_projection_addmm",
                    "source_operation": (
                        "SpatialVLA first divergent leaf: SigLIP vision layer 0 q_proj"
                    ),
                    "operation": "addmm",
                    "m_per_request": 256,
                    "k": 1152,
                    "n": 1152,
                    "layout": "right_transposed",
                }
        if row.get("aten_operation") == "aten::bmm" and len(arguments) >= 2:
            left = tensor(arguments[0], "bmm left")
            right = tensor(arguments[1], "bmm right")
            if (
                len(left["shape"]) == len(right["shape"]) == 3
                and left["shape"][0] >= 8
                and left["shape"][0] % 8 == 0
                and left["shape"][0] == right["shape"][0]
                and left["shape"][2] == right["shape"][1]
                and left["stride"] == contiguous_stride(left["shape"])
            ):
                heads, m, k = left["shape"]
                n = right["shape"][2]
                if right["stride"] == [k * n, n, 1]:
                    layout = "contiguous"
                elif right["stride"] == [k * n, 1, k]:
                    layout = "right_transposed"
                else:
                    continue
                bmm_candidates.append((m * k * n, heads, m, k, n, layout))
    if first_projection is None:
        raise ValueError("SpatialVLA B8 trace lacks the first-divergent q_proj addmm")
    configurations = [first_projection]
    if bmm_candidates:
        _, heads, m, k, n, layout = max(bmm_candidates)
        configurations.append({
            "id": "largest_traced_attention_bmm",
            "source_operation": "Largest directly traced representable SpatialVLA aten::bmm",
            "operation": "bmm",
            "m_per_request": heads // 8,
            "m": m,
            "k": k,
            "n": n,
            "layout": layout,
        })
    return configurations


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
        raise ValueError("B1/B8 SpatialVLA traces use different replay manifests")

    convolutions = convolution_configurations(trace_b8, args.trace_b8)
    reductions = reduction_configurations(trace_b8, args.trace_b8)
    matrices = matrix_configurations(trace_b8)
    attention = []
    if any(
        row.get("aten_operation") == "aten::scaled_dot_product_attention"
        for row in trace_b8.get("operations", [])
    ):
        attention = sdpa_configurations(
            trace_b8,
            args.trace_b8,
            id_prefix="spatialvla",
            source_name="SpatialVLA",
        )
    shards = sorted(args.checkpoint.glob("model-*.safetensors"))
    if len(shards) != 2:
        raise ValueError("expected SpatialVLA's exact two checkpoint shards")
    report = {
        "schema_version": 1,
        "model_id": "spatialvla",
        "checkpoint_weight_shards": [str(path.resolve()) for path in shards],
        "checkpoint_weight_shards_sha256": [sha256(path) for path in shards],
        "adapter_execution_configuration": EXPECTED_EXECUTION_CONFIGURATION,
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            "results/cross_model/spatialvla/first_divergence.json",
            "research/spatialvla/first_divergence.json",
            "research/spatialvla/investigation.md",
            repository_path(args.trace_b1),
            repository_path(args.trace_b8),
            str((args.checkpoint / "config.json").resolve()),
            "/workspace/batch-invariant-pizero-exp/PLAN.md",
        ],
        "configurations": [*matrices, *convolutions, *reductions, *attention],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "configurations": len(report["configurations"]),
        "convolution_configurations": len(convolutions),
        "reduction_configurations": len(reductions),
        "sdpa_configurations": len(attention),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()

"""Build RDT-1B's pinned real-shape operator qualification manifest.

RDT's official ManiSkill policy receives SigLIP tokens that were encoded before
request batching.  Consequently the policy trace must contain no convolution.
Every distinct fused-attention layout is retained, together with one exact
traced representative of each exercised matrix/reduction class and QK/PV
decompositions of the largest attention surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_pi0fast_operator_config import repository_path, sdpa_configurations, sha256, tensor


EXPECTED_ADAPTER = "scripts.model_invariance.adapters.rdt_maniskill"
EXPECTED_EXECUTION_CONFIGURATION = {
    "native_language_padding": "longest_in_batch_right",
    "invariant_language_padding": "canonical_right",
    "canonical_language_tokens": 30,
    "image_history": 2,
    "num_cameras": 3,
    "diffusion_steps": 5,
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
        raise ValueError(f"malformed or foreign RDT B{batch_size} trace: {path}")
    return report


def contiguous_stride(shape: list[int]) -> list[int]:
    result = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        result[index] = result[index + 1] * shape[index + 1]
    return result


def short_id(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()[:12]


def matrix_configurations(
    trace: dict, *, id_prefix: str = "rdt", source_name: str = "RDT",
) -> list[dict]:
    """Select one exact, request-factorable shape per exercised matrix class."""
    candidates: dict[str, list[tuple[int, dict]]] = {name: [] for name in ("mm", "addmm", "bmm")}
    for row in trace.get("operations", []):
        aten = row.get("aten_operation")
        operation = {"aten::mm": "mm", "aten::addmm": "addmm", "aten::bmm": "bmm"}.get(aten)
        if operation is None:
            continue
        arguments = row.get("arguments") or []
        offset = 1 if operation == "addmm" else 0
        if len(arguments) < offset + 2:
            raise ValueError(f"traced {aten} lacks matrix operands")
        left = tensor(arguments[offset], f"{aten} left")
        right = tensor(arguments[offset + 1], f"{aten} right")
        if operation == "bmm":
            if (len(left["shape"]) != 3 or len(right["shape"]) != 3
                    or left["shape"][0] % 8 or left["shape"][0] < 8
                    or left["shape"][0] != right["shape"][0]
                    or left["shape"][2] != right["shape"][1]
                    or left["stride"] != contiguous_stride(left["shape"])):
                continue
            layout = "right_transposed" if right["stride"][-2:] == [1, right["shape"][1]] else "contiguous"
            if layout == "contiguous" and right["stride"] != contiguous_stride(right["shape"]):
                continue
            config = {
                "id": f"{id_prefix}_traced_bmm_{short_id((left['shape'], left['stride'], right['shape'], right['stride']))}",
                "source_operation": f"{source_name} traced aten::bmm representative from pinned B8 replay",
                "operation": "bmm",
                "m_per_request": left["shape"][0] // 8,
                "m": left["shape"][1], "k": left["shape"][2], "n": right["shape"][2],
                "layout": layout,
            }
            footprint = left["shape"][0] * left["shape"][1] * left["shape"][2]
        else:
            if (len(left["shape"]) != 2 or len(right["shape"]) != 2
                    or left["shape"][0] % 8 or left["shape"][0] < 8
                    or left["shape"][1] != right["shape"][0]
                    or left["stride"] != contiguous_stride(left["shape"])):
                continue
            layout = "right_transposed" if right["stride"] == [1, right["shape"][0]] else "contiguous"
            if layout == "contiguous" and right["stride"] != contiguous_stride(right["shape"]):
                continue
            config = {
                "id": f"{id_prefix}_traced_{operation}_{short_id((left['shape'], left['stride'], right['shape'], right['stride']))}",
                "source_operation": f"{source_name} traced aten::{operation} representative from pinned B8 replay",
                "operation": operation,
                "m_per_request": left["shape"][0] // 8,
                "k": left["shape"][1], "n": right["shape"][1],
                "layout": layout,
            }
            footprint = left["shape"][0] * left["shape"][1]
        candidates[operation].append((footprint, config))
    selected = []
    for operation, rows in candidates.items():
        if rows:
            selected.append(min(rows, key=lambda item: item[0])[1])
    if not selected:
        raise ValueError("RDT B8 trace contains no request-factorable matrix operation")
    return selected


def reduction_configurations(
    trace: dict, trace_path: Path, *, id_prefix: str = "rdt", source_name: str = "RDT",
) -> list[dict]:
    result, seen = [], set()
    for row in trace.get("operations", []):
        aten = row.get("aten_operation")
        if aten not in ("aten::mean", "aten::_softmax"):
            continue
        arguments = row.get("arguments") or []
        if len(arguments) < 2:
            raise ValueError(f"traced {aten} lacks input/dimension")
        source = tensor(arguments[0], f"{aten} input")
        if (not source["shape"] or source["shape"][0] < 8 or source["shape"][0] % 8
                or source["stride"] != contiguous_stride(source["shape"])):
            continue
        if aten == "aten::mean":
            raw_dim = arguments[1]
            if not isinstance(raw_dim, list) or len(raw_dim) != 1:
                continue
            operation, dim = "mean", raw_dim[0]
            keepdim = arguments[2] if len(arguments) > 2 else False
            if not isinstance(keepdim, bool):
                continue
        else:
            if not isinstance(arguments[1], int):
                continue
            operation, dim, keepdim = "softmax", arguments[1], None
        if dim in (0, -len(source["shape"])):
            raise ValueError(f"RDT {operation} unexpectedly reduces the request axis")
        identity = (operation, tuple(source["shape"]), dim, keepdim)
        if identity in seen:
            continue
        seen.add(identity)
        config = {
            "id": f"{id_prefix}_{operation}_{short_id(identity)}",
            "source_operation": f"{source_name} traced {aten} from pinned B8 replay",
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
            config["keepdim"] = keepdim
        result.append(config)
    return result


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
        raise ValueError("RDT B1/B8 traces use different frozen replay manifests")
    if any(row.get("aten_operation") == "aten::convolution" for row in trace_b8["operations"]):
        raise ValueError("RDT policy trace unexpectedly contains convolution")

    attention = sdpa_configurations(
        trace_b8, args.trace_b8, id_prefix="rdt", source_name="RDT-1B"
    )
    configurations = matrix_configurations(trace_b8)
    configurations.extend(reduction_configurations(trace_b8, args.trace_b8))
    configurations.extend(attention)
    largest = max(
        attention,
        key=lambda item: item["m_per_request"] * item["query_heads"]
        * item["query_length"] * item["key_length"],
    )
    configurations.extend([
        {
            "id": "rdt_largest_attention_qk_transposed",
            "source_operation": f"QK^T decomposition of {largest['id']}",
            "operation": "bmm",
            "m_per_request": largest["m_per_request"] * largest["query_heads"],
            "m": largest["query_length"], "k": largest["head_dim"],
            "n": largest["key_length"], "layout": "right_transposed",
            "source_qkv_stride": largest["source_q_stride"],
        },
        {
            "id": "rdt_largest_attention_pv_contiguous",
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
            "/workspace/models/rdt/models/rdt_runner.py",
            "/workspace/models/rdt/models/rdt/model.py",
            "/workspace/models/rdt/configs/base.yaml",
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

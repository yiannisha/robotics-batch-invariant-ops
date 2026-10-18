"""Build PI0-FAST's pinned real-shape operator qualification manifest.

The two ATen traces must come from the exact frozen replay adapter at B=1 and
B=8.  Every distinct B=8 scaled-dot-product-attention shape/stride is carried
into qualification, alongside the checkpoint-backed vision patch projection
and the two matrix products already localized by the first-divergence study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/workspace/assets/pi_fast/checkpoint/model.safetensors")
EXPECTED_ADAPTER = "scripts.model_invariance.adapters.lerobot_pi0fast"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def tensor(argument: object, label: str) -> dict:
    if not isinstance(argument, dict):
        raise ValueError(f"{label} is not a traced tensor")
    shape = argument.get("shape")
    stride = argument.get("stride")
    if (not isinstance(shape, list) or not all(isinstance(value, int) for value in shape)
            or not isinstance(stride, list) or not all(isinstance(value, int) for value in stride)):
        raise ValueError(f"{label} lacks integer shape/stride evidence")
    return argument


def validate_trace(path: Path, batch_size: int) -> dict:
    report = json.loads(path.read_text())
    if (report.get("schema_version") != 1
            or report.get("adapter") != EXPECTED_ADAPTER
            or report.get("batch_size") != batch_size
            or len(report.get("request_ids") or []) != batch_size
            or not report.get("replay_manifest_sha256")
            or not report.get("gpu") or not report.get("torch")):
        raise ValueError(f"malformed or foreign PI0-FAST B{batch_size} trace: {path}")
    return report


def sdpa_configurations(
    trace: dict,
    trace_path: Path,
    *,
    id_prefix: str = "pi_fast",
    source_name: str = "PI0-FAST",
) -> list[dict]:
    configurations = []
    seen = set()
    for row in trace.get("operations", []):
        if row.get("aten_operation") != "aten::scaled_dot_product_attention":
            continue
        arguments = row.get("arguments") or []
        if len(arguments) < 3:
            raise ValueError("traced SDPA call lacks Q/K/V")
        query, key, value = (
            tensor(arguments[index], f"SDPA argument {index}") for index in range(3)
        )
        q_shape, k_shape, v_shape = query["shape"], key["shape"], value["shape"]
        q_stride, k_stride, v_stride = query["stride"], key["stride"], value["stride"]
        if (len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4
                or q_shape[0] % 8 or q_shape[0] < 8
                or q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]
                or k_shape[1] != v_shape[1] or q_shape[1] % k_shape[1]
                or q_shape[3] != k_shape[3]
                or k_shape[2] != v_shape[2]):
            raise ValueError(
                "PI0-FAST trace contains an SDPA layout that the independent "
                "qualification schema cannot represent exactly"
            )
        identity = (
            tuple(q_shape), tuple(k_shape), tuple(v_shape),
            tuple(q_stride), tuple(k_stride), tuple(v_stride),
        )
        if identity in seen:
            raise ValueError("duplicate SDPA shape/stride in de-duplicated ATen trace")
        seen.add(identity)
        configurations.append({
            "id": f"{id_prefix}_sdpa_trace_{len(configurations):02d}",
            "source_operation": (
                f"{source_name} traced aten::scaled_dot_product_attention occurrence "
                f"{len(configurations)}"
            ),
            "operation": "sdpa",
            "m_per_request": q_shape[0] // 8,
            "query_heads": q_shape[1],
            "key_heads": k_shape[1],
            "value_heads": v_shape[1],
            "enable_gqa": q_shape[1] != k_shape[1],
            "query_length": q_shape[2],
            "key_length": k_shape[2],
            "head_dim": q_shape[3],
            "value_dim": v_shape[3],
            "scale": 1.0 / math.sqrt(q_shape[3]),
            "layout": "qkv_source_strided",
            "source_trace_path": repository_path(trace_path),
            "source_q_stride": q_stride,
            "source_k_stride": k_stride,
            "source_v_stride": v_stride,
        })
    if not configurations:
        raise ValueError("B8 trace contains no scaled-dot-product-attention calls")
    return configurations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-b1", type=Path, required=True)
    parser.add_argument("--trace-b8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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
            "id": "siglip_patch_conv_contiguous",
            "source_operation": "PI0-FAST SigLIP patch projection from the pinned checkpoint",
            "source_weight_key": (
                "model.paligemma_with_expert.paligemma.model.vision_tower."
                "vision_model.embeddings.patch_embedding.weight"
            ),
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
            "source_operation": "PI0-FAST SigLIP patch projection channels-last diagnostic",
            "source_weight_key": (
                "model.paligemma_with_expert.paligemma.model.vision_tower."
                "vision_model.embeddings.patch_embedding.weight"
            ),
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
        {
            "id": "siglip_layer0_fc1_addmm",
            "source_operation": (
                "PI0-FAST first divergent leaf: vision_model.encoder.layers.0.mlp.fc1"
            ),
            "operation": "addmm",
            "m_per_request": 3 * 256,
            "k": 1152,
            "n": 4304,
            "layout": "right_transposed",
        },
        {
            "id": "language_layer0_down_mm",
            "source_operation": (
                "PI0-FAST profiled language_model.layers.0.mlp.down_proj prefill"
            ),
            "operation": "mm",
            "m_per_request": 969,
            "k": 16384,
            "n": 2048,
            "layout": "right_transposed",
        },
    ]
    attention = sdpa_configurations(trace_b8, args.trace_b8)
    configurations.extend(attention)

    # Qualify the exact QK^T and PV matrix layouts for the largest traced
    # attention score surface, in addition to the fused SDPA call itself.
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
    ])

    report = {
        "schema_version": 1,
        "checkpoint_weights": str(args.checkpoint.resolve()),
        "checkpoint_weights_sha256": sha256(args.checkpoint),
        "replay_manifest_sha256": trace_b8["replay_manifest_sha256"],
        "source_evidence": [
            "results/cross_model/pi_fast/first_divergence.json",
            "results/cross_model/pi_fast/first_divergence_profile.json",
            repository_path(args.trace_b1),
            repository_path(args.trace_b8),
            "/workspace/assets/pi_fast/checkpoint/config.json",
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

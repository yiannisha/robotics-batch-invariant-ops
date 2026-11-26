"""Inspect pinned model traces before selecting causal interventions (no inference).

This is an applicability inventory, not an ablation acceptance proof. In
particular, absence in an instrumented trace alone cannot prove that an
operation is absent from the entire model source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(trace_path: Path, manifest_path: Path, adapter: str) -> dict:
    trace = json.loads(trace_path.read_text())
    if trace.get("adapter") != adapter:
        raise ValueError("foreign adapter trace")
    if trace.get("replay_manifest_sha256") != digest(manifest_path):
        raise ValueError("foreign manifest trace")
    if trace.get("implementation") != "native" or trace.get("batch_size") != 8:
        raise ValueError("requires native B8 trace")
    rows = trace.get("operations", [])
    if not rows or any(not row.get("aten_operation") for row in rows):
        raise ValueError("missing operation inventory")
    operations = sorted({row["aten_operation"] for row in rows})
    convolution = any(
        op.split(".")[0] in {
            "aten::conv1d", "aten::conv2d", "aten::conv3d",
            "aten::convolution", "aten::_convolution",
        } for op in operations
    )
    sdpa = any("scaled_dot_product" in op for op in operations)
    return {
        "adapter": adapter,
        "trace": str(trace_path.resolve()), "trace_sha256": digest(trace_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest(manifest_path),
        "execution_configuration": trace.get("adapter_execution_configuration"),
        "observed_operations": operations,
        "convolution_intervention": (
            "required_observed" if convolution else "source_review_required_before_inapplicability"
        ),
        "explicit_sdpa_intervention": (
            "required_observed" if sdpa else "source_review_required_before_inapplicability"
        ),
        "ablation_measurements_complete": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = inspect(args.trace, args.manifest, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()

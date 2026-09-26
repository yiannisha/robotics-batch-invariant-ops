"""Benchmark SpatialVLA's actual ZoeDepth ConvTranspose2D checkpoint shape."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch
import triton
from safetensors import safe_open

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import conv_transpose2d_batch_invariant
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(os.environ.get("SPATIALVLA_CHECKPOINT", ""))
    shard = checkpoint / "model-00002-of-00002.safetensors"
    if not shard.is_file():
        raise FileNotFoundError(
            "set SPATIALVLA_CHECKPOINT to the official snapshot containing "
            "model-00002-of-00002.safetensors"
        )
    prefix = "vision_zoe_model.neck.reassemble_stage.layers.0.resize"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{prefix}.weight").to("cuda")
        bias = handle.get_tensor(f"{prefix}.bias").to("cuda")

    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    generator = torch.Generator(device="cuda").manual_seed(94_000)
    all_features = torch.randn(
        max(batch_sizes),
        256,
        24,
        24,
        device="cuda",
        dtype=weight.dtype,
        generator=generator,
    )
    stock_reference = torch.nn.functional.conv_transpose2d(all_features[:1], weight, bias, stride=4)
    invariant_reference = conv_transpose2d_batch_invariant(
        all_features[:1],
        weight,
        bias,
        stride=(4, 4),
        padding=(0, 0),
        dilation=(1, 1),
        output_padding=(0, 0),
        groups=1,
    )
    torch.testing.assert_close(invariant_reference, stock_reference, rtol=0.02, atol=0.02)

    cases = []
    for batch_size in batch_sizes:
        features = all_features[:batch_size]
        case = run_case(
            "spatialvla_zoedepth_conv_transpose2d",
            {
                "input": list(features.shape),
                "weight": list(weight.shape),
                "bias": list(bias.shape),
                "stride": [4, 4],
                "padding": [0, 0],
                "output_padding": [0, 0],
                "groups": 1,
                "dtype": str(weight.dtype),
                "checkpoint_weight": f"{prefix}.weight",
            },
            lambda: torch.nn.functional.conv_transpose2d(features, weight, bias, stride=4),
            lambda: conv_transpose2d_batch_invariant(
                features,
                weight,
                bias,
                stride=(4, 4),
                padding=(0, 0),
                dilation=(1, 1),
                output_padding=(0, 0),
                groups=1,
            ),
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        stock_candidate = torch.nn.functional.conv_transpose2d(features, weight, bias, stride=4)
        invariant_candidate = conv_transpose2d_batch_invariant(
            features,
            weight,
            bias,
            stride=(4, 4),
            padding=(0, 0),
            dilation=(1, 1),
            output_padding=(0, 0),
            groups=1,
        )
        case["batch_invariance"] = {
            "stock_exact": torch.equal(stock_reference, stock_candidate[:1]),
            "stock_differing_elements": int((stock_reference != stock_candidate[:1]).sum()),
            "batch_invariant_exact": torch.equal(invariant_reference, invariant_candidate[:1]),
            "batch_invariant_differing_elements": int(
                (invariant_reference != invariant_candidate[:1]).sum()
            ),
        }
        cases.append(case)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "environment": {
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "triton": triton.__version__,
        },
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cases": cases,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

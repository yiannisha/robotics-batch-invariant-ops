"""Benchmark UVA's learned temporal upsampler at its checkpoint shape."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import triton

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import conv_transpose3d_batch_invariant
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.float32
    generator = torch.Generator(device="cuda").manual_seed(92_500)
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    weight = torch.randn(
        1_024, 1_024, 4, 1, 1, device="cuda", dtype=dtype, generator=generator
    )
    bias = torch.randn(1_024, device="cuda", dtype=dtype, generator=generator)
    all_features = torch.randn(
        max(batch_sizes),
        1_024,
        4,
        16,
        16,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    stock_reference = torch.nn.functional.conv_transpose3d(
        all_features[:1], weight, bias, stride=(4, 1, 1)
    )
    invariant_reference = conv_transpose3d_batch_invariant(
        all_features[:1],
        weight,
        bias,
        stride=(4, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        output_padding=(0, 0, 0),
        groups=1,
    )
    cases = []
    for batch_size in batch_sizes:
        features = all_features[:batch_size]
        case = run_case(
            "uva_action_conv_transpose3d",
            {
                "input": list(features.shape),
                "weight": list(weight.shape),
                "bias": list(bias.shape),
                "stride": [4, 1, 1],
                "padding": [0, 0, 0],
                "output_padding": [0, 0, 0],
                "groups": 1,
                "dtype": str(dtype),
            },
            lambda: torch.nn.functional.conv_transpose3d(
                features, weight, bias, stride=(4, 1, 1)
            ),
            lambda: conv_transpose3d_batch_invariant(
                features,
                weight,
                bias,
                stride=(4, 1, 1),
                padding=(0, 0, 0),
                dilation=(1, 1, 1),
                output_padding=(0, 0, 0),
                groups=1,
            ),
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        stock_candidate = torch.nn.functional.conv_transpose3d(
            features, weight, bias, stride=(4, 1, 1)
        )
        invariant_candidate = conv_transpose3d_batch_invariant(
            features,
            weight,
            bias,
            stride=(4, 1, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            output_padding=(0, 0, 0),
            groups=1,
        )
        stock_difference = (stock_reference - stock_candidate[:1]).abs()
        invariant_difference = (invariant_reference - invariant_candidate[:1]).abs()
        case["batch_invariance"] = {
            "stock_exact": torch.equal(stock_reference, stock_candidate[:1]),
            "stock_differing_elements": int((stock_reference != stock_candidate[:1]).sum()),
            "stock_max_abs_difference": float(stock_difference.max()),
            "batch_invariant_exact": torch.equal(
                invariant_reference, invariant_candidate[:1]
            ),
            "batch_invariant_differing_elements": int(
                (invariant_reference != invariant_candidate[:1]).sum()
            ),
            "batch_invariant_max_abs_difference": float(invariant_difference.max()),
        }
        cases.append(case)
        del stock_candidate, invariant_candidate

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

"""Benchmark GR00T N1.7's first divergent BMM and addmm shapes."""

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

from batch_invariant_ops import bmm_persistent, matmul_persistent
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(71_700)
    cases = []
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        values = torch.randn(batch_size, 40, 3_072, device="cuda", dtype=dtype, generator=generator)
        weights = torch.randn(
            batch_size, 3_072, 1_536, device="cuda", dtype=dtype, generator=generator
        )
        cases.append(
            run_case(
                "groot_n17_action_encoder_w2_bmm",
                {
                    "left": list(values.shape),
                    "right": list(weights.shape),
                    "dtype": str(dtype),
                },
                lambda values=values, weights=weights: torch.bmm(values, weights),
                lambda values=values, weights=weights: bmm_persistent(values, weights),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del values, weights

        linear_values = torch.randn(
            batch_size * 156,
            2_048,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        linear_weight = torch.randn(
            2_048,
            1_536,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        linear_bias = torch.randn(1_536, device="cuda", dtype=dtype, generator=generator)
        cases.append(
            run_case(
                "groot_n17_action_dit_cross_attention_k_addmm",
                {
                    "left": list(linear_values.shape),
                    "right": list(linear_weight.shape),
                    "bias": list(linear_bias.shape),
                    "dtype": str(dtype),
                },
                lambda values=linear_values, weight=linear_weight, bias=linear_bias: (
                    torch.addmm(bias, values, weight)
                ),
                lambda values=linear_values, weight=linear_weight, bias=linear_bias: (
                    matmul_persistent(values, weight, bias)
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del linear_values, linear_weight, linear_bias

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

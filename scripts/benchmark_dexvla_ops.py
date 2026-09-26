"""Benchmark DexVLA's divergent vision SDPA and ScaleDP conditioning addmm."""

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

from batch_invariant_ops import matmul_persistent, scaled_dot_product_attention_batch_invariant
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(96_500)
    cases = []
    conditioning_weight = torch.randn(1_550, 1_024, device="cuda", dtype=dtype, generator=generator)
    conditioning_bias = torch.randn(1_024, device="cuda", dtype=dtype, generator=generator)
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        # The official fixture has three equal 1,564-token image segments per
        # robot observation. The invariant bridge evaluates those independent
        # segments as the leading attention batch.
        query = torch.randn(
            batch_size * 3,
            16,
            1_564,
            80,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cases.append(
            run_case(
                "dexvla_vision_block0_segmented_sdpa",
                {
                    "query": list(query.shape),
                    "key": list(key.shape),
                    "value": list(value.shape),
                    "logical_robot_batch": batch_size,
                    "images_per_observation": 3,
                    "dtype": str(dtype),
                    "causal": False,
                },
                lambda: torch.nn.functional.scaled_dot_product_attention(query, key, value),
                lambda: scaled_dot_product_attention_batch_invariant(query, key, value),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del query, key, value

        conditioning = torch.randn(
            batch_size,
            1_550,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "dexvla_scaledp_conditioning_addmm",
                {
                    "left": list(conditioning.shape),
                    "right": list(conditioning_weight.shape),
                    "bias": list(conditioning_bias.shape),
                    "dtype": str(dtype),
                },
                lambda: torch.addmm(conditioning_bias, conditioning, conditioning_weight),
                lambda: matmul_persistent(
                    conditioning, conditioning_weight, bias=conditioning_bias
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del conditioning

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

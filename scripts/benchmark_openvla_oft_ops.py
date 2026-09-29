"""Benchmark OpenVLA-OFT's divergent DINO SDPA and action-head GEMM."""

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
    generator = torch.Generator(device="cuda").manual_seed(95_500)
    cases = []
    action_weight = torch.randn(28_672, 4_096, device="cuda", dtype=dtype, generator=generator)
    action_bias = torch.randn(4_096, device="cuda", dtype=dtype, generator=generator)
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        query = torch.randn(
            batch_size,
            16,
            261,
            64,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cases.append(
            run_case(
                "openvla_oft_dino_block0_sdpa",
                {
                    "query": list(query.shape),
                    "key": list(key.shape),
                    "value": list(value.shape),
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

        activations = torch.randn(
            batch_size * 8,
            28_672,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "openvla_oft_action_head_fc1_addmm",
                {
                    "left": list(activations.shape),
                    "right": list(action_weight.shape),
                    "bias": list(action_bias.shape),
                    "logical_input": [batch_size, 8, 28_672],
                    "dtype": str(dtype),
                },
                lambda: torch.addmm(action_bias, activations, action_weight),
                lambda: matmul_persistent(activations, action_weight, bias=action_bias),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del activations

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

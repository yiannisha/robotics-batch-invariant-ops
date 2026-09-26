"""Benchmark the first divergent MolmoAct2 image-projector MM shape."""

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

from batch_invariant_ops import matmul_persistent
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(28_000)
    weight = torch.randn(9_728, 2_560, device="cuda", dtype=dtype, generator=generator)
    cases = []
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        values = torch.randn(
            batch_size * 392,
            9_728,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "molmoact2_image_projector_w2_mm",
                {
                    "left": list(values.shape),
                    "right": list(weight.shape),
                    "dtype": str(dtype),
                },
                lambda: torch.mm(values, weight),
                lambda: matmul_persistent(values, weight),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del values

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

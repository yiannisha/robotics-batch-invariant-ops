"""Benchmark the actual BAAI UniVLA autoregressive LM-head matrix shape."""

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

from batch_invariant_ops import matmul_persistent
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(os.environ.get("UNIVLA_CHECKPOINT", ""))
    shard = checkpoint / "model-00004-of-00004.safetensors"
    if not shard.is_file():
        raise FileNotFoundError(
            "set UNIVLA_CHECKPOINT to the released checkpoint containing "
            "model-00004-of-00004.safetensors"
        )
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("lm_head.weight").to("cuda")
    weight_t = weight.t()

    cases = []
    for batch_size in (int(item) for item in args.batch_sizes.split(",")):
        generator = torch.Generator(device="cuda").manual_seed(91_000 + batch_size)
        hidden = torch.randn(
            batch_size,
            weight.shape[1],
            device="cuda",
            dtype=weight.dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "univla_autoregressive_lm_head_mm",
                {
                    "left": list(hidden.shape),
                    "right": list(weight_t.shape),
                    "dtype": str(weight.dtype),
                    "checkpoint_weight": "lm_head.weight",
                },
                lambda: torch.mm(hidden, weight_t),
                lambda: matmul_persistent(hidden, weight_t),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )

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

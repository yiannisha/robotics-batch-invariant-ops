"""Benchmark stock CUDA operators against their batch-invariant replacements."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from pathlib import Path

import torch
import triton

from batch_invariant_ops import bmm_persistent, conv2d_batch_invariant


def benchmark(
    operation: Callable[[], torch.Tensor], *, warmup: int, repetitions: int
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        output = operation()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / repetitions
    peak_bytes = torch.cuda.max_memory_allocated() - baseline_memory
    del output
    return {"latency_ms": elapsed_ms, "peak_incremental_bytes": peak_bytes}


def run_case(
    name: str,
    shape: dict[str, object],
    stock: Callable[[], torch.Tensor],
    invariant: Callable[[], torch.Tensor],
    *,
    batch_size: int,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    stock_result = benchmark(stock, warmup=warmup, repetitions=repetitions)
    invariant_result = benchmark(invariant, warmup=warmup, repetitions=repetitions)
    return {
        "operator": name,
        "batch_size": batch_size,
        "shape": shape,
        "stock": stock_result,
        "batch_invariant": invariant_result,
        "latency_ratio": invariant_result["latency_ms"] / stock_result["latency_ms"],
        "stock_samples_per_second": batch_size * 1000 / stock_result["latency_ms"],
        "batch_invariant_samples_per_second": batch_size * 1000 / invariant_result["latency_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    dtype = torch.bfloat16
    cases = []
    for batch_size in batch_sizes:
        generator = torch.Generator(device="cuda").manual_seed(8000 + batch_size)
        images = torch.randn(
            batch_size, 3, 224, 224, device="cuda", dtype=dtype, generator=generator
        )
        conv_weight = torch.randn(1152, 3, 14, 14, device="cuda", dtype=dtype, generator=generator)
        conv_bias = torch.randn(1152, device="cuda", dtype=dtype, generator=generator)
        cases.append(
            run_case(
                "conv2d",
                {
                    "input": list(images.shape),
                    "weight": list(conv_weight.shape),
                    "stride": [14, 14],
                    "dtype": str(dtype),
                },
                lambda: torch.nn.functional.conv2d(images, conv_weight, conv_bias, stride=(14, 14)),
                lambda: conv2d_batch_invariant(
                    images,
                    conv_weight,
                    conv_bias,
                    (14, 14),
                    (0, 0),
                    (1, 1),
                    False,
                    (0, 0),
                    1,
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del images, conv_weight, conv_bias

        heads, queries, tokens, head_dim = 8, 4, 281, 256
        left = torch.randn(
            batch_size * heads,
            queries,
            tokens,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        right = torch.randn(
            batch_size * heads,
            tokens,
            head_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "bmm",
                {"left": list(left.shape), "right": list(right.shape), "dtype": str(dtype)},
                lambda: torch.bmm(left, right),
                lambda: bmm_persistent(left, right),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del left, right

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
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
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()

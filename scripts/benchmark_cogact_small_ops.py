"""Benchmark CogACT-Small's first SDPA and Llama down-projection shapes."""

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

    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    maximum_batch = max(batch_sizes)
    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(74_100)
    cases = []

    query_all = torch.randn(
        maximum_batch,
        16,
        261,
        64,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key_all = torch.randn(query_all.shape, device="cuda", dtype=dtype, generator=generator)
    value_all = torch.randn(query_all.shape, device="cuda", dtype=dtype, generator=generator)
    stock_attention_reference = torch.nn.functional.scaled_dot_product_attention(
        query_all[:1], key_all[:1], value_all[:1]
    )
    invariant_attention_reference = scaled_dot_product_attention_batch_invariant(
        query_all[:1], key_all[:1], value_all[:1]
    )
    for batch_size in batch_sizes:
        query, key, value = (
            query_all[:batch_size],
            key_all[:batch_size],
            value_all[:batch_size],
        )
        stock = lambda q=query, k=key, v=value: torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )
        invariant = lambda q=query, k=key, v=value: (
            scaled_dot_product_attention_batch_invariant(q, k, v)
        )
        case = run_case(
            "cogact_small_dino_block0_sdpa",
            {
                "query": list(query.shape),
                "key": list(key.shape),
                "value": list(value.shape),
                "dtype": str(dtype),
                "causal": False,
            },
            stock,
            invariant,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        stock_candidate = stock()[:1]
        invariant_candidate = invariant()[:1]
        case["batch_invariance"] = {
            "stock_exact": torch.equal(stock_attention_reference, stock_candidate),
            "stock_differing_elements": int((stock_attention_reference != stock_candidate).sum()),
            "batch_invariant_exact": torch.equal(
                invariant_attention_reference, invariant_candidate
            ),
            "batch_invariant_differing_elements": int(
                (invariant_attention_reference != invariant_candidate).sum()
            ),
        }
        cases.append(case)

    activations_all = torch.randn(
        maximum_batch,
        278,
        11_008,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(11_008, 4_096, device="cuda", dtype=dtype, generator=generator)
    stock_mm_reference = torch.mm(activations_all[:1].reshape(-1, 11_008), weight)
    invariant_mm_reference = matmul_persistent(activations_all[:1].reshape(-1, 11_008), weight)
    for batch_size in batch_sizes:
        activations = activations_all[:batch_size].reshape(-1, 11_008)
        stock = lambda x=activations: torch.mm(x, weight)
        invariant = lambda x=activations: matmul_persistent(x, weight)
        case = run_case(
            "cogact_small_llama_layer0_mlp_down_mm",
            {
                "left": list(activations.shape),
                "right": list(weight.shape),
                "dtype": str(dtype),
            },
            stock,
            invariant,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        stock_candidate = stock()[:278]
        invariant_candidate = invariant()[:278]
        case["batch_invariance"] = {
            "stock_exact": torch.equal(stock_mm_reference, stock_candidate),
            "stock_differing_elements": int((stock_mm_reference != stock_candidate).sum()),
            "batch_invariant_exact": torch.equal(invariant_mm_reference, invariant_candidate),
            "batch_invariant_differing_elements": int(
                (invariant_mm_reference != invariant_candidate).sum()
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

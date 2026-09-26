"""Benchmark the two operator shapes required by official RDT-1B inference."""

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


def _invariance(stock_reference, invariant_reference, stock, invariant):
    stock_candidate = stock()[:1]
    invariant_candidate = invariant()[:1]
    return {
        "stock_exact": torch.equal(stock_reference, stock_candidate),
        "stock_differing_elements": int((stock_reference != stock_candidate).sum()),
        "batch_invariant_exact": torch.equal(invariant_reference, invariant_candidate),
        "batch_invariant_differing_elements": int(
            (invariant_reference != invariant_candidate).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-batch-sizes", default="1,2,8")
    parser.add_argument("--addmm-batch-sizes", default="1,2,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(73_100)
    cases = []

    attention_batches = tuple(int(item) for item in args.attention_batch_sizes.split(","))
    maximum_attention_batch = max(attention_batches)
    query_all = torch.randn(
        maximum_attention_batch,
        32,
        67,
        64,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key_all = torch.randn(
        maximum_attention_batch,
        32,
        4_374,
        64,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    value_all = torch.randn(key_all.shape, device="cuda", dtype=dtype, generator=generator)
    stock_attention_reference = torch.nn.functional.scaled_dot_product_attention(
        query_all[:1], key_all[:1], value_all[:1]
    )
    invariant_attention_reference = scaled_dot_product_attention_batch_invariant(
        query_all[:1], key_all[:1], value_all[:1]
    )
    for batch_size in attention_batches:
        query, key, value = (query_all[:batch_size], key_all[:batch_size], value_all[:batch_size])
        stock = lambda q=query, k=key, v=value: torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )
        invariant = lambda q=query, k=key, v=value: (
            scaled_dot_product_attention_batch_invariant(q, k, v)
        )
        case = run_case(
            "rdt_1b_block1_image_cross_attention_sdpa",
            {
                "query": list(query.shape),
                "key": list(key.shape),
                "value": list(value.shape),
                "dtype": str(dtype),
                "causal": False,
                "checkpoint_boundary": "model.blocks.1.cross_attn",
            },
            stock,
            invariant,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        case["batch_invariance"] = _invariance(
            stock_attention_reference, invariant_attention_reference, stock, invariant
        )
        cases.append(case)
    del query_all, key_all, value_all

    addmm_batches = tuple(int(item) for item in args.addmm_batch_sizes.split(","))
    maximum_addmm_batch = max(addmm_batches)
    language_all = torch.randn(
        maximum_addmm_batch,
        20,
        4_096,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(4_096, 2_048, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(2_048, device="cuda", dtype=dtype, generator=generator)
    stock_addmm_reference = torch.addmm(bias, language_all[:1].reshape(-1, 4_096), weight)
    invariant_addmm_reference = matmul_persistent(language_all[:1].reshape(-1, 4_096), weight, bias)
    for batch_size in addmm_batches:
        language = language_all[:batch_size].reshape(-1, 4_096)
        stock = lambda x=language: torch.addmm(bias, x, weight)
        invariant = lambda x=language: matmul_persistent(x, weight, bias)
        case = run_case(
            "rdt_1b_language_adaptor_addmm",
            {
                "left": list(language.shape),
                "right": list(weight.shape),
                "bias": list(bias.shape),
                "dtype": str(dtype),
                "checkpoint_boundary": "lang_adaptor.0",
            },
            stock,
            invariant,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        stock_candidate = stock()[:20]
        invariant_candidate = invariant()[:20]
        case["batch_invariance"] = {
            "stock_exact": torch.equal(stock_addmm_reference, stock_candidate),
            "stock_differing_elements": int((stock_addmm_reference != stock_candidate).sum()),
            "batch_invariant_exact": torch.equal(invariant_addmm_reference, invariant_candidate),
            "batch_invariant_differing_elements": int(
                (invariant_addmm_reference != invariant_candidate).sum()
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

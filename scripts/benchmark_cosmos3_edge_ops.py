"""Benchmark Cosmos 3 Edge attention and decoder norm repairs at model shapes."""

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

from batch_invariant_ops import (
    linalg_vector_norm_batch_invariant,
    varlen_scaled_dot_product_attention_batch_invariant,
)
from scripts.benchmark_batch_invariant_ops import run_case


def _offsets(batch_size: int, length: int) -> torch.Tensor:
    return torch.arange(
        0, (batch_size + 1) * length, length, device="cuda", dtype=torch.int32
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from cosmos_framework.model.attention import attention as cosmos_attention
    from cosmos_framework.model.attention.masks import CausalType

    dtype = torch.bfloat16
    cases = []
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    geometries = (
        ("cosmos3_edge_causal_varlen_attention", 158, 158, True),
        ("cosmos3_edge_full_varlen_attention", 3093, 3251, False),
    )
    for batch_size in batch_sizes:
        generator = torch.Generator(device="cuda").manual_seed(304_000 + batch_size)
        for name, q_length, kv_length, causal in geometries:
            query = torch.randn(
                1,
                batch_size * q_length,
                16,
                128,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            key = torch.randn(
                1,
                batch_size * kv_length,
                8,
                128,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            value = torch.randn_like(key)
            q_offsets = _offsets(batch_size, q_length)
            kv_offsets = _offsets(batch_size, kv_length)
            causal_type = CausalType.DontCare if causal else None
            cases.append(
                run_case(
                    name,
                    {
                        "query": list(query.shape),
                        "key": list(key.shape),
                        "logical_query_length": q_length,
                        "logical_kv_length": kv_length,
                        "query_heads": 16,
                        "kv_heads": 8,
                        "head_dim": 128,
                        "causal": causal,
                        "dtype": str(dtype),
                    },
                    lambda: cosmos_attention(
                        query,
                        key,
                        value,
                        is_causal=causal,
                        causal_type=causal_type,
                        cumulative_seqlen_Q=q_offsets,
                        cumulative_seqlen_KV=kv_offsets,
                        max_seqlen_Q=q_length,
                        max_seqlen_KV=kv_length,
                    ),
                    lambda: varlen_scaled_dot_product_attention_batch_invariant(
                        query,
                        key,
                        value,
                        q_offsets,
                        kv_offsets,
                        is_causal=causal,
                        causal_type=causal_type,
                    ),
                    batch_size=batch_size,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
            )
            del query, key, value, q_offsets, kv_offsets

        decoder_values = torch.randn(
            batch_size,
            1024,
            4,
            132,
            160,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "cosmos3_wan_decoder_vector_norm",
                {"input": list(decoder_values.shape), "dim": 1, "dtype": str(dtype)},
                lambda: torch.linalg.vector_norm(decoder_values, dim=1, keepdim=True),
                lambda: linalg_vector_norm_batch_invariant(
                    decoder_values, dim=1, keepdim=True
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del decoder_values

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

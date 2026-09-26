"""Benchmark the GEMM shapes responsible for LingBot-VLA 2.0 divergence."""

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
    bmm_persistent,
    deterministic_token_choice_moe,
    matmul_persistent,
)
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.float32
    generator = torch.Generator(device="cuda").manual_seed(82_500)
    cases = []
    state_weight = torch.randn(55, 768, device="cuda", dtype=dtype, generator=generator)
    state_bias = torch.randn(768, device="cuda", dtype=dtype, generator=generator)
    expert_weight = torch.randn(768, 4_096, device="cuda", dtype=dtype, generator=generator)
    moe_gate = torch.randn(32, 512, 768, device="cuda", dtype=dtype, generator=generator)
    moe_up = torch.randn_like(moe_gate)
    moe_down = torch.randn(32, 768, 512, device="cuda", dtype=dtype, generator=generator)
    from lingbotvla.ops.robby_moe import robby_moe_forward

    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        state = torch.randn(batch_size, 55, device="cuda", dtype=dtype, generator=generator)
        cases.append(
            run_case(
                "lingbot_vla2_state_projection_addmm",
                {
                    "left": list(state.shape),
                    "right": list(state_weight.shape),
                    "bias": list(state_bias.shape),
                    "dtype": str(dtype),
                },
                lambda: torch.addmm(state_bias, state, state_weight),
                lambda: matmul_persistent(state, state_weight, bias=state_bias),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )

        token_count = batch_size * 51
        moe_hidden = torch.randn(
            token_count,
            768,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        selected = torch.randint(
            0,
            32,
            (token_count, 4),
            device="cuda",
            generator=generator,
        )
        routing = torch.rand(
            token_count,
            4,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        routing = routing / routing.sum(dim=-1, keepdim=True)
        max_routes = token_count * 4
        workspace = {
            "counts": torch.empty(32, device="cuda", dtype=torch.int32),
            "rows": torch.empty(32, max_routes, device="cuda", dtype=torch.int32),
            "slots": torch.empty(32, max_routes, device="cuda", dtype=torch.int32),
            "inter": torch.empty(token_count, 4, 512, device="cuda", dtype=dtype),
            "out": torch.empty(token_count, 768, device="cuda", dtype=dtype),
        }
        cases.append(
            run_case(
                "lingbot_vla2_token_choice_moe",
                {
                    "hidden": list(moe_hidden.shape),
                    "experts": 32,
                    "top_k": 4,
                    "intermediate": 512,
                    "logical_robot_batch": batch_size,
                    "dtype": str(dtype),
                },
                lambda: robby_moe_forward(
                    moe_hidden,
                    routing,
                    selected,
                    moe_gate,
                    moe_up,
                    moe_down,
                    workspace=workspace,
                ),
                lambda: deterministic_token_choice_moe(
                    moe_hidden,
                    routing,
                    selected,
                    moe_gate,
                    moe_up,
                    moe_down,
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )

        expert = torch.randn(
            batch_size * 51,
            768,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "lingbot_vla2_action_expert_mm",
                {
                    "left": list(expert.shape),
                    "right": list(expert_weight.shape),
                    "logical_robot_batch": batch_size,
                    "dtype": str(dtype),
                },
                lambda: torch.mm(expert, expert_weight),
                lambda: matmul_persistent(expert, expert_weight),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )

        query = torch.randn(
            batch_size * 32,
            51,
            128,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn(
            batch_size * 32,
            128,
            321,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        cases.append(
            run_case(
                "lingbot_vla2_action_attention_bmm",
                {
                    "left": list(query.shape),
                    "right": list(key.shape),
                    "logical_robot_batch": batch_size,
                    "dtype": str(dtype),
                },
                lambda: torch.bmm(query, key),
                lambda: bmm_persistent(query, key),
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

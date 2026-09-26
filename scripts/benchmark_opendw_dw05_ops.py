"""Benchmark DW05's actual first-divergence ActionDiT projection."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import linear_batch_invariant
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.opendw_dw05")
    parser.add_argument("--batch-sizes", default="1,2,8,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]
    projection = model.action_expert.action_encoder
    value_b1 = target["action_noise"].to(adapter.device, dtype=torch.bfloat16).unsqueeze(0)
    weight = projection.weight.detach()
    bias = projection.bias.detach()
    stock_reference = F.linear(value_b1, weight, bias)
    invariant_reference = linear_batch_invariant(value_b1, weight, bias)
    cases = []
    for batch_size in batch_sizes:
        value = value_b1.repeat(batch_size, 1, 1)
        stock = lambda value=value: F.linear(value, weight, bias)
        invariant = lambda value=value: linear_batch_invariant(value, weight, bias)
        case = run_case(
            "opendw_action_expert_action_encoder_linear",
            {
                "input": list(value.shape),
                "flattened_input": [batch_size * 32, 14],
                "weight": list(weight.shape),
                "bias": list(bias.shape),
                "dtype": str(value.dtype),
                "checkpoint_boundary": "action_expert.action_encoder",
            },
            stock,
            invariant,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        case["batch_invariance"] = {
            "stock_exact": torch.equal(stock_reference, stock()[:1]),
            "stock_differing_elements": int((stock_reference != stock()[:1]).sum()),
            "batch_invariant_exact": torch.equal(invariant_reference, invariant()[:1]),
            "batch_invariant_differing_elements": int(
                (invariant_reference != invariant()[:1]).sum()
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

"""Benchmark OpenHelix's newly required rank-3 linear operator."""

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

from batch_invariant_ops import linear_batch_invariant, set_batch_invariant_mode
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.openhelix_calvin")
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    adapter = importlib.import_module(args.adapter).adapter
    models = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]
    projector = models.planner.get_model().mm_projector
    captured = []

    def capture_input(_module, arguments):
        captured.append(arguments[0].detach())

    handle = projector.register_forward_pre_hook(capture_input)
    try:
        with set_batch_invariant_mode(), torch.inference_mode():
            adapter._planner_features(
                models,
                [target["planner_image"]] * max(batch_sizes),
                [target["instruction"]] * max(batch_sizes),
            )
    finally:
        handle.remove()
    inputs = captured[0]
    weight = projector.weight.detach()
    bias = projector.bias.detach()
    dtype = inputs.dtype
    stock_reference = F.linear(inputs[:1], weight, bias)
    invariant_reference = linear_batch_invariant(inputs[:1], weight, bias)
    cases = []
    for batch_size in batch_sizes:
        value = inputs[:batch_size]
        stock = lambda value=value: F.linear(value, weight, bias)
        invariant = lambda value=value: linear_batch_invariant(value, weight, bias)
        case = run_case(
            "openhelix_mm_projector_rank3_linear",
            {
                "input": list(value.shape),
                "weight": list(weight.shape),
                "bias": list(bias.shape),
                "dtype": str(dtype),
                "checkpoint_boundary": "model.mm_projector",
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
            "stock_exact": torch.equal(stock_reference, stock_candidate),
            "stock_differing_elements": int((stock_reference != stock_candidate).sum()),
            "batch_invariant_exact": torch.equal(invariant_reference, invariant_candidate),
            "batch_invariant_differing_elements": int(
                (invariant_reference != invariant_candidate).sum()
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

"""Benchmark the actual VITRA FOV projection that first diverges."""

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
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.vitra_vla_3b")
    parser.add_argument("--batch-sizes", default="1,2,8,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]
    projection = model.fov_encoder.projector[2]
    captured = []

    def capture_input(_module, arguments):
        if not captured:
            captured.append(arguments[0].detach())

    handle = projection.register_forward_pre_hook(capture_input)
    try:
        with torch.inference_mode():
            adapter.run_model(model, adapter.compose_batch([target]))
    finally:
        handle.remove()

    value_b1 = captured[0]
    weight = projection.weight.detach()
    bias = projection.bias.detach()
    stock_reference = F.linear(value_b1, weight, bias)
    invariant_reference = linear_batch_invariant(value_b1, weight, bias)
    cases = []
    for batch_size in batch_sizes:
        value = value_b1.repeat(batch_size, 1)
        stock = lambda value=value: F.linear(value, weight, bias)
        invariant = lambda value=value: linear_batch_invariant(value, weight, bias)
        case = run_case(
            "vitra_fov_encoder_projector_2_linear",
            {
                "input": list(value.shape),
                "weight": list(weight.shape),
                "bias": list(bias.shape),
                "dtype": str(value.dtype),
                "checkpoint_boundary": "fov_encoder.projector.2",
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

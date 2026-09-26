"""Profile and benchmark OpenWAM's actual first-divergence video head."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from torch.profiler import ProfilerActivity, profile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import linear_batch_invariant
from scripts.benchmark_batch_invariant_ops import run_case
from scripts.model_invariance.harness import compare_outputs, seed_everything


def _profile_linear(value, weight, bias):
    F.linear(value, weight, bias)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        output = F.linear(value, weight, bias)
        torch.cuda.synchronize()
    kernels = sorted(
        {
            event.name
            for event in result.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )
    return output, kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.openwam_alpha_robotwin",
    )
    parser.add_argument("--batch-sizes", default="1,2,8,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]
    projection = model.video_backbone.dit.head.head
    captured = {}

    def capture_input(_module, arguments):
        if "value" not in captured:
            captured["value"] = arguments[0].detach().contiguous()

    handle = projection.register_forward_pre_hook(capture_input)
    try:
        with torch.inference_mode():
            adapter.run_model(model, adapter.compose_batch([target]))
    finally:
        handle.remove()

    value_b1 = captured["value"]
    weight = projection.weight.detach()
    bias = projection.bias.detach()
    output_b1, kernels_b1 = _profile_linear(value_b1, weight, bias)
    value_b2 = value_b1.repeat(2, 1, 1)
    output_b2, kernels_b2 = _profile_linear(value_b2, weight, bias)
    cases = []
    invariant_reference = linear_batch_invariant(value_b1, weight, bias)
    for batch_size in batch_sizes:
        value = value_b1.repeat(batch_size, 1, 1)
        stock = lambda value=value: F.linear(value, weight, bias)
        invariant = lambda value=value: linear_batch_invariant(value, weight, bias)
        case = run_case(
            "openwam_video_dit_head_linear",
            {
                "input": list(value.shape),
                "flattened_input": [batch_size * 360, 3072],
                "weight": list(weight.shape),
                "bias": list(bias.shape),
                "dtype": str(value.dtype),
                "checkpoint_boundary": "video_backbone.dit.head.head",
            },
            stock,
            invariant,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        stock_output = stock()[:1]
        invariant_output = invariant()[:1]
        case["batch_invariance"] = {
            "stock_exact": torch.equal(output_b1, stock_output),
            "stock_differing_elements": int((output_b1 != stock_output).sum()),
            "batch_invariant_exact": torch.equal(invariant_reference, invariant_output),
            "batch_invariant_differing_elements": int(
                (invariant_reference != invariant_output).sum()
            ),
        }
        cases.append(case)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "model": adapter.name,
        "module": "video_backbone.dit.head.head",
        "operation": "aten::linear",
        "environment": {
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "triton": triton.__version__,
        },
        "profile": {
            "b1_shapes": {
                "input": list(value_b1.shape),
                "flattened_input": [360, 3072],
                "weight": list(weight.shape),
                "bias": list(bias.shape),
            },
            "b2_shapes": {
                "input": list(value_b2.shape),
                "flattened_input": [720, 3072],
                "weight": list(weight.shape),
                "bias": list(bias.shape),
            },
            "input_target_comparison": asdict(
                compare_outputs(value_b1.cpu(), value_b2[:1].cpu())[0]
            ),
            "output_target_comparison": asdict(
                compare_outputs(output_b1.cpu(), output_b2[:1].cpu())[0]
            ),
            "stock_cuda_kernels_b1": kernels_b1,
            "stock_cuda_kernels_b2": kernels_b2,
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

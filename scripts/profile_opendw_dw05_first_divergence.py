"""Profile the exact DW05 ActionDiT projection that first diverges."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        default="scripts.model_invariance.adapters.opendw_dw05",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target, companion = adapter.load_example_inputs(2)
    value_b1 = target["action_noise"].to(adapter.device, dtype=torch.bfloat16).unsqueeze(0)
    value_b2 = torch.stack([target["action_noise"], companion["action_noise"]], dim=0).to(
        adapter.device, dtype=torch.bfloat16
    )
    projection = model.action_expert.action_encoder
    weight = projection.weight.detach()
    bias = projection.bias.detach()
    output_b1, kernels_b1 = _profile_linear(value_b1, weight, bias)
    output_b2, kernels_b2 = _profile_linear(value_b2, weight, bias)

    report = {
        "model": adapter.name,
        "module": "action_expert.action_encoder",
        "operation": "aten::linear",
        "dtype": str(value_b1.dtype),
        "b1_shapes": {
            "input": list(value_b1.shape),
            "flattened_input": [32, 14],
            "weight": list(weight.shape),
            "bias": list(bias.shape),
        },
        "b2_shapes": {
            "input": list(value_b2.shape),
            "flattened_input": [64, 14],
            "weight": list(weight.shape),
            "bias": list(bias.shape),
        },
        "input_target_comparison": asdict(compare_outputs(value_b1.cpu(), value_b2[:1].cpu())[0]),
        "output_target_comparison": asdict(
            compare_outputs(output_b1.cpu(), output_b2[:1].cpu())[0]
        ),
        "stock_cuda_kernels_b1": kernels_b1,
        "stock_cuda_kernels_b2": kernels_b2,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

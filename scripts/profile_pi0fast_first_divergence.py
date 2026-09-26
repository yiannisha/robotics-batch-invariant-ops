"""Profile the exact Pi0-FAST MLP projection that first changes with batch."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_invariance.harness import (
    _compose_batch,
    compare_outputs,
    seed_everything,
)


def _capture_prefill_down_input(adapter, model, examples, batch_size: int, seed: int):
    down_projection = (
        model.paligemma_with_expert.paligemma.model.language_model.layers[0].mlp.down_proj
    )
    captured = []

    def hook(_module, inputs):
        if not captured:
            captured.append(inputs[0].detach().clone())

    handle = down_projection.register_forward_pre_hook(hook)
    try:
        batch = _compose_batch(adapter, examples[:batch_size])
        seed_everything(seed)
        with torch.inference_mode():
            adapter.run_model(model, batch)
    finally:
        handle.remove()
    return captured[0], down_projection.weight.detach().t()


def _profile_mm(left: torch.Tensor, right: torch.Tensor):
    torch.mm(left, right)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        output = torch.mm(left, right)
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
        default="scripts.model_invariance.adapters.lerobot_pi0fast",
    )
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    input_b1, weight_t = _capture_prefill_down_input(
        adapter, model, examples, batch_size=1, seed=args.seed
    )
    input_b2, _ = _capture_prefill_down_input(
        adapter, model, examples, batch_size=2, seed=args.seed
    )

    input_comparison = compare_outputs(input_b1.cpu(), input_b2[:1].cpu())
    flat_b1 = input_b1.reshape(-1, input_b1.shape[-1])
    flat_b2 = input_b2.reshape(-1, input_b2.shape[-1])
    output_b1, kernels_b1 = _profile_mm(flat_b1, weight_t)
    output_b2, kernels_b2 = _profile_mm(flat_b2, weight_t)
    target_rows = input_b1.shape[1]
    output_comparison = compare_outputs(
        output_b1.reshape(1, target_rows, -1).cpu(),
        output_b2[:target_rows].reshape(1, target_rows, -1).cpu(),
    )

    report = {
        "model": adapter.name,
        "operation": "aten::mm",
        "module": "language_model.layers.0.mlp.down_proj",
        "dtype": str(input_b1.dtype),
        "b1_shape": [list(flat_b1.shape), list(weight_t.shape)],
        "b2_shape": [list(flat_b2.shape), list(weight_t.shape)],
        "input_target_comparison": [asdict(item) for item in input_comparison],
        "output_target_comparison": [asdict(item) for item in output_comparison],
        "stock_cuda_kernels_b1": kernels_b1,
        "stock_cuda_kernels_b2": kernels_b2,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

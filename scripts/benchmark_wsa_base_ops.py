"""Benchmark the two operator repairs exercised by official WSA Base."""

from __future__ import annotations

import argparse
import contextlib
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
from batch_invariant_ops.batch_invariant_ops import mean_batch_invariant
from scripts.benchmark_batch_invariant_ops import run_case


@contextlib.contextmanager
def _linear_mode():
    library = torch.library.Library("aten", "IMPL")
    library.impl("aten::linear", linear_batch_invariant, "CUDA")
    try:
        yield
    finally:
        library._destroy()


def _capture_inputs(adapter, policy, target):
    experts = policy.model.qwen3_vl_with_expert
    down_proj = experts.und_expert.language_model.layers[0].mlp.down_proj
    post_norm = experts.act_expert.layers[0].post_attention_layernorm
    captures = {}

    def capture(label):
        def hook(_module, arguments):
            if label not in captures:
                captures[label] = arguments[0].detach().contiguous()

        return hook

    handles = [
        down_proj.register_forward_pre_hook(capture("linear")),
        post_norm.register_forward_pre_hook(capture("mean")),
    ]
    try:
        with torch.inference_mode(), _linear_mode():
            adapter.run_model(policy, adapter.compose_batch([target]))
    finally:
        for handle in handles:
            handle.remove()
    bias = None if down_proj.bias is None else down_proj.bias.detach()
    return captures, down_proj.weight.detach(), bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.wsa_base_libero",
    )
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]
    inputs, weight, bias = _capture_inputs(adapter, policy, target)

    linear_reference_stock = F.linear(inputs["linear"], weight, bias)
    linear_reference_fixed = linear_batch_invariant(inputs["linear"], weight, bias)
    mean_reference_stock = inputs["mean"].float().pow(2).mean(dim=-1, keepdim=True)
    mean_reference_fixed = mean_batch_invariant(
        inputs["mean"].float().pow(2),
        [-1],
        keepdim=True,
    )
    cases = []
    for batch_size in batch_sizes:
        linear_input = inputs["linear"].repeat(batch_size, 1, 1)
        stock_linear = lambda value=linear_input: F.linear(value, weight, bias)
        fixed_linear = lambda value=linear_input: linear_batch_invariant(value, weight, bias)
        case = run_case(
            "wsa_und_layer0_mlp_down_linear",
            {
                "input": list(linear_input.shape),
                "weight": list(weight.shape),
                "bias": None if bias is None else list(bias.shape),
                "dtype": str(linear_input.dtype),
            },
            stock_linear,
            fixed_linear,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        case["batch_invariance"] = {
            "stock_exact": torch.equal(linear_reference_stock, stock_linear()[:1]),
            "batch_invariant_exact": torch.equal(
                linear_reference_fixed,
                fixed_linear()[:1],
            ),
        }
        cases.append(case)

        mean_input = inputs["mean"].repeat(batch_size, 1, 1).float().pow(2)
        stock_mean = lambda value=mean_input: value.mean(dim=-1, keepdim=True)
        fixed_mean = lambda value=mean_input: mean_batch_invariant(
            value,
            [-1],
            keepdim=True,
        )
        case = run_case(
            "wsa_act_layer0_rmsnorm_mean",
            {
                "input": list(mean_input.shape),
                "dimension": -1,
                "keepdim": True,
                "dtype": str(mean_input.dtype),
            },
            stock_mean,
            fixed_mean,
            batch_size=batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        case["batch_invariance"] = {
            "stock_exact": torch.equal(mean_reference_stock, stock_mean()[:1]),
            "batch_invariant_exact": torch.equal(mean_reference_fixed, fixed_mean()[:1]),
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

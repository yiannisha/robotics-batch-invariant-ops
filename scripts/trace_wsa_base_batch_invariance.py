"""Trace the iterative operator boundaries in official PyTorch WSA Base."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from batch_invariant_ops.batch_invariant_ops import (
    linear_batch_invariant,
    mean_batch_invariant,
)
from scripts.model_invariance.harness import compare_outputs, seed_everything


@contextlib.contextmanager
def _operator_mode(*operators):
    library = torch.library.Library("aten", "IMPL")
    for schema, implementation in operators:
        library.impl(schema, implementation, "CUDA")
    try:
        yield
    finally:
        library._destroy()


def _difference(reference: torch.Tensor, candidate: torch.Tensor):
    candidate = candidate[: reference.shape[0]].contiguous()
    return asdict(compare_outputs(reference.contiguous(), candidate)[0])


def _capture_modules(adapter, policy, samples, modules, mode):
    captures = {}
    handles = []

    def output_hook(label):
        def hook(_module, _arguments, output):
            if label not in captures:
                captures[label] = output.detach().cpu().contiguous()

        return hook

    def input_hook(label):
        def hook(_module, arguments):
            if label not in captures:
                captures[label] = arguments[0].detach().cpu().contiguous()

        return hook

    for label, module, capture_input in modules:
        hook = input_hook(label) if capture_input else output_hook(label)
        registration = (
            module.register_forward_pre_hook if capture_input else module.register_forward_hook
        )
        handles.append(registration(hook))
    try:
        with torch.inference_mode(), mode():
            output = adapter.run_model(policy, adapter.compose_batch(samples))
        captures["final_action"] = output.detach().cpu().contiguous()
    finally:
        for handle in handles:
            handle.remove()
    return captures


def _capture_trace(adapter, policy, samples):
    captures = {}

    def callback(label, value):
        captures[label] = value.detach().cpu().contiguous()

    adapter.trace_callback = callback
    try:
        with torch.inference_mode(), set_batch_invariant_mode():
            output = adapter.run_model(policy, adapter.compose_batch(samples))
        captures["final_action"] = output.detach().cpu().contiguous()
    finally:
        adapter.trace_callback = None
    return captures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.wsa_base_libero",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    target, companion = examples

    experts = policy.model.qwen3_vl_with_expert
    und_layer0 = experts.und_expert.language_model.layers[0]
    act_layer0 = experts.act_expert.layers[0]
    stock_modules = (
        ("und.layer0.gate_proj", und_layer0.mlp.gate_proj, False),
        ("und.layer0.up_proj", und_layer0.mlp.up_proj, False),
        ("und.layer0.down_proj.input", und_layer0.mlp.down_proj, True),
        ("und.layer0.down_proj.output", und_layer0.mlp.down_proj, False),
    )
    stock_b1 = _capture_modules(adapter, policy, [target], stock_modules, contextlib.nullcontext)
    stock_b2 = _capture_modules(
        adapter,
        policy,
        [target, companion],
        stock_modules,
        contextlib.nullcontext,
    )

    @contextlib.contextmanager
    def linear_only():
        with _operator_mode(("aten::linear", linear_batch_invariant)):
            yield

    linear_modules = (
        ("act.layer0.post_norm.input", act_layer0.post_attention_layernorm, True),
        ("act.layer0.post_norm.output", act_layer0.post_attention_layernorm, False),
    )
    linear_b1 = _capture_modules(adapter, policy, [target], linear_modules, linear_only)
    linear_b2 = _capture_modules(adapter, policy, [target, companion], linear_modules, linear_only)

    @contextlib.contextmanager
    def linear_mean():
        with _operator_mode(
            ("aten::linear", linear_batch_invariant),
            ("aten::mean.dim", mean_batch_invariant),
        ):
            yield

    minimal_b1 = _capture_modules(adapter, policy, [target], (), linear_mean)
    minimal_b2 = _capture_modules(adapter, policy, [target, companion], (), linear_mean)

    fixed_b1 = _capture_trace(adapter, policy, [target])
    fixed_b2 = _capture_trace(adapter, policy, [target, companion])
    if fixed_b1.keys() != fixed_b2.keys():
        raise RuntimeError("B=1 and B=2 WSA traces have different structures")

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "comparison": "target LIBERO frame 0 at B=1 versus unrelated B=2",
        "stock_first_divergence": {
            "module": "und_expert.layers.0.mlp.down_proj",
            "operator": "aten::linear",
            "input_shape": [1, 246, 6144],
            "weight_shape": [2048, 6144],
            "boundaries": {
                label: _difference(stock_b1[label], stock_b2[label])
                for label in (
                    "und.layer0.gate_proj",
                    "und.layer0.up_proj",
                    "und.layer0.down_proj.input",
                    "und.layer0.down_proj.output",
                    "final_action",
                )
            },
        },
        "next_divergence_after_linear_repair": {
            "module": "act_expert.layers.0.post_attention_layernorm",
            "operator": "aten::mean.dim",
            "reduction_shape": [1, 11, 1024],
            "boundaries": {
                label: _difference(linear_b1[label], linear_b2[label])
                for label in (
                    "act.layer0.post_norm.input",
                    "act.layer0.post_norm.output",
                    "final_action",
                )
            },
        },
        "minimal_linear_mean_repair": {
            "final_action": _difference(minimal_b1["final_action"], minimal_b2["final_action"]),
        },
        "full_library_trace": {
            label: _difference(fixed_b1[label], fixed_b2[label]) for label in fixed_b1
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

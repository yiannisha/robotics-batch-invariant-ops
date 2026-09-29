"""Locate Xiaomi-Robotics-1's successive batch-dependent operators."""

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


def _capture(adapter, model, samples, modules, mode, *, trace=False):
    captures = {}
    handles = []
    for label, module, capture_input, call_index in modules:
        calls = []

        def hook(_module, arguments, output=None, *, calls=calls, capture_input=capture_input):
            value = arguments[0] if capture_input else output
            calls.append(value.detach().cpu().contiguous())

        registration = (
            module.register_forward_pre_hook if capture_input else module.register_forward_hook
        )
        handles.append(registration(hook))
        captures[label] = calls

    traced = {}
    if trace:
        adapter.trace_callback = lambda label, value: traced.setdefault(
            label, value.detach().cpu().contiguous()
        )
    try:
        with torch.inference_mode(), mode():
            output = adapter.run_model(model, adapter.compose_batch(samples))
        result = {
            label: values[call_index]
            for (label, _module, _capture_input, call_index), values in zip(
                modules, captures.values(), strict=True
            )
        }
        result.update(traced)
        result["final_action"] = output["actions"].detach().cpu().contiguous()
        return result
    finally:
        adapter.trace_callback = None
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.xiaomi_robotics_1_robocasa",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target, companion = adapter.load_example_inputs(2)

    stock_modules = (
        (
            "vlm.layer0.attention_output",
            model.vlm.model.language_model.layers[0].self_attn.o_proj,
            False,
            0,
        ),
        (
            "vlm.layer0.mlp_down.input",
            model.vlm.model.language_model.layers[0].mlp.down_proj,
            True,
            0,
        ),
        (
            "vlm.layer0.mlp_down.output",
            model.vlm.model.language_model.layers[0].mlp.down_proj,
            False,
            0,
        ),
    )
    stock_b1 = _capture(adapter, model, [target], stock_modules, contextlib.nullcontext)
    stock_b2 = _capture(adapter, model, [target, companion], stock_modules, contextlib.nullcontext)

    linear_ops = (("aten::linear", linear_batch_invariant),)

    @contextlib.contextmanager
    def linear_mode():
        with _operator_mode(*linear_ops):
            yield

    linear_modules = (
        ("dit.layer7.step2.norm.input", model.dit.layers[7].post_layernorm, True, 2),
        ("dit.layer7.step2.norm.output", model.dit.layers[7].post_layernorm, False, 2),
    )
    linear_b1 = _capture(adapter, model, [target], linear_modules, linear_mode, trace=True)
    linear_b2 = _capture(
        adapter, model, [target, companion], linear_modules, linear_mode, trace=True
    )

    minimal_ops = (
        *linear_ops,
        ("aten::mean.dim", mean_batch_invariant),
    )

    @contextlib.contextmanager
    def minimal_mode():
        with _operator_mode(*minimal_ops):
            yield

    minimal_b1 = _capture(adapter, model, [target], (), minimal_mode, trace=True)
    minimal_b2 = _capture(adapter, model, [target, companion], (), minimal_mode, trace=True)
    fixed_b1 = _capture(adapter, model, [target], (), set_batch_invariant_mode, trace=True)
    fixed_b2 = _capture(
        adapter, model, [target, companion], (), set_batch_invariant_mode, trace=True
    )

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "comparison": "target RoboCasa frame 0 at B=1 versus unrelated B=2",
        "stock_first_divergence": {
            "module": "vlm.model.language_model.layers.0.mlp.down_proj",
            "operator": "aten::linear",
            "input_shape": [1, 259, 9728],
            "weight_shape": [2560, 9728],
            "boundaries": {
                label: _difference(stock_b1[label], stock_b2[label]) for label in stock_b1
            },
        },
        "next_divergence_after_linear_repair": {
            "module": "dit.layers.7.post_layernorm, Euler step 2",
            "operator": "aten::mean.dim",
            "reduction_shape": [1, 12, 1024],
            "boundaries": {
                label: _difference(linear_b1[label], linear_b2[label])
                for label in linear_b1
                if label != "vlm.position_ids"
            },
        },
        "minimal_linear_mean_repair": {
            label: _difference(minimal_b1[label], minimal_b2[label])
            for label in minimal_b1
            if label != "vlm.position_ids"
        },
        "full_library_trace": {
            label: _difference(fixed_b1[label], fixed_b2[label])
            for label in fixed_b1
            if label != "vlm.position_ids"
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

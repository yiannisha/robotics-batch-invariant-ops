"""Trace operator boundaries and DDIM propagation in official PyTorch VITRA."""

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
from batch_invariant_ops.batch_invariant_ops import linear_batch_invariant
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


def _capture_modules(adapter, model, samples, modules, mode):
    captures = {}
    handles = []

    def output_hook(label):
        def hook(_module, _arguments, output):
            if label not in captures:
                value = output[0] if isinstance(output, tuple) else output
                captures[label] = value.detach().cpu().contiguous()

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
            output = adapter.run_model(model, adapter.compose_batch(samples))
        captures["final_action"] = output.detach().cpu().contiguous()
    finally:
        for handle in handles:
            handle.remove()
    return captures


def _capture_trace(adapter, model, samples, mode):
    captures = {}

    def callback(label, value):
        captures[label] = value.detach().cpu().contiguous()

    adapter.trace_callback = callback
    try:
        with torch.inference_mode(), mode():
            output = adapter.run_model(model, adapter.compose_batch(samples))
        captures["final_action"] = output.detach().cpu().contiguous()
    finally:
        adapter.trace_callback = None
    return captures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.vitra_vla_3b",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target, companion = adapter.load_example_inputs(2)

    fov = model.fov_encoder.projector
    llm_layer0 = model.backbone.language_model.model.layers[0]
    stock_modules = (
        ("fov.linear0.input", fov[0], True),
        ("fov.linear0.output", fov[0], False),
        ("fov.gelu.output", fov[1], False),
        ("fov.linear2.input", fov[2], True),
        ("fov.linear2.output", fov[2], False),
        ("llm.layer0.input_norm.input", llm_layer0.input_layernorm, True),
    )
    stock_b1 = _capture_modules(adapter, model, [target], stock_modules, contextlib.nullcontext)
    stock_b2 = _capture_modules(
        adapter, model, [target, companion], stock_modules, contextlib.nullcontext
    )

    @contextlib.contextmanager
    def linear_only():
        with _operator_mode(("aten::linear", linear_batch_invariant)):
            yield

    repaired_modules = (
        ("llm.layer0.input_norm.input", llm_layer0.input_layernorm, True),
        ("llm.layer0.input_norm.output", llm_layer0.input_layernorm, False),
    )
    repaired_b1 = _capture_modules(adapter, model, [target], repaired_modules, linear_only)
    repaired_b2 = _capture_modules(
        adapter, model, [target, companion], repaired_modules, linear_only
    )

    stock_trace_b1 = _capture_trace(adapter, model, [target], contextlib.nullcontext)
    stock_trace_b2 = _capture_trace(adapter, model, [target, companion], contextlib.nullcontext)
    fixed_b1 = _capture_trace(adapter, model, [target], set_batch_invariant_mode)
    fixed_b2 = _capture_trace(adapter, model, [target, companion], set_batch_invariant_mode)
    if fixed_b1.keys() != fixed_b2.keys():
        raise RuntimeError("B=1 and B=2 VITRA traces have different structures")

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "comparison": "official target image at B=1 versus unrelated B=2",
        "stock_first_divergence": {
            "module": "fov_encoder.projector.0",
            "operator": "aten::linear",
            "input_shape": [1, 2],
            "weight_shape": [2304, 2],
            "boundaries": {
                label: _difference(stock_b1[label], stock_b2[label]) for label in stock_b1
            },
        },
        "minimal_linear_repair": {
            "boundaries": {
                label: _difference(repaired_b1[label], repaired_b2[label]) for label in repaired_b1
            }
        },
        "stock_iterative_trace": {
            label: _difference(stock_trace_b1[label], stock_trace_b2[label])
            for label in stock_trace_b1
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

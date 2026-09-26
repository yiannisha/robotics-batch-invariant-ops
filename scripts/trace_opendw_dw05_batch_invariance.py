"""Trace the first batch-dependent boundary in official PyTorch DW05."""

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
    convolution_batch_invariant,
    linear_batch_invariant,
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


@contextlib.contextmanager
def _convolution_only():
    with _operator_mode(("aten::convolution", convolution_batch_invariant)):
        yield


@contextlib.contextmanager
def _linear_only():
    with _operator_mode(("aten::linear", linear_batch_invariant)):
        yield


@contextlib.contextmanager
def _convolution_linear():
    with _operator_mode(
        ("aten::convolution", convolution_batch_invariant),
        ("aten::linear", linear_batch_invariant),
    ):
        yield


def _difference(reference: torch.Tensor, candidate: torch.Tensor):
    candidate = candidate[: reference.shape[0]].contiguous()
    return asdict(compare_outputs(reference.contiguous(), candidate)[0])


def _capture(adapter, model, samples, modules, mode):
    captures = {}
    handles = []

    def capture(label, output_index=None):
        def hook(_module, _arguments, output):
            if label in captures:
                return
            value = output
            if output_index is not None:
                value = output[output_index]
            captures[label] = value.detach().cpu().contiguous()

        return hook

    def capture_input(label):
        def hook(_module, arguments):
            if label not in captures:
                captures[label] = arguments[0].detach().cpu().contiguous()

        return hook

    for label, module, kind in modules:
        if kind == "input":
            handles.append(module.register_forward_pre_hook(capture_input(label)))
        else:
            handles.append(module.register_forward_hook(capture(label)))

    def trace_callback(label, value):
        if label not in captures:
            captures[label] = value.detach().cpu().contiguous()

    adapter.trace_callback = trace_callback
    try:
        with torch.inference_mode(), mode():
            output = adapter.run_model(model, adapter.compose_batch(samples))
        captures["final_action"] = output.detach().cpu().contiguous()
    finally:
        adapter.trace_callback = None
        for handle in handles:
            handle.remove()
    return captures


def _compare_traces(reference, candidate):
    if reference.keys() != candidate.keys():
        raise RuntimeError(
            "B=1 and B=2 DW05 traces differ: "
            f"B1-only={reference.keys() - candidate.keys()}, "
            f"B2-only={candidate.keys() - reference.keys()}"
        )
    return {label: _difference(reference[label], candidate[label]) for label in reference}


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

    video = model.video_expert
    action = model.action_expert
    video_block0 = video.blocks[0]
    action_block0 = action.blocks[0]
    modules = (
        ("video.time_embedding.0.input", video.time_embedding[0], "input"),
        ("video.time_embedding.0.output", video.time_embedding[0], "output"),
        ("video.time_embedding.2.output", video.time_embedding[2], "output"),
        ("video.time_projection.1.output", video.time_projection[1], "output"),
        ("video.patch_embedding.input", video.patch_embedding, "input"),
        ("video.patch_embedding.output", video.patch_embedding, "output"),
        ("video.text_embedding.0.input", video.text_embedding[0], "input"),
        ("video.text_embedding.0.output", video.text_embedding[0], "output"),
        ("video.text_embedding.2.output", video.text_embedding[2], "output"),
        ("video.block0.norm1.input", video_block0.norm1, "input"),
        ("video.block0.norm1.output", video_block0.norm1, "output"),
        ("video.block0.q.input", video_block0.self_attn.q, "input"),
        ("video.block0.q.output", video_block0.self_attn.q, "output"),
        ("video.block0.k.output", video_block0.self_attn.k, "output"),
        ("video.block0.v.output", video_block0.self_attn.v, "output"),
        ("action.time_embedding.0.input", action.time_embedding[0], "input"),
        ("action.time_embedding.0.output", action.time_embedding[0], "output"),
        ("action.time_embedding.2.output", action.time_embedding[2], "output"),
        ("action.time_projection.1.output", action.time_projection[1], "output"),
        ("action.encoder.input", action.action_encoder, "input"),
        ("action.encoder.output", action.action_encoder, "output"),
        ("action.block0.norm1.input", action_block0.norm1, "input"),
        ("action.block0.norm1.output", action_block0.norm1, "output"),
        ("action.block0.q.output", action_block0.self_attn.q, "output"),
    )
    modes = {
        "stock": contextlib.nullcontext,
        "convolution_only": _convolution_only,
        "linear_only": _linear_only,
        "convolution_linear": _convolution_linear,
        "full_library": set_batch_invariant_mode,
    }
    traces = {}
    for name, mode in modes.items():
        b1 = _capture(adapter, model, [target], modules, mode)
        b2 = _capture(adapter, model, [target, companion], modules, mode)
        traces[name] = _compare_traces(b1, b2)

    stock_first = next(
        (
            {"boundary": label, **comparison}
            for label, comparison in traces["stock"].items()
            if not comparison["exact"]
        ),
        None,
    )
    report = {
        "model": adapter.name,
        "seed": args.seed,
        "comparison": "official target frame at B=1 versus unrelated frame at B=2",
        "stock_first_divergence": stock_first,
        "operator_mapping": {
            "video.patch_embedding.output": "aten::convolution (Conv3d)",
            "linear_boundaries": "aten::linear",
            "normalization_boundaries": "native_layer_norm or mean",
        },
        "traces": traces,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

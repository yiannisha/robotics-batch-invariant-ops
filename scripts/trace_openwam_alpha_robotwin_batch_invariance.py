"""Trace the first batch-dependent boundary in official PyTorch OpenWAM."""

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

    def capture(label, *, input_value=False):
        def hook(_module, arguments, output=None):
            if label in captures:
                return
            value = arguments[0] if input_value else output
            captures[label] = value.detach().cpu().contiguous()

        return hook

    for label, module, kind in modules:
        if kind == "input":
            handles.append(module.register_forward_pre_hook(capture(label, input_value=True)))
        else:
            handles.append(module.register_forward_hook(capture(label)))

    def trace_callback(label, value):
        if label not in captures:
            captures[label] = value.detach().cpu().contiguous()

    adapter.trace_callback = trace_callback
    try:
        with torch.inference_mode(), mode():
            output = adapter.run_model(model, adapter.compose_batch(samples))
        captures["final.actions"] = output["actions"].contiguous()
        captures["final.video_latent"] = output["video_latent"].contiguous()
    finally:
        adapter.trace_callback = None
        for handle in handles:
            handle.remove()
    return captures


def _compare_traces(reference, candidate):
    if reference.keys() != candidate.keys():
        raise RuntimeError(
            "B=1 and B=2 traces differ: "
            f"B1-only={reference.keys() - candidate.keys()}, "
            f"B2-only={candidate.keys() - reference.keys()}"
        )
    return {label: _difference(reference[label], candidate[label]) for label in reference}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.openwam_alpha_robotwin",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target, companion = adapter.load_example_inputs(2)

    video = model.video_backbone.dit
    action = model.action_backbone
    video_block0 = video.blocks[0]
    action_block0 = action.blocks[0]
    modules = (
        ("video.time_embedding.0.input", video.time_embedding[0], "input"),
        ("video.time_embedding.0.output", video.time_embedding[0], "output"),
        ("video.time_embedding.2.output", video.time_embedding[2], "output"),
        ("video.time_projection.1.output", video.time_projection[1], "output"),
        ("video.text_embedding.0.input", video.text_embedding[0], "input"),
        ("video.text_embedding.0.output", video.text_embedding[0], "output"),
        ("video.text_embedding.2.output", video.text_embedding[2], "output"),
        ("video.patch_embedding.input", video.patch_embedding, "input"),
        ("video.patch_embedding.output", video.patch_embedding, "output"),
        ("action.encoder.input", action.action_encoder, "input"),
        ("action.encoder.output", action.action_encoder, "output"),
        ("action.time_embedding.0.input", action.time_embedding.mlp[0], "input"),
        ("action.time_embedding.0.output", action.time_embedding.mlp[0], "output"),
        ("action.time_embedding.2.output", action.time_embedding.mlp[2], "output"),
        ("action.time_projection.output", action.time_projection.proj[1], "output"),
        ("action.text_embedding.0.input", action.text_embedding[0], "input"),
        ("action.text_embedding.0.output", action.text_embedding[0], "output"),
        ("video.block0.q.input", video_block0.self_attn.q, "input"),
        ("video.block0.q.output", video_block0.self_attn.q, "output"),
        ("video.block0.k.output", video_block0.self_attn.k, "output"),
        ("video.block0.v.output", video_block0.self_attn.v, "output"),
        ("action.block0.q.input", action_block0.self_attn.q, "input"),
        ("action.block0.q.output", action_block0.self_attn.q, "output"),
        ("action.block0.k.output", action_block0.self_attn.k, "output"),
        ("action.block0.v.output", action_block0.self_attn.v, "output"),
        ("video.block0.self_o.input", video_block0.self_attn.o, "input"),
        ("video.block0.self_o.output", video_block0.self_attn.o, "output"),
        ("video.block0.cross_q.input", video_block0.cross_attn.q, "input"),
        ("video.block0.cross_q.output", video_block0.cross_attn.q, "output"),
        ("video.block0.cross_k.output", video_block0.cross_attn.k, "output"),
        ("video.block0.cross_v.output", video_block0.cross_attn.v, "output"),
        ("video.block0.cross_o.input", video_block0.cross_attn.o, "input"),
        ("video.block0.cross_o.output", video_block0.cross_attn.o, "output"),
        ("video.block0.ffn0.input", video_block0.ffn[0], "input"),
        ("video.block0.ffn0.output", video_block0.ffn[0], "output"),
        ("video.block0.ffn2.output", video_block0.ffn[2], "output"),
        ("action.block0.self_o.input", action_block0.self_attn.o, "input"),
        ("action.block0.self_o.output", action_block0.self_attn.o, "output"),
        ("action.block0.cross_q.input", action_block0.cross_attn.q, "input"),
        ("action.block0.cross_q.output", action_block0.cross_attn.q, "output"),
        ("action.block0.cross_k.output", action_block0.cross_attn.k, "output"),
        ("action.block0.cross_v.output", action_block0.cross_attn.v, "output"),
        ("action.block0.cross_o.input", action_block0.cross_attn.o, "input"),
        ("action.block0.cross_o.output", action_block0.cross_attn.o, "output"),
        ("action.block0.ffn0.input", action_block0.ffn[0], "input"),
        ("action.block0.ffn0.output", action_block0.ffn[0], "output"),
        ("action.block0.ffn2.output", action_block0.ffn[2], "output"),
    )
    modules = list(modules)
    for layer_id in range(1, len(video.blocks)):
        modules.extend(
            (
                (
                    f"video.block{layer_id}.ffn2.output",
                    video.blocks[layer_id].ffn[2],
                    "output",
                ),
                (
                    f"action.block{layer_id}.ffn2.output",
                    action.blocks[layer_id].ffn[2],
                    "output",
                ),
            )
        )
    modules.extend(
        (
            ("video.head.input", video.head.head, "input"),
            ("video.head.output", video.head.head, "output"),
            ("action.decoder.input", action.action_decoder, "input"),
            ("action.decoder.output", action.action_decoder, "output"),
        )
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

    execution_order = [
        "input.video_latent",
        "input.action_noise",
        "input.proprio",
        "video.time_embedding.0.input",
        "video.time_embedding.0.output",
        "video.time_embedding.2.output",
        "video.time_projection.1.output",
        "video.text_embedding.0.input",
        "video.text_embedding.0.output",
        "video.text_embedding.2.output",
        "video.patch_embedding.input",
        "video.patch_embedding.output",
        "action.encoder.input",
        "action.encoder.output",
        "action.time_embedding.0.input",
        "action.time_embedding.0.output",
        "action.time_embedding.2.output",
        "action.time_projection.output",
        "action.text_embedding.0.input",
        "action.text_embedding.0.output",
        "video.block0.q.input",
        "video.block0.q.output",
        "video.block0.k.output",
        "video.block0.v.output",
        "action.block0.q.input",
        "action.block0.q.output",
        "action.block0.k.output",
        "action.block0.v.output",
        "video.block0.self_o.input",
        "video.block0.self_o.output",
        "video.block0.cross_q.input",
        "video.block0.cross_q.output",
        "video.block0.cross_k.output",
        "video.block0.cross_v.output",
        "video.block0.cross_o.input",
        "video.block0.cross_o.output",
        "video.block0.ffn0.input",
        "video.block0.ffn0.output",
        "video.block0.ffn2.output",
        "action.block0.self_o.input",
        "action.block0.self_o.output",
        "action.block0.cross_q.input",
        "action.block0.cross_q.output",
        "action.block0.cross_k.output",
        "action.block0.cross_v.output",
        "action.block0.cross_o.input",
        "action.block0.cross_o.output",
        "action.block0.ffn0.input",
        "action.block0.ffn0.output",
        "action.block0.ffn2.output",
    ]
    for layer_id in range(1, len(video.blocks)):
        execution_order.extend(
            (
                f"video.block{layer_id}.ffn2.output",
                f"action.block{layer_id}.ffn2.output",
            )
        )
    execution_order.extend(
        (
            "video.head.input",
            "video.head.output",
            "action.decoder.input",
            "action.decoder.output",
        )
    )
    stock_first = next(
        (
            {"boundary": label, **traces["stock"][label]}
            for label in execution_order
            if not traces["stock"][label]["exact"]
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
        },
        "traces": traces,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

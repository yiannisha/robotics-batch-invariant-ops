"""Trace the operator boundaries required by official PyTorch SpatialVLA."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops.batch_invariant_ops import (
    _log_softmax_batch_invariant,
    _softmax_batch_invariant,
    addmm_batch_invariant,
    bmm_batch_invariant,
    convolution_batch_invariant,
    linalg_vector_norm_batch_invariant,
    mean_batch_invariant,
    mm_batch_invariant,
    scaled_dot_product_attention_batch_invariant,
)
from scripts.model_invariance.harness import seed_everything

OPERATORS = (
    ("convolution", "aten::convolution", convolution_batch_invariant),
    ("mm", "aten::mm", mm_batch_invariant),
    ("addmm", "aten::addmm", addmm_batch_invariant),
    ("bmm", "aten::bmm", bmm_batch_invariant),
    (
        "scaled_dot_product_attention",
        "aten::scaled_dot_product_attention",
        scaled_dot_product_attention_batch_invariant,
    ),
    ("softmax", "aten::_softmax", _softmax_batch_invariant),
    ("log_softmax", "aten::_log_softmax", _log_softmax_batch_invariant),
    ("mean", "aten::mean.dim", mean_batch_invariant),
    ("linalg_vector_norm", "aten::linalg_vector_norm", linalg_vector_norm_batch_invariant),
)


@contextlib.contextmanager
def _operator_mode(omit: str | None = None, include: set[str] | None = None):
    library = torch.library.Library("aten", "IMPL")
    for name, schema, implementation in OPERATORS:
        if name != omit and (include is None or name in include):
            library.impl(schema, implementation, "CUDA")
    try:
        yield
    finally:
        library._destroy()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    candidate = candidate[: reference.shape[0]]
    mask = reference != candidate
    absolute = (reference.float() - candidate.float()).abs()
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact": not bool(mask.any()),
        "differing_elements": int(mask.sum()),
        "max_abs_difference": float(absolute.max()) if absolute.numel() else 0.0,
    }


def _capture_module(module, run):
    captures = []

    def hook(_module, args, kwargs, output):
        inputs = kwargs.get("hidden_states")
        if inputs is None:
            inputs = next(value for value in args if torch.is_tensor(value))
        if isinstance(output, (tuple, list)):
            output = next(value for value in output if torch.is_tensor(value))
        captures.append((inputs[:1].detach().cpu(), output[:1].detach().cpu()))

    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        run()
    finally:
        handle.remove()
    return captures


def _image_run(adapter, policy, images):
    inputs = adapter.prepare_batch(policy, images)
    with torch.inference_mode():
        return policy.model.get_image_features(inputs["pixel_values"], inputs["intrinsic"])


def _module_boundary(
    adapter, policy, target, module, *, omit=None, include=None, call=0, generate=False
):
    mode = contextlib.nullcontext() if omit == "stock" else _operator_mode(omit, include=include)
    with mode:
        reference = _capture_module(
            module,
            lambda: (
                adapter.generate(policy, [target])
                if generate
                else _image_run(adapter, policy, [target])
            ),
        )
        candidate = _capture_module(
            module,
            lambda: (
                adapter.generate(policy, [target, target])
                if generate
                else _image_run(adapter, policy, [target, target])
            ),
        )
    reference_input, reference_output = reference[call]
    candidate_input, candidate_output = candidate[call]
    return {
        "call": call,
        "input": _comparison(reference_input, candidate_input),
        "output": _comparison(reference_output, candidate_output),
    }


def _ablation(adapter, policy, target, omit):
    with _operator_mode(omit):
        reference_tokens, reference_scores = adapter.generate(policy, [target], output_scores=True)
        candidate_tokens, candidate_scores = adapter.generate(
            policy, [target, target], output_scores=True
        )
    score_differences = []
    for call, (reference, candidate) in enumerate(zip(reference_scores, candidate_scores)):
        comparison = _comparison(reference.cpu(), candidate[:1].cpu())
        if not comparison["exact"]:
            score_differences.append({"call": call, **comparison})
    return {
        "tokens": _comparison(reference_tokens.cpu(), candidate_tokens[:1].cpu()),
        "score_difference_calls": score_differences,
    }


def _stage(adapter, policy, target, include):
    with _operator_mode(include=set(include)):
        reference_tokens, reference_scores = adapter.generate(policy, [target], output_scores=True)
        candidate_tokens, candidate_scores = adapter.generate(
            policy, [target, target], output_scores=True
        )
    score_differences = []
    for call, (reference, candidate) in enumerate(zip(reference_scores, candidate_scores)):
        comparison = _comparison(reference.cpu(), candidate[:1].cpu())
        if not comparison["exact"]:
            score_differences.append({"call": call, **comparison})
    return {
        "operators": list(include),
        "tokens": _comparison(reference_tokens.cpu(), candidate_tokens[:1].cpu()),
        "score_difference_calls": score_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.spatialvla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]

    zoe = policy.model.vision_zoe_model
    baseline_conv = zoe.neck.reassemble_stage.layers[3].resize
    mean_attractor = zoe.metric_head.attractors["nyu"][1]
    bmm_attention = policy.model.language_model.model.layers[2].self_attn

    # Evaluate this boundary before any stock cuDNN warmup changes the later
    # operator-selection cache.  The exact target tensor entering the second
    # ZoeDepth attractor is the first reduction boundary after convolution is
    # repaired in this staged run.
    mean_boundary = _module_boundary(
        adapter,
        policy,
        target,
        mean_attractor,
        include={"convolution"},
    )
    baseline_boundary = _module_boundary(adapter, policy, target, baseline_conv, omit="stock")
    bmm_boundary = _module_boundary(
        adapter,
        policy,
        target,
        bmm_attention,
        omit="bmm",
        call=2,
        generate=True,
    )

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "boundaries": {
            "baseline_first_convolution": {
                "module": "vision_zoe_model.neck.reassemble_stage.layers.3.resize",
                "operator": "aten::convolution",
                **baseline_boundary,
            },
            "mean_after_convolution_repair": {
                "module": "vision_zoe_model.metric_head.attractors.nyu.1",
                "operator": "aten::mean.dim",
                **mean_boundary,
            },
            "without_bmm": {
                "module": "language_model.model.layers.2.self_attn",
                "operator": "aten::bmm",
                **bmm_boundary,
            },
        },
        "removal_ablations": {
            omit or "full": _ablation(adapter, policy, target, omit)
            for omit in ("convolution", "mm", "bmm", "softmax", "mean", None)
        },
        "staged_repair": {
            name: _stage(adapter, policy, target, operators)
            for name, operators in (
                ("convolution", ("convolution",)),
                ("convolution_mean", ("convolution", "mean")),
                ("convolution_mean_mm", ("convolution", "mean", "mm")),
                (
                    "convolution_mean_mm_bmm",
                    ("convolution", "mean", "mm", "bmm"),
                ),
                ("full", tuple(item[0] for item in OPERATORS)),
            )
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

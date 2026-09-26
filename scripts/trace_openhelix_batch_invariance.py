"""Trace the two operator boundaries exposed by official PyTorch OpenHelix."""

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
from scripts.model_invariance.harness import compare_outputs, seed_everything

PRE_LINEAR_OPERATORS = (
    ("aten::mm", mm_batch_invariant),
    ("aten::addmm", addmm_batch_invariant),
    ("aten::bmm", bmm_batch_invariant),
    ("aten::convolution", convolution_batch_invariant),
    ("aten::scaled_dot_product_attention", scaled_dot_product_attention_batch_invariant),
    ("aten::_softmax", _softmax_batch_invariant),
    ("aten::_log_softmax", _log_softmax_batch_invariant),
    ("aten::mean.dim", mean_batch_invariant),
    ("aten::linalg_vector_norm", linalg_vector_norm_batch_invariant),
)


@contextlib.contextmanager
def _pre_linear_mode():
    library = torch.library.Library("aten", "IMPL")
    for schema, implementation in PRE_LINEAR_OPERATORS:
        library.impl(schema, implementation, "CUDA")
    try:
        yield
    finally:
        library._destroy()


def _difference(reference, candidate):
    return asdict(compare_outputs(reference, candidate[: reference.shape[0]])[0])


def _planner_capture(adapter, models, samples, modules):
    captures = {}
    handles = []
    for label, module, capture_input in modules:
        if capture_input:

            def hook(_module, arguments, *, label=label):
                captures[label] = arguments[0].detach().cpu()

            handles.append(module.register_forward_pre_hook(hook))
        else:

            def hook(_module, _arguments, output, *, label=label):
                if isinstance(output, (tuple, list)):
                    output = next(value for value in output if torch.is_tensor(value))
                captures[label] = output.detach().cpu()

            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            features = (
                adapter._planner_features(
                    models,
                    [sample["planner_image"] for sample in samples],
                    [sample["instruction"] for sample in samples],
                )
                .detach()
                .cpu()
            )
    finally:
        for handle in handles:
            handle.remove()
    captures["planner.action_token"] = features
    return captures


def _attention_values(captures, attention):
    query = captures["clip.layer0.q"].to("cuda") * attention.scale
    key = captures["clip.layer0.k"].to("cuda")
    value = captures["clip.layer0.v"].to("cuda")
    batch_size, sequence_length, _ = query.shape

    def shaped(tensor):
        return (
            tensor.view(batch_size, sequence_length, attention.num_heads, attention.head_dim)
            .transpose(1, 2)
            .reshape(batch_size * attention.num_heads, sequence_length, attention.head_dim)
        )

    query, key, value = shaped(query), shaped(key), shaped(value)
    scores = torch.bmm(query, key.transpose(1, 2))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.bmm(probabilities, value)
    return scores.cpu(), probabilities.cpu(), output.cpu()


def _full_run(adapter, models, samples):
    with torch.inference_mode():
        output = adapter.run_model(models, adapter.compose_batch(samples))
    return output.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.openhelix_calvin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    models = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    target = examples[0]

    vision = models.planner.get_model().get_vision_tower().vision_tower.vision_model
    attention = vision.encoder.layers[0].self_attn
    projector = models.planner.get_model().mm_projector
    stock_modules = (
        ("clip.patch_embedding", vision.embeddings.patch_embedding, False),
        ("clip.layer0.q", attention.q_proj, False),
        ("clip.layer0.k", attention.k_proj, False),
        ("clip.layer0.v", attention.v_proj, False),
        ("clip.layer0.attention", attention, False),
    )
    stock_b1 = _planner_capture(adapter, models, [target], stock_modules)
    stock_b2 = _planner_capture(adapter, models, [target, target], stock_modules)
    scores_b1, probabilities_b1, output_b1 = _attention_values(stock_b1, attention)
    scores_b2, probabilities_b2, output_b2 = _attention_values(stock_b2, attention)

    projector_modules = (
        ("projector.input", projector, True),
        ("projector.output", projector, False),
    )
    with _pre_linear_mode():
        pre_linear_b1 = _planner_capture(adapter, models, [target], projector_modules)
        pre_linear_b2 = _planner_capture(adapter, models, [target, target], projector_modules)

    with set_batch_invariant_mode():
        seed_everything(args.seed)
        fixed_b1 = _full_run(adapter, models, [target])
        seed_everything(args.seed)
        fixed_b2 = _full_run(adapter, models, [target, examples[1]])

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "stock_first_divergence": {
            "module": "vision_tower.vision_model.encoder.layers.0.self_attn",
            "operator": "aten::bmm",
            "logical_shapes": {
                "query": [1, 16, 257, 64],
                "key": [1, 16, 257, 64],
                "attention_scores": [1, 16, 257, 257],
            },
            "boundaries": {
                label: _difference(stock_b1[label], stock_b2[label])
                for label, _, _ in stock_modules
            }
            | {
                "attention_scores": _difference(scores_b1[:16], scores_b2[:16]),
                "attention_probabilities": _difference(
                    probabilities_b1[:16], probabilities_b2[:16]
                ),
                "attention_value_bmm": _difference(output_b1[:16], output_b2[:16]),
            },
        },
        "next_divergence_after_existing_operator_repairs": {
            "module": "model.mm_projector",
            "operator": "aten::linear",
            "input_shape": [1, 256, 1024],
            "weight_shape": [4096, 1024],
            "boundaries": {
                label: _difference(pre_linear_b1[label], pre_linear_b2[label])
                for label in ("projector.input", "projector.output", "planner.action_token")
            },
        },
        "full_repair_unrelated_b2": {
            "final_action": _difference(fixed_b1, fixed_b2),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

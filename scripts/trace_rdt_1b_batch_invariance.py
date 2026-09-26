"""Trace RDT-1B's stock SDPA and large-batch addmm boundaries."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops.batch_invariant_ops import (
    addmm_batch_invariant,
    scaled_dot_product_attention_batch_invariant,
)
from scripts.model_invariance.harness import compare_outputs


def _tensor_difference(reference, candidate):
    return asdict(compare_outputs(reference, candidate)[0])


def _run(adapter, policy, samples, captures):
    values = {}
    handles = []
    for label, module, capture_input in captures:
        if capture_input:

            def hook(current_module, arguments, *, label=label):
                if label not in values:
                    values[label] = arguments[0][:1].detach().cpu()

            handles.append(module.register_forward_pre_hook(hook))
        else:

            def hook(current_module, arguments, output, *, label=label):
                if label not in values:
                    tensor = output[0] if isinstance(output, tuple) else output
                    values[label] = tensor[:1].detach().cpu()

            handles.append(module.register_forward_hook(hook))
    adapter.reset_model(policy)
    with torch.inference_mode():
        output = adapter.run_model(policy, adapter.compose_batch(samples))
    for handle in handles:
        handle.remove()
    return values, output[:1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.rdt_maniskill")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(64))
    block = policy.model.model.blocks[1]
    stock_captures = (
        ("preceding_block_output", policy.model.model.blocks[0], False),
        ("query_input", block.cross_attn.q, True),
        ("query_projection", block.cross_attn.q, False),
        ("key_value_input", block.cross_attn.kv, True),
        ("key_value_projection", block.cross_attn.kv, False),
        ("sdpa_output", block.cross_attn.proj, True),
        ("attention_projection", block.cross_attn.proj, False),
    )
    stock_b1, stock_output_b1 = _run(adapter, policy, [examples[0]], stock_captures)
    stock_b2, stock_output_b2 = _run(adapter, policy, [examples[0], examples[0]], stock_captures)

    library = torch.library.Library("aten", "IMPL")
    library.impl(
        "aten::scaled_dot_product_attention",
        scaled_dot_product_attention_batch_invariant,
        "CUDA",
    )
    language_linear = policy.model.lang_adaptor[0]
    addmm_captures = (
        ("language_adaptor_input", language_linear, True),
        ("language_adaptor_output", language_linear, False),
    )
    sdpa_b1, sdpa_output_b1 = _run(adapter, policy, [examples[0]], addmm_captures)
    _, sdpa_output_b2 = _run(adapter, policy, [examples[0], examples[0]], addmm_captures)
    sdpa_b64, sdpa_output_b64 = _run(adapter, policy, [examples[0]] * 64, addmm_captures)

    library.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    _, minimal_output_b1 = _run(adapter, policy, [examples[0]], ())
    _, minimal_duplicate_b64 = _run(adapter, policy, [examples[0]] * 64, ())
    _, minimal_unrelated_b64 = _run(adapter, policy, examples, ())
    library._destroy()

    report = {
        "model": adapter.name,
        "stock_b2_first_divergence": {
            "module": "model.blocks.1.cross_attn",
            "operator": "aten::scaled_dot_product_attention",
            "preceding_block_output": _tensor_difference(
                stock_b1["preceding_block_output"], stock_b2["preceding_block_output"]
            ),
            "query_input": _tensor_difference(stock_b1["query_input"], stock_b2["query_input"]),
            "query_projection": _tensor_difference(
                stock_b1["query_projection"], stock_b2["query_projection"]
            ),
            "key_value_input": _tensor_difference(
                stock_b1["key_value_input"], stock_b2["key_value_input"]
            ),
            "key_value_projection": _tensor_difference(
                stock_b1["key_value_projection"], stock_b2["key_value_projection"]
            ),
            "operator_output_before_projection": _tensor_difference(
                stock_b1["sdpa_output"], stock_b2["sdpa_output"]
            ),
            "attention_projection": _tensor_difference(
                stock_b1["attention_projection"], stock_b2["attention_projection"]
            ),
            "final_action": _tensor_difference(stock_output_b1, stock_output_b2),
            "logical_qkv_shapes": {
                "query": [1, 32, 67, 64],
                "key": [1, 32, 4374, 64],
                "value": [1, 32, 4374, 64],
            },
        },
        "sdpa_only": {
            "b2_final_action": _tensor_difference(sdpa_output_b1, sdpa_output_b2),
            "b64_next_divergence": {
                "module": "lang_adaptor.0",
                "operator": "aten::addmm",
                "input": _tensor_difference(
                    sdpa_b1["language_adaptor_input"],
                    sdpa_b64["language_adaptor_input"],
                ),
                "output": _tensor_difference(
                    sdpa_b1["language_adaptor_output"],
                    sdpa_b64["language_adaptor_output"],
                ),
                "final_action": _tensor_difference(sdpa_output_b1, sdpa_output_b64),
            },
        },
        "minimal_sdpa_plus_addmm": {
            "duplicate_b64": _tensor_difference(minimal_output_b1, minimal_duplicate_b64),
            "unrelated_b64": _tensor_difference(minimal_output_b1, minimal_unrelated_b64),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Trace CogACT-Small's DINO SDPA and Llama MLP GEMM boundaries."""

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
    mm_batch_invariant,
    scaled_dot_product_attention_batch_invariant,
)
from scripts.model_invariance.harness import compare_outputs


def _difference(reference, candidate):
    return asdict(compare_outputs(reference, candidate)[0])


def _run(adapter, policy, samples, captures=()):
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
    with torch.inference_mode():
        output = adapter.run_model(policy, adapter.compose_batch(samples))[:1]
    for handle in handles:
        handle.remove()
    return values, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.cogact_small")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(64))
    vision = policy.model.vlm.vision_backbone
    dino_block = vision.dino_featurizer.blocks[0]
    stock_captures = (
        ("patch_input", vision.dino_featurizer.patch_embed.proj, True),
        ("patch_output", vision.dino_featurizer.patch_embed.proj, False),
        ("attention_input", dino_block.attn.qkv, True),
        ("qkv_projection", dino_block.attn.qkv, False),
        ("sdpa_output", dino_block.attn.proj, True),
        ("attention_projection", dino_block.attn.proj, False),
    )
    stock_b1, stock_output_b1 = _run(adapter, policy, [examples[0]], stock_captures)
    stock_b2, stock_output_b2 = _run(adapter, policy, [examples[0], examples[0]], stock_captures)

    library = torch.library.Library("aten", "IMPL")
    library.impl(
        "aten::scaled_dot_product_attention",
        scaled_dot_product_attention_batch_invariant,
        "CUDA",
    )
    llama_layer = policy.model.vlm.llm_backbone.llm.model.layers[0]
    sdpa_captures = (
        ("dino_final", vision.dino_featurizer.blocks[-1], False),
        ("siglip_final", vision.siglip_featurizer.blocks[-1], False),
        ("projector_final", policy.model.vlm.projector, False),
        ("llama_input_norm", llama_layer.input_layernorm, False),
        ("llama_q_projection", llama_layer.self_attn.q_proj, False),
        ("llama_k_projection", llama_layer.self_attn.k_proj, False),
        ("llama_v_projection", llama_layer.self_attn.v_proj, False),
        ("llama_attention_output", llama_layer.self_attn.o_proj, True),
        ("llama_o_projection", llama_layer.self_attn.o_proj, False),
        ("llama_post_attention_norm", llama_layer.post_attention_layernorm, False),
        ("llama_gate_projection", llama_layer.mlp.gate_proj, False),
        ("llama_up_projection", llama_layer.mlp.up_proj, False),
        ("llama_down_input", llama_layer.mlp.down_proj, True),
        ("llama_down_projection", llama_layer.mlp.down_proj, False),
        ("llama_layer_output", llama_layer, False),
    )
    sdpa_b1, sdpa_output_b1 = _run(adapter, policy, [examples[0]], sdpa_captures)
    sdpa_b2, sdpa_output_b2 = _run(adapter, policy, [examples[0], examples[0]], sdpa_captures)

    library.impl("aten::mm", mm_batch_invariant, "CUDA")
    library.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    _, minimal_output_b1 = _run(adapter, policy, [examples[0]])
    _, minimal_duplicate_b64 = _run(adapter, policy, [examples[0]] * 64)
    _, minimal_unrelated_b64 = _run(adapter, policy, examples)
    library._destroy()

    ordered_sdpa_boundaries = [
        "dino_final",
        "siglip_final",
        "projector_final",
        "llama_input_norm",
        "llama_q_projection",
        "llama_k_projection",
        "llama_v_projection",
        "llama_attention_output",
        "llama_o_projection",
        "llama_post_attention_norm",
        "llama_gate_projection",
        "llama_up_projection",
        "llama_down_input",
        "llama_down_projection",
        "llama_layer_output",
    ]
    report = {
        "model": adapter.name,
        "stock_b2_first_divergence": {
            "module": "vlm.vision_backbone.dino_featurizer.blocks.0.attn",
            "operator": "aten::scaled_dot_product_attention",
            "logical_qkv_shapes": {
                "query": [1, 16, 261, 64],
                "key": [1, 16, 261, 64],
                "value": [1, 16, 261, 64],
            },
            "boundaries": {
                label: _difference(stock_b1[label], stock_b2[label])
                for label, _, _ in stock_captures
            },
            "final_action": _difference(stock_output_b1, stock_output_b2),
        },
        "sdpa_only_b2": {
            "ordered_boundaries": [
                {
                    "label": label,
                    **_difference(sdpa_b1[label], sdpa_b2[label]),
                }
                for label in ordered_sdpa_boundaries
            ],
            "final_action": _difference(sdpa_output_b1, sdpa_output_b2),
        },
        "minimal_sdpa_plus_mm_addmm": {
            "duplicate_b64": _difference(minimal_output_b1, minimal_duplicate_b64),
            "unrelated_b64": _difference(minimal_output_b1, minimal_unrelated_b64),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

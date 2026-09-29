"""Trace the first numerical batch divergence in the LeRobot Pi0-FAST port.

This script complements the end-to-end model harness.  Greedy decoding can
produce identical action tokens even when its logits differ, so we record the
target sample at the first vision convolution, vision tower, multimodal
projector, and language head for B=1 and unrelated B=2 runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, seed_everything


def _tensor_from_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for attribute in ("pooler_output", "last_hidden_state"):
        tensor = getattr(value, attribute, None)
        if isinstance(tensor, torch.Tensor):
            return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _tensor_from_output(item)
            except TypeError:
                pass
    raise TypeError(f"cannot select a tensor from hook output {type(value).__name__}")


def _target_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach()[:1].contiguous().cpu()


def _hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    record: dict[str, Any] = {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact": exact,
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }
    if reference.shape != candidate.shape:
        record["candidate_shape"] = list(candidate.shape)
        return record
    different = reference != candidate
    record["differing_elements"] = int(different.sum().item())
    locations = torch.nonzero(different, as_tuple=False)
    record["first_differing_index"] = (
        [int(item) for item in locations[0].tolist()] if locations.numel() else None
    )
    if reference.is_floating_point():
        absolute = (reference.float() - candidate.float()).abs()
        record["max_abs_difference"] = float(absolute.max().item())
        record["mean_abs_difference"] = float(absolute.mean().item())
    return record


def _register_hooks(model: torch.nn.Module, captures: dict[str, list[torch.Tensor]]):
    core = model.paligemma_with_expert.paligemma
    vision_tower = core.model.vision_tower
    projector = core.model.multi_modal_projector
    lm_head = core.lm_head
    first_conv_name, first_conv = next(
        (name, module)
        for name, module in vision_tower.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    )
    first_language_layer = core.model.language_model.layers[0]

    handles = []

    def capture_output(label: str):
        def hook(_module, _inputs, output):
            captures[label].append(_target_cpu(_tensor_from_output(output)))

        return hook

    def capture_input(label: str):
        def hook(_module, inputs):
            captures[label].append(_target_cpu(_tensor_from_output(inputs)))

        return hook

    def capture_conv(_module, inputs, output):
        captures["patch_conv.input"].append(_target_cpu(inputs[0]))
        captures["patch_conv.output"].append(_target_cpu(output))

    handles.append(first_conv.register_forward_hook(capture_conv))
    handles.append(vision_tower.register_forward_hook(capture_output("vision_tower.output")))
    handles.append(projector.register_forward_hook(capture_output("projector.output")))
    handles.append(
        first_language_layer.input_layernorm.register_forward_hook(
            capture_output("language.layer0.input_layernorm.output")
        )
    )
    handles.append(
        first_language_layer.self_attn.q_proj.register_forward_hook(
            capture_output("language.layer0.q_proj.output")
        )
    )
    handles.append(
        first_language_layer.self_attn.k_proj.register_forward_hook(
            capture_output("language.layer0.k_proj.output")
        )
    )
    handles.append(
        first_language_layer.self_attn.v_proj.register_forward_hook(
            capture_output("language.layer0.v_proj.output")
        )
    )
    handles.append(
        first_language_layer.self_attn.register_forward_hook(
            capture_output("language.layer0.self_attn.output")
        )
    )
    handles.append(
        first_language_layer.self_attn.o_proj.register_forward_hook(
            capture_output("language.layer0.o_proj.output")
        )
    )
    handles.append(
        first_language_layer.post_attention_layernorm.register_forward_pre_hook(
            capture_input("language.layer0.post_attention_layernorm.input")
        )
    )
    handles.append(
        first_language_layer.post_attention_layernorm.register_forward_hook(
            capture_output("language.layer0.post_attention_layernorm.output")
        )
    )
    handles.append(
        first_language_layer.mlp.gate_proj.register_forward_hook(
            capture_output("language.layer0.mlp.gate_proj.output")
        )
    )
    handles.append(
        first_language_layer.mlp.up_proj.register_forward_hook(
            capture_output("language.layer0.mlp.up_proj.output")
        )
    )
    handles.append(
        first_language_layer.mlp.down_proj.register_forward_pre_hook(
            capture_input("language.layer0.mlp.down_proj.input")
        )
    )
    handles.append(
        first_language_layer.mlp.down_proj.register_forward_hook(
            capture_output("language.layer0.mlp.down_proj.output")
        )
    )
    handles.append(
        first_language_layer.mlp.register_forward_hook(capture_output("language.layer0.mlp.output"))
    )
    handles.append(
        first_language_layer.register_forward_hook(capture_output("language.layer0.output"))
    )
    handles.append(lm_head.register_forward_hook(capture_output("lm_head.output")))
    return handles, first_conv_name


def _run(adapter, model, samples, batch_size: int, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, first_conv_name = _register_hooks(model, captures)
    batch = _compose_batch(adapter, samples[:batch_size])
    try:
        seed_everything(seed)
        with torch.inference_mode():
            output = adapter.run_model(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return captures, output, first_conv_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.lerobot_pi0fast",
    )
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--batch-invariant-ops", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model()
    samples = adapter.load_example_inputs(2)
    with set_batch_invariant_mode(args.batch_invariant_ops):
        b1, output_b1, first_conv_name = _run(adapter, model, samples, 1, args.seed)
        b2, output_b2, _ = _run(adapter, model, samples, 2, args.seed)

    boundaries = {}
    for label in b1:
        comparisons = [
            _comparison(reference, candidate)
            for reference, candidate in zip(b1[label], b2[label], strict=False)
        ]
        boundaries[label] = {
            "b1_calls": len(b1[label]),
            "b2_calls": len(b2[label]),
            "all_common_calls_exact": all(item["exact"] for item in comparisons),
            "calls": comparisons,
        }

    token_comparison = _comparison(
        output_b1["action_tokens"][:1].detach().cpu(),
        output_b2["action_tokens"][:1].detach().cpu(),
    )
    action_comparison = _comparison(
        output_b1["actions"][:1].detach().cpu(),
        output_b2["actions"][:1].detach().cpu(),
    )
    report = {
        "model": adapter.name,
        "batch_invariant_ops": args.batch_invariant_ops,
        "composition": "unrelated",
        "batch_sizes": [1, 2],
        "seed": args.seed,
        "first_vision_conv_module": first_conv_name,
        "boundaries": boundaries,
        "raw_action_tokens": token_comparison,
        "decoded_actions": action_comparison,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Trace InternVLA-A1.5's first batch-dependent boundary and flow steps."""

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
from torch.profiler import ProfilerActivity, profile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, seed_everything


def _tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for attribute in ("pooler_output", "last_hidden_state"):
        result = getattr(value, attribute, None)
        if isinstance(result, torch.Tensor):
            return result
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _tensor(item)
            except TypeError:
                continue
    raise TypeError(f"cannot select tensor from {type(value).__name__}")


def _hash(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _target_shape(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if reference.ndim != candidate.ndim:
        return candidate
    slices = tuple(slice(0, size) for size in reference.shape)
    return candidate[slices]


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    candidate_full_shape = list(candidate.shape)
    candidate = _target_shape(reference, candidate)
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_full_shape": candidate_full_shape,
        "dtype": str(reference.dtype),
        "exact": exact,
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }
    if reference.shape != candidate.shape:
        return result
    different = reference != candidate
    locations = torch.nonzero(different, as_tuple=False)
    result["differing_elements"] = int(different.sum().item())
    result["first_differing_index"] = (
        [int(index) for index in locations[0].tolist()] if locations.numel() else None
    )
    if reference.is_floating_point():
        absolute = (reference.float() - candidate.float()).abs()
        result["max_abs_difference"] = float(absolute.max().item())
        result["mean_abs_difference"] = float(absolute.mean().item())
    return result


def _register(model, captures: dict[str, list[torch.Tensor]]):
    core = model.model
    joint = core.qwen3_5_with_expert
    visual = joint.qwen3_5.visual
    language_layer = joint.qwen3_5.language_model.layers[0]
    action_layer = joint.action_expert.layers[0]
    handles = []

    def output_hook(label: str):
        def hook(_module, _inputs, output):
            captures[label].append(_tensor(output).detach().contiguous().cpu())

        return hook

    def input_hook(label: str):
        def hook(_module, inputs):
            captures[label].append(_tensor(inputs).detach().contiguous().cpu())

        return hook

    patch_conv = visual.patch_embed.proj
    handles.append(patch_conv.register_forward_pre_hook(input_hook("vision.patch_conv.input")))
    handles.append(patch_conv.register_forward_hook(output_hook("vision.patch_conv.output")))
    handles.append(
        visual.patch_embed.register_forward_hook(output_hook("vision.patch_embed.output"))
    )
    handles.append(visual.blocks[0].register_forward_hook(output_hook("vision.block0.output")))
    handles.append(visual.register_forward_hook(output_hook("vision.output")))

    handles.append(
        language_layer.input_layernorm.register_forward_hook(
            output_hook("language.layer0.input_norm.output")
        )
    )
    token_mixer = (
        language_layer.linear_attn
        if language_layer.layer_type == "linear_attention"
        else language_layer.self_attn
    )
    handles.append(token_mixer.register_forward_hook(output_hook("language.layer0.mixer.output")))
    if hasattr(token_mixer, "in_proj_qkv"):
        handles.append(
            token_mixer.in_proj_qkv.register_forward_pre_hook(
                input_hook("language.layer0.in_proj_qkv.input")
            )
        )
        handles.append(
            token_mixer.in_proj_qkv.register_forward_hook(
                output_hook("language.layer0.in_proj_qkv.output")
            )
        )
        handles.append(
            token_mixer.conv1d.register_forward_pre_hook(input_hook("language.layer0.conv1d.input"))
        )
        handles.append(
            token_mixer.conv1d.register_forward_hook(output_hook("language.layer0.conv1d.output"))
        )
    handles.append(
        language_layer.post_attention_layernorm.register_forward_hook(
            output_hook("language.layer0.post_norm.output")
        )
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(language_layer.mlp, projection)
        handles.append(
            module.register_forward_pre_hook(input_hook(f"language.layer0.mlp.{projection}.input"))
        )
        handles.append(
            module.register_forward_hook(output_hook(f"language.layer0.mlp.{projection}.output"))
        )
    handles.append(language_layer.register_forward_hook(output_hook("language.layer0.output")))
    for layer_index, layer in enumerate(joint.qwen3_5.language_model.layers[1:], start=1):
        handles.append(
            layer.register_forward_hook(output_hook(f"language.layer{layer_index}.output"))
        )

    handles.append(
        action_layer.input_layernorm.register_forward_hook(
            output_hook("action.layer0.input_norm.output")
        )
    )
    handles.append(action_layer.register_forward_hook(output_hook("action.layer0.output")))
    for layer_index, layer in enumerate(joint.action_expert.layers[1:], start=1):
        handles.append(
            layer.register_forward_hook(output_hook(f"action.layer{layer_index}.output"))
        )
    frontier_layer = joint.action_expert.layers[19]
    handles.append(
        frontier_layer.input_layernorm.register_forward_pre_hook(
            input_hook("action.layer19.input_norm.input")
        )
    )
    handles.append(
        frontier_layer.input_layernorm.register_forward_hook(
            output_hook("action.layer19.input_norm.output")
        )
    )
    frontier_mixer = (
        frontier_layer.linear_attn
        if frontier_layer.layer_type == "linear_attention"
        else frontier_layer.self_attn
    )
    if frontier_layer.layer_type == "full_attention":
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(frontier_mixer, projection)
            handles.append(
                module.register_forward_pre_hook(
                    input_hook(f"action.layer19.attention.{projection}.input")
                )
            )
            handles.append(
                module.register_forward_hook(
                    output_hook(f"action.layer19.attention.{projection}.output")
                )
            )
        for norm in ("q_norm", "k_norm"):
            module = getattr(frontier_mixer, norm)
            handles.append(
                module.register_forward_pre_hook(
                    input_hook(f"action.layer19.attention.{norm}.input")
                )
            )
            handles.append(
                module.register_forward_hook(output_hook(f"action.layer19.attention.{norm}.output"))
            )
    handles.append(frontier_mixer.register_forward_hook(output_hook("action.layer19.mixer.output")))
    handles.append(
        frontier_layer.post_attention_layernorm.register_forward_pre_hook(
            input_hook("action.layer19.post_norm.input")
        )
    )
    handles.append(
        frontier_layer.post_attention_layernorm.register_forward_hook(
            output_hook("action.layer19.post_norm.output")
        )
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(frontier_layer.mlp, projection)
        handles.append(
            module.register_forward_pre_hook(input_hook(f"action.layer19.mlp.{projection}.input"))
        )
        handles.append(
            module.register_forward_hook(output_hook(f"action.layer19.mlp.{projection}.output"))
        )
    handles.append(
        joint.action_expert.norm.register_forward_hook(output_hook("action.final_norm.output"))
    )
    handles.append(
        core.action_out_proj.register_forward_pre_hook(input_hook("flow.velocity.input"))
    )
    handles.append(core.action_out_proj.register_forward_hook(output_hook("flow.velocity.output")))
    return handles, patch_conv


def _run(adapter, model, examples, batch_size: int, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, patch_conv = _register(model, captures)
    from transformers.models.qwen3_5 import modeling_qwen3_5

    frontier_attention = model.model.qwen3_5_with_expert.action_expert.layers[19].self_attn
    original_eager_attention = modeling_qwen3_5.eager_attention_forward

    def traced_eager_attention(module, query, key, value, attention_mask, scaling, **kwargs):
        output = original_eager_attention(
            module, query, key, value, attention_mask, scaling, **kwargs
        )
        if module is frontier_attention:
            expanded_key = modeling_qwen3_5.repeat_kv(key, module.num_key_value_groups)
            expanded_value = modeling_qwen3_5.repeat_kv(value, module.num_key_value_groups)
            scores = torch.matmul(query, expanded_key.transpose(2, 3)) * scaling
            captures["action.layer19.attention.query"].append(query.detach().contiguous().cpu())
            captures["action.layer19.attention.key"].append(
                expanded_key.detach().contiguous().cpu()
            )
            captures["action.layer19.attention.value"].append(
                expanded_value.detach().contiguous().cpu()
            )
            captures["action.layer19.attention.scores.pre_mask"].append(
                scores.detach().contiguous().cpu()
            )
            if attention_mask is not None:
                scores = scores + attention_mask
            captures["action.layer19.attention.scores.masked"].append(
                scores.detach().contiguous().cpu()
            )
            probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            captures["action.layer19.attention.probabilities"].append(
                probabilities.detach().contiguous().cpu()
            )
            value_product = torch.matmul(probabilities, expanded_value)
            captures["action.layer19.attention.value_product"].append(
                value_product.detach().contiguous().cpu()
            )
        return output

    modeling_qwen3_5.eager_attention_forward = traced_eager_attention
    batch = _compose_batch(adapter, examples[:batch_size])
    adapter.reset_model(model)
    try:
        seed_everything(seed)
        with torch.inference_mode():
            output = adapter.run_model(model, batch)
    finally:
        modeling_qwen3_5.eager_attention_forward = original_eager_attention
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu(), patch_conv


def _profile_patch_conv(module, value: torch.Tensor) -> list[str]:
    value = value.to(device=module.weight.device, dtype=module.weight.dtype)
    module(value)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        module(value)
        torch.cuda.synchronize()
    return sorted(
        {
            event.name
            for event in result.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def _profile_matmul(left: torch.Tensor, right: torch.Tensor) -> list[str]:
    left = left.cuda()
    right = right.cuda()
    torch.matmul(left, right)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        torch.matmul(left, right)
        torch.cuda.synchronize()
    return sorted(
        {
            event.name
            for event in result.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.internvla_a15")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--candidate-batch", type=int, default=2)
    parser.add_argument("--batch-invariant-ops", action="store_true")
    parser.add_argument("--profile-patch-conv", action="store_true")
    parser.add_argument("--profile-value-matmul", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(args.candidate_batch))
    with set_batch_invariant_mode(args.batch_invariant_ops):
        b1, output_b1, patch_conv = _run(adapter, model, examples, 1, args.seed)
        candidate, output_candidate, _ = _run(
            adapter, model, examples, args.candidate_batch, args.seed
        )

    boundaries = {}
    ordered_divergences = []
    for label, reference_calls in b1.items():
        comparisons = [
            _comparison(reference, other)
            for reference, other in zip(reference_calls, candidate[label], strict=False)
        ]
        boundaries[label] = {
            "b1_calls": len(reference_calls),
            "candidate_calls": len(candidate[label]),
            "all_common_calls_exact": all(item["exact"] for item in comparisons),
            "calls": comparisons,
        }
        for call, comparison in enumerate(comparisons):
            if not comparison["exact"]:
                ordered_divergences.append({"boundary": label, "call": call})
                break

    report: dict[str, Any] = {
        "model": adapter.name,
        "batch_invariant_ops": args.batch_invariant_ops,
        "composition": "unrelated",
        "batch_sizes": [1, args.candidate_batch],
        "seed": args.seed,
        "boundaries": boundaries,
        "first_non_exact_calls_by_boundary": ordered_divergences,
        "final_action": _comparison(output_b1, output_candidate),
    }
    if args.profile_patch_conv:
        report["patch_conv_profile"] = {
            "operation": "aten::convolution (Conv3d)",
            "b1_input_shape": list(b1["vision.patch_conv.input"][0].shape),
            "candidate_input_shape": list(candidate["vision.patch_conv.input"][0].shape),
            "stock_cuda_kernels_b1": _profile_patch_conv(
                patch_conv, b1["vision.patch_conv.input"][0]
            ),
            "stock_cuda_kernels_candidate": _profile_patch_conv(
                patch_conv, candidate["vision.patch_conv.input"][0]
            ),
        }
    if args.profile_value_matmul:
        label_probabilities = "action.layer19.attention.probabilities"
        label_value = "action.layer19.attention.value"
        report["value_matmul_profile"] = {
            "operation": "aten::bmm through rank-4 torch.matmul",
            "b1_shapes": [
                list(b1[label_probabilities][0].shape),
                list(b1[label_value][0].shape),
            ],
            "candidate_shapes": [
                list(candidate[label_probabilities][0].shape),
                list(candidate[label_value][0].shape),
            ],
            "stock_cuda_kernels_b1": _profile_matmul(
                b1[label_probabilities][0], b1[label_value][0]
            ),
            "stock_cuda_kernels_candidate": _profile_matmul(
                candidate[label_probabilities][0], candidate[label_value][0]
            ),
        }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

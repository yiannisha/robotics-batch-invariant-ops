"""Trace X-VLA's first batch-dependent boundary and flow propagation."""

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
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _tensor(item)
            except TypeError:
                pass
    for attribute in ("last_hidden_state", "pooler_output"):
        item = getattr(value, attribute, None)
        if isinstance(item, torch.Tensor):
            return item
    raise TypeError(f"cannot select tensor from {type(value).__name__}")


def _hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    full_shape = list(candidate.shape)
    if reference.ndim == candidate.ndim:
        candidate = candidate[tuple(slice(0, size) for size in reference.shape)]
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_full_shape": full_shape,
        "dtype": str(reference.dtype),
        "exact": exact,
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }
    if reference.shape == candidate.shape:
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


def _register(policy, captures: dict[str, list[torch.Tensor]]):
    model = policy.model
    vision = model.vlm.vision_tower
    action = model.transformer
    handles = []
    profiled_modules = {}

    def output_hook(label: str):
        def hook(_module, _inputs, output):
            captures[label].append(_tensor(output).detach().contiguous().cpu())

        return hook

    def input_hook(label: str):
        def hook(_module, inputs):
            captures[label].append(_tensor(inputs).detach().contiguous().cpu())

        return hook

    # Register every Florence vision leaf in execution order.  Hooks on the
    # composite attention and block modules expose functional SDPA/residual
    # boundaries which do not have their own leaf module.
    for name, module in vision.named_modules():
        if not name:
            continue
        label = f"vision.{name}"
        if not any(module.children()):
            handles.append(module.register_forward_hook(output_hook(f"{label}.output")))
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(module.register_forward_pre_hook(input_hook(f"{label}.input")))
                profiled_modules[f"{label}.output"] = module
        elif type(module).__name__ in {
            "Florence2VisionConvEmbed",
            "Florence2VisionWindowAttention",
            "Florence2VisionChannelAttention",
            "Florence2VisionSpatialBlock",
            "Florence2VisionChannelBlock",
            "Florence2VisionBlock",
        }:
            handles.append(module.register_forward_hook(output_hook(f"{label}.output")))

    projector = model.vlm.multi_modal_projector
    for name, module in projector.named_modules():
        if not name:
            continue
        label = f"vision.multimodal_projector.{name}"
        if not any(module.children()):
            handles.append(module.register_forward_hook(output_hook(f"{label}.output")))
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(module.register_forward_pre_hook(input_hook(f"{label}.input")))
                profiled_modules[f"{label}.output"] = module

    handles.extend(
        [
            model.vlm.multi_modal_projector.register_forward_hook(
                output_hook("vision.multimodal_projector")
            ),
            model.vlm.language_model.encoder.layers[0].register_forward_hook(
                output_hook("language.block0.output")
            ),
            model.vlm.language_model.encoder.layers[-1].register_forward_hook(
                output_hook("language.block11.output")
            ),
            action.vlm_proj.register_forward_hook(output_hook("flow.vlm_projection")),
            action.aux_visual_proj.register_forward_hook(output_hook("flow.aux_visual_projection")),
            action.action_encoder.register_forward_pre_hook(
                input_hook("flow.action_encoder.input")
            ),
            action.action_encoder.register_forward_hook(output_hook("flow.action_encoder.output")),
            action.blocks[0].attn.qkv.register_forward_hook(
                output_hook("flow.block0.attention.qkv")
            ),
            action.blocks[0].attn.register_forward_hook(
                output_hook("flow.block0.attention.output")
            ),
            action.blocks[0].register_forward_hook(output_hook("flow.block0.output")),
            action.blocks[-1].register_forward_hook(output_hook("flow.block23.output")),
            action.action_decoder.register_forward_hook(output_hook("flow.velocity")),
        ]
    )
    return handles, profiled_modules


def _run(adapter, model, examples, batch_size: int, *, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, profiled_modules = _register(model, captures)
    batch = _compose_batch(adapter, examples[:batch_size])
    for key, value in batch.items():
        captures[f"preprocess.{key}"].append(value.detach().cpu())
    seed_everything(seed)
    try:
        with set_batch_invariant_mode(invariant), torch.inference_mode():
            output = adapter.run_model(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu(), profiled_modules


def _compare_captures(reference, candidate, output_reference, output_candidate):
    records = []
    for label, reference_values in reference.items():
        candidate_values = candidate[label]
        if len(reference_values) != len(candidate_values):
            records.append(
                {
                    "label": label,
                    "error": "capture count differs",
                    "reference_count": len(reference_values),
                    "candidate_count": len(candidate_values),
                }
            )
            continue
        for index, (left, right) in enumerate(zip(reference_values, candidate_values)):
            record = {"label": label, "call": index}
            record.update(_comparison(left, right))
            records.append(record)
    final = {"label": "final.actions", "call": 0}
    final.update(_comparison(output_reference, output_candidate))
    records.append(final)
    return records


def _profile_conv(module, values: list[torch.Tensor]):
    reports = []
    for value in values:
        value = value.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.inference_mode():
            module(value)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
                module(value)
                torch.cuda.synchronize()
        reports.append(
            {
                "input_shape": list(value.shape),
                "weight_shape": list(module.weight.shape),
                "stride": list(module.stride),
                "padding": list(module.padding),
                "aten_operators": sorted(
                    {
                        event.name
                        for event in result.events()
                        if event.device_type == torch.autograd.DeviceType.CPU
                        and event.name.startswith("aten::")
                    }
                ),
                "cuda_kernels": sorted(
                    {
                        event.name
                        for event in result.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                    }
                ),
            }
        )
    return reports


def _profile_linear(module, values: list[torch.Tensor]):
    reports = []
    for value in values:
        value = value.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.inference_mode():
            module(value)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
                module(value)
                torch.cuda.synchronize()
        reports.append(
            {
                "input_shape": list(value.shape),
                "weight_shape": list(module.weight.shape),
                "bias": module.bias is not None,
                "aten_operators": sorted(
                    {
                        event.name
                        for event in result.events()
                        if event.device_type == torch.autograd.DeviceType.CPU
                        and event.name.startswith("aten::")
                    }
                ),
                "cuda_kernels": sorted(
                    {
                        event.name
                        for event in result.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                    }
                ),
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.lerobot_xvla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    stock_b1, stock_output_b1, profiled_modules = _run(
        adapter, model, examples, 1, invariant=False, seed=args.seed
    )
    stock_b2, stock_output_b2, _ = _run(
        adapter, model, examples, 2, invariant=False, seed=args.seed
    )

    # Repair the first convolution boundary alone, then trace again to expose
    # any independent downstream batch-shaped arithmetic choice.
    from batch_invariant_ops.batch_invariant_ops import convolution_batch_invariant

    conv_only_library = torch.library.Library("aten", "IMPL")
    conv_only_library.impl("aten::convolution", convolution_batch_invariant, "CUDA")
    try:
        conv_b1, conv_output_b1, _ = _run(
            adapter, model, examples, 1, invariant=False, seed=args.seed
        )
        conv_b2, conv_output_b2, _ = _run(
            adapter, model, examples, 2, invariant=False, seed=args.seed
        )
    finally:
        conv_only_library._destroy()

    from batch_invariant_ops.batch_invariant_ops import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        mm_batch_invariant,
    )

    conv_gemm_library = torch.library.Library("aten", "IMPL")
    for operator, implementation in (
        ("aten::convolution", convolution_batch_invariant),
        ("aten::mm", mm_batch_invariant),
        ("aten::addmm", addmm_batch_invariant),
        ("aten::bmm", bmm_batch_invariant),
    ):
        conv_gemm_library.impl(operator, implementation, "CUDA")
    try:
        conv_gemm_b1, conv_gemm_output_b1, _ = _run(
            adapter, model, examples, 1, invariant=False, seed=args.seed
        )
        conv_gemm_b2, conv_gemm_output_b2, _ = _run(
            adapter, model, examples, 2, invariant=False, seed=args.seed
        )
    finally:
        conv_gemm_library._destroy()
    fixed_b1, fixed_output_b1, _ = _run(adapter, model, examples, 1, invariant=True, seed=args.seed)
    fixed_b2, fixed_output_b2, _ = _run(adapter, model, examples, 2, invariant=True, seed=args.seed)
    stock = _compare_captures(stock_b1, stock_b2, stock_output_b1, stock_output_b2)
    conv_only = _compare_captures(conv_b1, conv_b2, conv_output_b1, conv_output_b2)
    conv_gemm = _compare_captures(
        conv_gemm_b1, conv_gemm_b2, conv_gemm_output_b1, conv_gemm_output_b2
    )
    fixed = _compare_captures(fixed_b1, fixed_b2, fixed_output_b1, fixed_output_b2)
    first_stock_difference = next(record for record in stock if not record.get("exact", True))
    first_after_convolution_difference = next(
        record for record in conv_only if not record.get("exact", True)
    )
    frontier_label = first_stock_difference["label"]
    frontier = profiled_modules.get(frontier_label)
    frontier_input_label = frontier_label.removesuffix(".output") + ".input"
    profile_report = None
    if isinstance(frontier, torch.nn.Conv2d) and frontier_input_label in stock_b1:
        profile_report = _profile_conv(
            frontier,
            [stock_b1[frontier_input_label][0], stock_b2[frontier_input_label][0]],
        )
    second_label = first_after_convolution_difference["label"]
    second_frontier = profiled_modules.get(second_label)
    second_input_label = second_label.removesuffix(".output") + ".input"
    second_profile_report = None
    if isinstance(second_frontier, torch.nn.Conv2d) and second_input_label in conv_b1:
        second_profile_report = _profile_conv(
            second_frontier,
            [conv_b1[second_input_label][0], conv_b2[second_input_label][0]],
        )
    elif isinstance(second_frontier, torch.nn.Linear) and second_input_label in conv_b1:
        second_profile_report = _profile_linear(
            second_frontier,
            [conv_b1[second_input_label][0], conv_b2[second_input_label][0]],
        )
    report = {
        "model": adapter.name,
        "comparison": "B=1 target versus first sample at unrelated B=2",
        "first_stock_difference": first_stock_difference,
        "first_after_convolution_difference": first_after_convolution_difference,
        "stock_trace": stock,
        "convolution_only_trace": conv_only,
        "convolution_gemm_only_trace": conv_gemm,
        "convolution_gemm_only_all_exact": all(record.get("exact", False) for record in conv_gemm),
        "fixed_trace": fixed,
        "first_divergent_operator_profile": profile_report,
        "second_divergent_operator_profile": second_profile_report,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

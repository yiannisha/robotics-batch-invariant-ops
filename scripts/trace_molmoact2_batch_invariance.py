"""Trace MolmoAct2's first batch-dependent boundary and flow propagation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
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
    for attribute in ("last_hidden_state", "image_hidden_states"):
        item = getattr(value, attribute, None)
        if isinstance(item, torch.Tensor):
            return item
    raise TypeError(f"cannot select tensor from {type(value).__name__}")


def _hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
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


def _register(model, captures: dict[str, list[torch.Tensor]]):
    core = model.model
    vision = core.vision_backbone
    expert = core.action_expert
    handles = []

    def output_hook(label: str):
        def hook(_module, _inputs, output):
            captures[label].append(_tensor(output).detach().contiguous().cpu())

        return hook

    def input_hook(label: str):
        def hook(_module, inputs):
            captures[label].append(_tensor(inputs).detach().contiguous().cpu())

        return hook

    patch = vision.image_vit.patch_embedding
    handles.append(patch.register_forward_pre_hook(input_hook("vision.patch_embedding.input")))
    handles.append(patch.register_forward_hook(output_hook("vision.patch_embedding.output")))
    for index, block in enumerate(vision.image_vit.transformer.resblocks):
        handles.append(block.register_forward_hook(output_hook(f"vision.block{index}.output")))
    handles.append(
        vision.image_pooling_2d.register_forward_hook(output_hook("vision.pooling.output"))
    )
    projector = vision.image_projector
    handles.append(projector.register_forward_pre_hook(input_hook("vision.projector.input")))
    handles.append(projector.w1.register_forward_hook(output_hook("vision.projector.w1.output")))
    handles.append(projector.act.register_forward_hook(output_hook("vision.projector.act.output")))
    handles.append(projector.w3.register_forward_hook(output_hook("vision.projector.w3.output")))
    handles.append(projector.w2.register_forward_pre_hook(input_hook("vision.projector.w2.input")))
    handles.append(projector.w2.register_forward_hook(output_hook("vision.projector.w2.output")))
    handles.append(
        vision.image_projector.register_forward_hook(output_hook("vision.projector.output"))
    )
    handles.append(vision.register_forward_hook(output_hook("vision.backbone.output")))
    for index, block in enumerate(core.transformer.blocks):
        handles.append(block.register_forward_hook(output_hook(f"language.block{index}.output")))

    handles.append(
        expert.action_embed.register_forward_pre_hook(input_hook("flow.action_embed.input"))
    )
    handles.append(
        expert.action_embed.register_forward_hook(output_hook("flow.action_embed.output"))
    )
    handles.append(expert.blocks[0].register_forward_hook(output_hook("action.block0.output")))
    handles.append(expert.blocks[-1].register_forward_hook(output_hook("action.block35.output")))
    handles.append(expert.final_layer.register_forward_hook(output_hook("flow.velocity.output")))
    return handles, projector.w2


def _run(adapter, model, examples, batch_size: int, *, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, frontier = _register(model, captures)
    batch = _compose_batch(adapter, examples[:batch_size])
    captures["preprocess.input_ids"].append(batch["input_ids"].detach().cpu())
    captures["preprocess.pixel_values"].append(batch["pixel_values"].detach().cpu())
    adapter.reset_model(model)
    seed_everything(seed)
    context = set_batch_invariant_mode() if invariant else torch.no_grad()
    try:
        with context, torch.inference_mode():
            output = adapter.run_model(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu(), frontier


def _check_upstream_batch_bug(adapter, model, examples, seed: int):
    core = model.model
    corrected_builder = core._build_native_attention_bias
    official_builder = types.MethodType(core.__class__._build_native_attention_bias, core)

    def run(batch_size: int):
        adapter.reset_model(model)
        seed_everything(seed)
        batch = _compose_batch(adapter, examples[:batch_size])
        with torch.inference_mode():
            return adapter.run_model(model, batch).detach().contiguous().cpu()

    core._build_native_attention_bias = official_builder
    official_b1 = run(1)
    official_b2_error = None
    try:
        run(2)
    except RuntimeError as error:
        official_b2_error = str(error)
    finally:
        core._build_native_attention_bias = corrected_builder
    corrected_b1 = run(1)
    return {
        "official_vs_corrected_b1": _comparison(official_b1, corrected_b1),
        "official_b2_error": official_b2_error,
    }


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


def _profile_linear(module, values: list[torch.Tensor], dtype: torch.dtype):
    reports = []
    for value in values:
        value = value.to(device=module.weight.device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            module(value)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
                module(value)
                torch.cuda.synchronize()
        reports.append(
            {
                "input_shape": list(value.shape),
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
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.molmoact2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))

    compatibility = _check_upstream_batch_bug(adapter, model, examples, args.seed)
    stock_b1, stock_output_b1, frontier = _run(
        adapter, model, examples, 1, invariant=False, seed=args.seed
    )
    stock_b2, stock_output_b2, _ = _run(
        adapter, model, examples, 2, invariant=False, seed=args.seed
    )
    profile_report = _profile_linear(
        frontier,
        [
            stock_b1["vision.projector.w2.input"][0],
            stock_b2["vision.projector.w2.input"][0],
        ],
        adapter.dtype,
    )

    fixed_b1, fixed_output_b1, _ = _run(adapter, model, examples, 1, invariant=True, seed=args.seed)
    fixed_b2, fixed_output_b2, _ = _run(adapter, model, examples, 2, invariant=True, seed=args.seed)

    stock = _compare_captures(stock_b1, stock_b2, stock_output_b1, stock_output_b2)
    fixed = _compare_captures(fixed_b1, fixed_b2, fixed_output_b1, fixed_output_b2)
    first_stock_difference = next(
        (record for record in stock if record.get("exact") is False), None
    )
    report = {
        "model": adapter.name,
        "comparison": "B=1 target versus first sample at B=2",
        "upstream_batch_compatibility": compatibility,
        "first_stock_difference": first_stock_difference,
        "stock_trace": stock,
        "fixed_trace": fixed,
        "first_divergent_linear_profile": profile_report,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

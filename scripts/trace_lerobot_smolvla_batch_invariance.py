"""Trace SmolVLA's first batch-dependent boundary and flow propagation."""

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
    item = getattr(value, "last_hidden_state", None)
    if isinstance(item, torch.Tensor):
        return item
    raise TypeError(f"cannot select tensor from {type(value).__name__}")


def _hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    full_shape = list(candidate.shape)
    if reference.ndim == candidate.ndim and all(
        left <= right for left, right in zip(reference.shape, candidate.shape)
    ):
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
    handles = []
    profiled_modules = {}

    def capture(label: str, value: Any) -> None:
        captures[label].append(_tensor(value).detach().contiguous().cpu())

    def output_hook(label: str):
        def hook(_module, _inputs, output):
            capture(label, output)

        return hook

    def input_hook(label: str):
        def hook(_module, inputs):
            capture(label, inputs)

        return hook

    def add(label: str, module: torch.nn.Module, *, inputs: bool = False) -> None:
        if inputs:
            handles.append(module.register_forward_pre_hook(input_hook(f"{label}.input")))
        handles.append(module.register_forward_hook(output_hook(f"{label}.output")))
        profiled_modules[f"{label}.output"] = module

    combined = policy.model.vlm_with_expert
    vlm = combined.get_vlm_model()
    vision = vlm.vision_model
    add("vision.patch_embedding", vision.embeddings.patch_embedding, inputs=True)
    for index, block in enumerate(vision.encoder.layers):
        if index == 0:
            add("vision.block0.attention.q_proj", block.self_attn.q_proj, inputs=True)
            add("vision.block0.attention.k_proj", block.self_attn.k_proj)
            add("vision.block0.attention.v_proj", block.self_attn.v_proj)
            add("vision.block0.attention.out_proj", block.self_attn.out_proj)
            add("vision.block0.mlp.fc1", block.mlp.fc1, inputs=True)
            add("vision.block0.mlp.fc2", block.mlp.fc2, inputs=True)
        add(f"vision.block{index}", block)
    add("vision.connector", vlm.connector.modality_projection.proj, inputs=True)

    text0 = vlm.text_model.layers[0]
    add("vlm.block0.attention.q_proj", text0.self_attn.q_proj, inputs=True)
    add("vlm.block0.attention.o_proj", text0.self_attn.o_proj, inputs=True)
    add("vlm.block0.mlp.gate_proj", text0.mlp.gate_proj, inputs=True)
    add("vlm.block0.mlp.down_proj", text0.mlp.down_proj, inputs=True)

    expert0 = combined.lm_expert.layers[0]
    add("expert.block0.attention.q_proj", expert0.self_attn.q_proj, inputs=True)
    add("expert.block0.attention.o_proj", expert0.self_attn.o_proj, inputs=True)
    add("expert.block0.mlp.gate_proj", expert0.mlp.gate_proj, inputs=True)
    add("expert.block0.mlp.down_proj", expert0.mlp.down_proj, inputs=True)

    model = policy.model
    add("state.projection", model.state_proj, inputs=True)
    add("action.input_projection", model.action_in_proj, inputs=True)
    add("action.time_projection_in", model.action_time_mlp_in, inputs=True)
    add("action.time_projection_out", model.action_time_mlp_out)
    add("action.output_projection", model.action_out_proj, inputs=True)
    return handles, profiled_modules, capture


def _run(adapter, model, examples, batch_size: int, *, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, modules, capture = _register(model, captures)
    adapter.trace_callback = capture
    batch = _compose_batch(adapter, examples[:batch_size])
    seed_everything(seed)
    try:
        with set_batch_invariant_mode(invariant), torch.inference_mode():
            output = adapter.run_model(model, batch)
    finally:
        adapter.trace_callback = None
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu(), modules


def _compare(reference, candidate, reference_output, candidate_output):
    records = []
    for label, reference_values in reference.items():
        candidate_values = candidate.get(label, [])
        if len(reference_values) != len(candidate_values):
            records.append(
                {
                    "label": label,
                    "exact": False,
                    "error": "capture count differs",
                    "reference_count": len(reference_values),
                    "candidate_count": len(candidate_values),
                }
            )
            continue
        for call, (left, right) in enumerate(zip(reference_values, candidate_values)):
            record = {"label": label, "call": call}
            record.update(_comparison(left, right))
            records.append(record)
    final = {"label": "final.actions", "call": 0}
    final.update(_comparison(reference_output, candidate_output))
    records.append(final)
    return records


def _first_difference(records):
    return next((record for record in records if not record.get("exact", False)), None)


def _profile_module(module: torch.nn.Module, values: list[torch.Tensor]):
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
                "module": type(module).__name__,
                "input_shape": list(value.shape),
                "weight_shape": list(module.weight.shape),
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


def _run_stage(adapter, model, examples, implementations, seed):
    library = torch.library.Library("aten", "IMPL")
    for operator, implementation in implementations:
        library.impl(operator, implementation, "CUDA")
    try:
        b1, output_b1, _ = _run(adapter, model, examples, 1, invariant=False, seed=seed)
        b2, output_b2, _ = _run(adapter, model, examples, 2, invariant=False, seed=seed)
    finally:
        library._destroy()
    return _compare(b1, b2, output_b1, output_b2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.lerobot_smolvla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    stock_b1, stock_output_b1, modules = _run(
        adapter, model, examples, 1, invariant=False, seed=args.seed
    )
    stock_b2, stock_output_b2, _ = _run(
        adapter, model, examples, 2, invariant=False, seed=args.seed
    )
    stock = _compare(stock_b1, stock_b2, stock_output_b1, stock_output_b2)

    from batch_invariant_ops.batch_invariant_ops import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        convolution_batch_invariant,
        mm_batch_invariant,
    )

    convolution_only = _run_stage(
        adapter,
        model,
        examples,
        (("aten::convolution", convolution_batch_invariant),),
        args.seed,
    )
    mm_addmm_only = _run_stage(
        adapter,
        model,
        examples,
        (
            ("aten::mm", mm_batch_invariant),
            ("aten::addmm", addmm_batch_invariant),
        ),
        args.seed,
    )
    mm_only = _run_stage(
        adapter,
        model,
        examples,
        (("aten::mm", mm_batch_invariant),),
        args.seed,
    )
    conv_gemm_implementations = (
        ("aten::convolution", convolution_batch_invariant),
        ("aten::mm", mm_batch_invariant),
        ("aten::addmm", addmm_batch_invariant),
        ("aten::bmm", bmm_batch_invariant),
    )
    convolution_gemm = _run_stage(adapter, model, examples, conv_gemm_implementations, args.seed)
    fixed_b1, fixed_output_b1, _ = _run(adapter, model, examples, 1, invariant=True, seed=args.seed)
    fixed_b2, fixed_output_b2, _ = _run(adapter, model, examples, 2, invariant=True, seed=args.seed)
    fixed = _compare(fixed_b1, fixed_b2, fixed_output_b1, fixed_output_b2)

    first_stock = _first_difference(stock)
    profile_report = None
    if first_stock is not None:
        module = modules.get(first_stock["label"])
        input_label = first_stock["label"].removesuffix(".output") + ".input"
        if module is not None and input_label in stock_b1:
            profile_report = _profile_module(
                module, [stock_b1[input_label][0], stock_b2[input_label][0]]
            )
    second_difference = _first_difference(mm_only)
    second_profile_report = None
    if second_difference is not None:
        module = modules.get(second_difference["label"])
        input_label = second_difference["label"].removesuffix(".output") + ".input"
        if module is not None and input_label in stock_b1:
            second_profile_report = _profile_module(
                module, [stock_b1[input_label][0], stock_b2[input_label][0]]
            )

    report = {
        "model": adapter.name,
        "comparison": "B=1 target versus first sample at unrelated B=2",
        "first_stock_difference": first_stock,
        "first_after_convolution_difference": _first_difference(convolution_only),
        "first_after_mm_difference": _first_difference(mm_only),
        "first_after_mm_addmm_difference": _first_difference(mm_addmm_only),
        "first_after_convolution_gemm_difference": _first_difference(convolution_gemm),
        "stock_trace": stock,
        "convolution_only_trace": convolution_only,
        "mm_only_trace": mm_only,
        "mm_addmm_only_trace": mm_addmm_only,
        "convolution_gemm_trace": convolution_gemm,
        "fixed_trace": fixed,
        "fixed_all_exact": all(record.get("exact", False) for record in fixed),
        "first_divergent_operator_profile": profile_report,
        "second_divergent_operator_profile": second_profile_report,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Trace the first batch-dependent boundary in official DexVLA PyTorch code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping
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
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _tensor(item)
            except TypeError:
                pass
    hidden = getattr(value, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError(f"cannot select tensor from {type(value).__name__}")


def _hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    candidate_shape = list(candidate.shape)
    if reference.ndim == candidate.ndim and all(
        left <= right for left, right in zip(reference.shape, candidate.shape)
    ):
        candidate = candidate[tuple(slice(0, size) for size in reference.shape)]
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_full_shape": candidate_shape,
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
    head = model.policy_head
    handles = []
    modules = {}

    def capture(label: str, value: Any) -> None:
        captures[label].append(_tensor(value).detach().contiguous().cpu())

    def add(label: str, module: torch.nn.Module) -> None:
        handles.append(
            module.register_forward_pre_hook(
                lambda _module, inputs, kwargs, label=label: capture(
                    f"{label}.input", (inputs, kwargs)
                ),
                with_kwargs=True,
            )
        )
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, label=label: capture(f"{label}.output", output)
            )
        )
        modules[f"{label}.output"] = module

    add("vision.patch", model.visual.patch_embed.proj)
    add("vision.block0.qkv", model.visual.blocks[0].attn.qkv)
    add("vision.block0.attention", model.visual.blocks[0].attn)
    add("vision.block0", model.visual.blocks[0])
    add("vision.block1", model.visual.blocks[1])
    add("vision.block31", model.visual.blocks[-1])
    add("vision.merger.fc1", model.visual.merger.mlp[0])
    add("vision.merger.fc2", model.visual.merger.mlp[2])
    add("language.block0.q", model.model.layers[0].self_attn.q_proj)
    add("language.block0.attention", model.model.layers[0].self_attn)
    add("language.block0", model.model.layers[0])
    add("language.block27", model.model.layers[-1])
    add("film.input_projection", model.input_action_proj.mlps[0])
    add("film.reasoning_projection", model.reasoning_action_proj.mlps[0])
    add("film.scale", model.reasoning_film.scale_fc)
    add("policy.combine", head.combine[0])
    add("policy.action_embedding", head.x_embedder)
    add("policy.block0.qkv", head.blocks[0].attn.qkv)
    add("policy.block0.attention", head.blocks[0].attn)
    add("policy.block0", head.blocks[0])
    add("policy.block31", head.blocks[-1])
    add("policy.output", head.final_layer.linear)
    return handles, modules, capture


def _run(adapter, policy, examples, batch_size: int, *, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, modules, capture = _register(policy, captures)
    adapter.trace_callback = capture
    batch = _compose_batch(adapter, examples[:batch_size])
    seed_everything(seed)
    try:
        with set_batch_invariant_mode(invariant), torch.inference_mode():
            output = adapter.run_model(policy, batch)
    finally:
        adapter.trace_callback = None
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu(), modules


def _compare(reference, candidate, reference_output, candidate_output):
    records = []
    for label, left_values in reference.items():
        right_values = candidate.get(label, [])
        if len(right_values) < len(left_values):
            records.append(
                {
                    "label": label,
                    "exact": False,
                    "error": "capture count differs",
                    "reference_count": len(left_values),
                    "candidate_count": len(right_values),
                }
            )
            continue
        for call, (left, right) in enumerate(zip(left_values, right_values)):
            record = {"label": label, "call": call}
            record.update(_comparison(left, right))
            records.append(record)
    final = {"label": "final.actions", "call": 0}
    final.update(_comparison(reference_output, candidate_output))
    records.append(final)
    return records


def _first_difference(records):
    return next((record for record in records if not record.get("exact", False)), None)


def _run_stage(adapter, policy, examples, implementations, seed):
    library = torch.library.Library("aten", "IMPL")
    for operator, implementation in implementations:
        library.impl(operator, implementation, "CUDA")
    try:
        b1, output_b1, _ = _run(adapter, policy, examples, 1, invariant=False, seed=seed)
        b2, output_b2, _ = _run(adapter, policy, examples, 2, invariant=False, seed=seed)
    finally:
        library._destroy()
    return _compare(b1, b2, output_b1, output_b2)


def _profile_module(module: torch.nn.Module, values: list[torch.Tensor]):
    reports = []
    parameter = next(module.parameters())
    for value in values:
        value = value.to(device=parameter.device, dtype=parameter.dtype)
        with torch.inference_mode():
            module(value)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as run:
                module(value)
                torch.cuda.synchronize()
        reports.append(
            {
                "module": type(module).__name__,
                "input_shape": list(value.shape),
                "weight_shape": (list(module.weight.shape) if hasattr(module, "weight") else None),
                "aten_operators": sorted(
                    {
                        event.name
                        for event in run.events()
                        if event.device_type == torch.autograd.DeviceType.CPU
                        and event.name.startswith("aten::")
                    }
                ),
                "cuda_kernels": sorted(
                    {
                        event.name
                        for event in run.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                    }
                ),
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.dexvla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    stock_b1, stock_output_b1, modules = _run(
        adapter, policy, examples, 1, invariant=False, seed=args.seed
    )
    stock_b2, stock_output_b2, _ = _run(
        adapter, policy, examples, 2, invariant=False, seed=args.seed
    )
    stock = _compare(stock_b1, stock_b2, stock_output_b1, stock_output_b2)

    from batch_invariant_ops.batch_invariant_ops import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        convolution_batch_invariant,
        mm_batch_invariant,
        scaled_dot_product_attention_batch_invariant,
    )

    stages = {}
    implementations = []
    for name, operator, implementation in (
        ("convolution", "aten::convolution", convolution_batch_invariant),
        ("gemm", "aten::mm", mm_batch_invariant),
        ("addmm", "aten::addmm", addmm_batch_invariant),
        (
            "attention",
            "aten::scaled_dot_product_attention",
            scaled_dot_product_attention_batch_invariant,
        ),
        ("bmm", "aten::bmm", bmm_batch_invariant),
    ):
        implementations.append((operator, implementation))
        stages[name] = _run_stage(adapter, policy, examples, tuple(implementations), args.seed)

    attention = (
        "aten::scaled_dot_product_attention",
        scaled_dot_product_attention_batch_invariant,
    )
    isolated_stages = {
        "attention_only": _run_stage(adapter, policy, examples, (attention,), args.seed),
        "attention_mm": _run_stage(
            adapter,
            policy,
            examples,
            (attention, ("aten::mm", mm_batch_invariant)),
            args.seed,
        ),
        "attention_addmm": _run_stage(
            adapter,
            policy,
            examples,
            (attention, ("aten::addmm", addmm_batch_invariant)),
            args.seed,
        ),
        "attention_gemm": _run_stage(
            adapter,
            policy,
            examples,
            (
                attention,
                ("aten::mm", mm_batch_invariant),
                ("aten::addmm", addmm_batch_invariant),
            ),
            args.seed,
        ),
        "attention_gemm_bmm": _run_stage(
            adapter,
            policy,
            examples,
            (
                attention,
                ("aten::mm", mm_batch_invariant),
                ("aten::addmm", addmm_batch_invariant),
                ("aten::bmm", bmm_batch_invariant),
            ),
            args.seed,
        ),
    }

    fixed_b1, fixed_output_b1, _ = _run(
        adapter, policy, examples, 1, invariant=True, seed=args.seed
    )
    fixed_b2, fixed_output_b2, _ = _run(
        adapter, policy, examples, 2, invariant=True, seed=args.seed
    )
    fixed = _compare(fixed_b1, fixed_b2, fixed_output_b1, fixed_output_b2)

    first_stock = _first_difference(stock)
    profiler = None
    if (
        first_stock is not None
        and first_stock["label"] in modules
        and isinstance(
            modules[first_stock["label"]],
            (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d),
        )
    ):
        input_label = first_stock["label"].removesuffix(".output") + ".input"
        if input_label in stock_b1:
            profiler = _profile_module(
                modules[first_stock["label"]],
                [stock_b1[input_label][0], stock_b2[input_label][0]],
            )

    report = {
        "model": adapter.name,
        "comparison": "B=1 target versus first sample at unrelated B=2",
        "first_stock_difference": first_stock,
        "first_after_each_cumulative_patch": {
            name: _first_difference(records) for name, records in stages.items()
        },
        "first_after_isolated_patch_sets": {
            name: _first_difference(records) for name, records in isolated_stages.items()
        },
        "first_fixed_difference": _first_difference(fixed),
        "stock_trace": stock,
        "cumulative_patch_traces": stages,
        "isolated_patch_traces": isolated_stages,
        "fixed_trace": fixed,
        "first_stock_profiler": profiler,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

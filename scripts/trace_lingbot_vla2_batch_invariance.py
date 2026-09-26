"""Trace batch-dependent boundaries in official PyTorch LingBot-VLA 2.0."""

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
    record: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_full_shape": candidate_shape,
        "dtype": str(reference.dtype),
        "exact": exact,
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }
    if reference.shape == candidate.shape:
        difference = reference != candidate
        locations = torch.nonzero(difference, as_tuple=False)
        record["differing_elements"] = int(difference.sum().item())
        record["first_differing_index"] = (
            [int(index) for index in locations[0].tolist()] if locations.numel() else None
        )
        if reference.is_floating_point():
            absolute = (reference.float() - candidate.float()).abs()
            record["max_abs_difference"] = float(absolute.max().item())
            record["mean_abs_difference"] = float(absolute.mean().item())
    return record


def _register(policy, captures: dict[str, list[torch.Tensor]]):
    vision = policy.qwenvl_with_expert.qwenvl.visual
    language = policy.qwenvl_with_expert.qwenvl.model.language_model
    expert = policy.qwenvl_with_expert.qwen_expert.model
    handles = []

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

    add("vision.patch_conv3d", vision.patch_embed.proj)
    add("vision.patch", vision.patch_embed)
    add("vision.block0.qkv", vision.blocks[0].attn.qkv)
    add("vision.block0.attention", vision.blocks[0].attn)
    add("vision.block0", vision.blocks[0])
    add("vision.block1", vision.blocks[1])
    add("vision.block26", vision.blocks[-1])
    add("vision.merger.fc1", vision.merger.linear_fc1)
    add("vision.merger.fc2", vision.merger.linear_fc2)
    add("language.block0.q", language.layers[0].self_attn.q_proj)
    add("language.block0", language.layers[0])
    add("language.block35", language.layers[-1])
    add("flow.state_projection", policy.state_proj)
    add("flow.action_projection", policy.action_in_proj)
    add("expert.block0.q", expert.layers[0].self_attn.q_proj)
    add("expert.block0", expert.layers[0])
    add("expert.block35", expert.layers[-1])
    add("flow.output", policy.action_out_proj)
    return handles


def _run(adapter, policy, examples, batch_size: int, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = _register(policy, captures)
    batch = _compose_batch(adapter, examples[:batch_size])
    seed_everything(seed)
    adapter.reset_model(policy)
    try:
        with set_batch_invariant_mode(invariant), torch.inference_mode():
            output = adapter.run_model(policy, batch)
    finally:
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu()


def _compare(reference, candidate, reference_output, candidate_output):
    records = []
    for label, left_values in reference.items():
        right_values = candidate.get(label, [])
        if len(left_values) != len(right_values):
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


def _run_output(adapter, policy, examples, batch_size: int, seed: int):
    batch = _compose_batch(adapter, examples[:batch_size])
    seed_everything(seed)
    adapter.reset_model(policy)
    with torch.inference_mode():
        return adapter.run_model(policy, batch).detach().contiguous().cpu()


def _run_operator_stage(adapter, policy, examples, implementations, seed):
    library = torch.library.Library("aten", "IMPL")
    for operator, implementation in implementations:
        library.impl(operator, implementation, "CUDA")
    try:
        reference = _run_output(adapter, policy, examples, 1, seed)
        candidate = _run_output(adapter, policy, examples, 2, seed)
    finally:
        library._destroy()
    return _comparison(reference, candidate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.lingbot_vla2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    report = {}
    for mode_name, invariant in (("stock", False), ("fixed", True)):
        b1, output_b1 = _run(adapter, policy, examples, 1, invariant, args.seed)
        b2, output_b2 = _run(adapter, policy, examples, 2, invariant, args.seed)
        records = _compare(b1, b2, output_b1, output_b2)
        report[mode_name] = {
            "first_difference": _first_difference(records),
            "records": records,
        }

    from batch_invariant_ops.batch_invariant_ops import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        mm_batch_invariant,
    )

    addmm = ("aten::addmm", addmm_batch_invariant)
    mm = ("aten::mm", mm_batch_invariant)
    bmm = ("aten::bmm", bmm_batch_invariant)
    report["operator_stages"] = {
        "addmm_only": _run_operator_stage(adapter, policy, examples, (addmm,), args.seed),
        "mm_only": _run_operator_stage(adapter, policy, examples, (mm,), args.seed),
        "addmm_mm": _run_operator_stage(adapter, policy, examples, (addmm, mm), args.seed),
        "addmm_mm_bmm": _run_operator_stage(adapter, policy, examples, (addmm, mm, bmm), args.seed),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stock": report["stock"]["first_difference"],
                "fixed": report["fixed"]["first_difference"],
                "operator_stages": report["operator_stages"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

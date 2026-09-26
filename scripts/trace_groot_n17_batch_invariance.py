"""Trace GR00T N1.7's first batch-dependent boundary and flow propagation."""

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
    for attribute in ("last_hidden_state", "backbone_features"):
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
            [int(index) for index in locations[0].tolist()]
            if locations.numel()
            else None
        )
        if reference.is_floating_point():
            absolute = (reference.float() - candidate.float()).abs()
            result["max_abs_difference"] = float(absolute.max().item())
            result["mean_abs_difference"] = float(absolute.mean().item())
    return result


def _register(model, captures: dict[str, list[torch.Tensor]]):
    backbone = model.backbone
    head = model.action_head
    handles = []

    def output_hook(label: str):
        def hook(_module, _inputs, output):
            captures[label].append(_tensor(output).detach().contiguous().cpu())

        return hook

    def input_hook(label: str):
        def hook(_module, inputs):
            captures[label].append(_tensor(inputs).detach().contiguous().cpu())

        return hook

    visual = backbone.model.visual
    handles.append(
        visual.patch_embed.proj.register_forward_pre_hook(
            input_hook("vision.patch_conv.input")
        )
    )
    handles.append(
        visual.patch_embed.proj.register_forward_hook(
            output_hook("vision.patch_conv.output")
        )
    )
    handles.append(
        visual.blocks[0].attn.qkv.register_forward_hook(
            output_hook("vision.block0.qkv.output")
        )
    )
    handles.append(
        visual.blocks[0].register_forward_hook(output_hook("vision.block0.output"))
    )
    handles.append(
        visual.blocks[-1].register_forward_hook(output_hook("vision.block23.output"))
    )
    handles.append(
        visual.merger.register_forward_hook(output_hook("vision.merger.output"))
    )
    handles.append(
        backbone.model.language_model.layers[0].register_forward_hook(
            output_hook("language.block0.output")
        )
    )
    handles.append(
        backbone.model.language_model.layers[-1].register_forward_hook(
            output_hook("language.block15.output")
        )
    )
    handles.append(
        head.vl_self_attention.register_forward_hook(output_hook("action.vl_projector"))
    )
    handles.append(
        head.state_encoder.register_forward_hook(output_hook("action.state_encoder"))
    )
    for name in ("W1", "W2", "W3"):
        module = getattr(head.action_encoder, name)
        handles.append(
            module.register_forward_pre_hook(
                input_hook(f"flow.action_encoder.{name}.input")
            )
        )
        handles.append(
            module.register_forward_hook(
                output_hook(f"flow.action_encoder.{name}.output")
            )
        )
    block0 = head.model.transformer_blocks[0]
    handles.append(
        block0.register_forward_pre_hook(input_hook("action.dit.block0.input"))
    )
    handles.append(
        block0.norm1.register_forward_hook(output_hook("action.dit.block0.norm1"))
    )
    handles.append(
        block0.attn1.to_q.register_forward_pre_hook(
            input_hook("action.dit.block0.q.input")
        )
    )
    handles.append(
        block0.attn1.to_q.register_forward_hook(
            output_hook("action.dit.block0.q.output")
        )
    )
    handles.append(
        block0.attn1.to_k.register_forward_hook(
            output_hook("action.dit.block0.k.output")
        )
    )
    handles.append(
        block0.attn1.to_k.register_forward_pre_hook(
            input_hook("action.dit.block0.k.input")
        )
    )
    handles.append(
        block0.attn1.to_v.register_forward_hook(
            output_hook("action.dit.block0.v.output")
        )
    )
    handles.append(
        block0.attn1.register_forward_hook(output_hook("action.dit.block0.attention"))
    )
    handles.append(block0.register_forward_hook(output_hook("action.dit.block0")))
    handles.append(
        head.model.transformer_blocks[-1].register_forward_hook(
            output_hook("action.dit.block31")
        )
    )
    handles.append(
        head.action_decoder.register_forward_hook(output_hook("flow.velocity"))
    )
    return handles, head.action_encoder.W2


def _run(adapter, model, examples, batch_size: int, *, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, frontier = _register(model, captures)
    batch = _compose_batch(adapter, examples[:batch_size])
    captures["preprocess.input_ids"].append(batch["inputs"]["input_ids"].detach().cpu())
    captures["preprocess.pixel_values"].append(
        batch["inputs"]["pixel_values"].detach().cpu()
    )
    captures["flow.initial_noise"].append(batch["noise"].detach().cpu())
    seed_everything(seed)
    context = set_batch_invariant_mode() if invariant else torch.no_grad()
    try:
        with context, torch.inference_mode():
            output = adapter.run_model(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu(), frontier


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


def _profile_frontier(module, values: list[torch.Tensor], category_id: int):
    reports = []
    for value in values:
        value = value.to(device=module.W.device)
        ids = torch.full(
            (value.shape[0],), category_id, dtype=torch.int32, device=module.W.device
        )
        with torch.inference_mode():
            module(value, ids)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as result:
                module(value, ids)
                torch.cuda.synchronize()
        reports.append(
            {
                "input_shape": list(value.shape),
                "selected_weight_shape": list(module.W[ids].shape),
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
        value = value.to(device=module.weight.device)
        with torch.inference_mode():
            module(value)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as result:
                module(value)
                torch.cuda.synchronize()
        reports.append(
            {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", default="scripts.model_invariance.adapters.groot_n17"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    stock_b1, stock_output_b1, frontier = _run(
        adapter, model, examples, 1, invariant=False, seed=args.seed
    )
    stock_b2, stock_output_b2, _ = _run(
        adapter, model, examples, 2, invariant=False, seed=args.seed
    )
    category_id = int(
        adapter.compose_batch(examples[:1])["inputs"]["embodiment_id"].item()
    )
    profile_report = _profile_frontier(
        frontier,
        [
            stock_b1["flow.action_encoder.W2.input"][0],
            stock_b2["flow.action_encoder.W2.input"][0],
        ],
        category_id,
    )

    # Isolate the next boundary after repairing only the first BMM. A fresh
    # torch.Library scope installs just that override and is destroyed before
    # testing the full public context below.
    from batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant

    bmm_only_library = torch.library.Library("aten", "IMPL")
    bmm_only_library.impl("aten::bmm", bmm_batch_invariant, "CUDA")
    try:
        bmm_b1, bmm_output_b1, _ = _run(
            adapter, model, examples, 1, invariant=False, seed=args.seed
        )
        bmm_b2, bmm_output_b2, _ = _run(
            adapter, model, examples, 2, invariant=False, seed=args.seed
        )
    finally:
        bmm_only_library._destroy()
    second_frontier = model.action_head.model.transformer_blocks[0].attn1.to_k
    second_profile_report = _profile_linear(
        second_frontier,
        [
            bmm_b1["action.dit.block0.k.input"][0],
            bmm_b2["action.dit.block0.k.input"][0],
        ],
    )

    fixed_b1, fixed_output_b1, _ = _run(
        adapter, model, examples, 1, invariant=True, seed=args.seed
    )
    fixed_b2, fixed_output_b2, _ = _run(
        adapter, model, examples, 2, invariant=True, seed=args.seed
    )
    stock = _compare_captures(stock_b1, stock_b2, stock_output_b1, stock_output_b2)
    bmm_only = _compare_captures(bmm_b1, bmm_b2, bmm_output_b1, bmm_output_b2)
    fixed = _compare_captures(fixed_b1, fixed_b2, fixed_output_b1, fixed_output_b2)
    # Records are grouped by module label rather than emitted in execution
    # order.  W2 call 0 is the first differing boundary: its input and the
    # preceding W1 output are exact, while the category-specific BMM output is
    # not.  Later W1 calls already contain propagated flow-state differences.
    first_stock_difference = next(
        record
        for record in stock
        if record.get("label") == "flow.action_encoder.W2.output"
        and record.get("call") == 0
    )
    first_after_bmm_difference = next(
        record
        for record in bmm_only
        if record.get("label") == "action.dit.block0.k.output"
        and record.get("call") == 0
    )
    report = {
        "model": adapter.name,
        "comparison": "B=1 target versus first sample at unrelated B=2",
        "first_stock_difference": first_stock_difference,
        "first_after_bmm_difference": first_after_bmm_difference,
        "stock_trace": stock,
        "bmm_only_trace": bmm_only,
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

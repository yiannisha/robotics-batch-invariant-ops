"""Trace batch-dependent boundaries in official PyTorch UVA LIBERO inference."""

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
    for attribute in ("last_hidden_state", "pooler_output", "sample"):
        item = getattr(value, attribute, None)
        if isinstance(item, torch.Tensor):
            return item
    raise TypeError(f"cannot select tensor from {type(value).__name__}")


def _hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    candidate_full_shape = list(candidate.shape)
    if reference.ndim == candidate.ndim and all(
        left <= right for left, right in zip(reference.shape, candidate.shape)
    ):
        candidate = candidate[tuple(slice(0, size) for size in reference.shape)]
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_full_shape": candidate_full_shape,
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

    text = policy.text_model.text_model
    first_attention = text.encoder.layers[0].self_attn
    add("text.block0.attention", first_attention)
    add("text.block0.attention.q_proj", first_attention.q_proj, inputs=True)
    add("text.block0.attention.k_proj", first_attention.k_proj)
    add("text.block0.attention.v_proj", first_attention.v_proj)
    add("text.block0.mlp.fc1", text.encoder.layers[0].mlp.fc1, inputs=True)
    add("text.block0.mlp.fc2", text.encoder.layers[0].mlp.fc2, inputs=True)
    add("text.block0", text.encoder.layers[0])
    add("text.block11", text.encoder.layers[-1])

    vae = policy.vae_model
    add("vae.encoder.conv_in", vae.encoder.conv_in, inputs=True)
    for index, down in enumerate(vae.encoder.down[:-1]):
        add(f"vae.encoder.down{index}.downsample", down.downsample)
    add("vae.encoder.mid.attention", vae.encoder.mid.attn_1)
    add("vae.encoder.conv_out", vae.encoder.conv_out)
    add("vae.quant_conv", vae.quant_conv)

    model = policy.model
    add("mar.text_projection", model.text_proj_cond, inputs=True)
    add("mar.condition_projection", model.proj_cond_x_layer, inputs=True)
    add("mar.encoder.block0", model.encoder_blocks[0])
    add("mar.encoder.block15", model.encoder_blocks[-1])
    add("mar.decoder.block0", model.decoder_blocks[0])
    add("mar.decoder.block15", model.decoder_blocks[-1])
    add("action.diffusion.input_projection", model.diffactloss.net.input_proj, inputs=True)
    add("action.diffusion.resblock0", model.diffactloss.net.res_blocks[0])
    add("action.diffusion.final", model.diffactloss.net.final_layer)

    policy.trace_callback = capture
    return handles, profiled_modules


def _run(adapter, model, examples, batch_size: int, *, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles, profiled_modules = _register(model, captures)
    batch = _compose_batch(adapter, examples[:batch_size])
    for key in ("vae_noise", "action_initial_noise", "action_step_noise"):
        captures[f"input.{key}"].append(batch[key].detach().cpu())
    seed_everything(seed)
    try:
        with set_batch_invariant_mode(invariant), torch.inference_mode():
            output = adapter.run_model(model, batch)
    finally:
        adapter.trace_callback = None
        for handle in handles:
            handle.remove()
    return captures, output.detach().contiguous().cpu(), profiled_modules


def _compare(reference, candidate, reference_output, candidate_output):
    records = []
    for label, reference_values in reference.items():
        candidate_values = candidate.get(label, [])
        if len(reference_values) != len(candidate_values):
            records.append(
                {
                    "label": label,
                    "error": "capture count differs",
                    "reference_count": len(reference_values),
                    "candidate_count": len(candidate_values),
                    "exact": False,
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
        parameter = next(module.parameters())
        value = value.to(device=parameter.device, dtype=parameter.dtype)
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
                "weight_shape": (
                    list(module.weight.shape) if hasattr(module, "weight") else None
                ),
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


def _stock_transposed_convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    from batch_invariant_ops.batch_invariant_ops import convolution_batch_invariant

    if not transposed:
        return convolution_batch_invariant(
            input, weight, bias, stride, padding, dilation, False, output_padding, groups
        )
    if input.ndim != 5:
        raise NotImplementedError("UVA trace only expects a transposed 3-D convolution")
    output = torch.ops.aten.cudnn_convolution_transpose.default(
        input,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        False,
        False,
        True,
    )
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1, 1)
    return output


def _run_stage(adapter, model, examples, implementations, seed):
    library = torch.library.Library("aten", "IMPL")
    for operator, implementation in implementations:
        library.impl(operator, implementation, "CUDA")
    try:
        b1, output_b1, modules = _run(adapter, model, examples, 1, invariant=False, seed=seed)
        b2, output_b2, _ = _run(adapter, model, examples, 2, invariant=False, seed=seed)
    finally:
        library._destroy()
    return _compare(b1, b2, output_b1, output_b2), b1, b2, modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.uva_libero10")
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
        mm_batch_invariant,
    )

    gemm_implementations = (
        ("aten::mm", mm_batch_invariant),
        ("aten::addmm", addmm_batch_invariant),
        ("aten::bmm", bmm_batch_invariant),
    )
    gemm, _, _, _ = _run_stage(adapter, model, examples, gemm_implementations, args.seed)
    gemm_regular_conv, _, _, _ = _run_stage(
        adapter,
        model,
        examples,
        (*gemm_implementations, ("aten::convolution", _stock_transposed_convolution)),
        args.seed,
    )
    fixed_b1, fixed_output_b1, _ = _run(
        adapter, model, examples, 1, invariant=True, seed=args.seed
    )
    fixed_b2, fixed_output_b2, _ = _run(
        adapter, model, examples, 2, invariant=True, seed=args.seed
    )
    fixed = _compare(fixed_b1, fixed_b2, fixed_output_b1, fixed_output_b2)

    first_stock = _first_difference(stock)
    profile_report = None
    if first_stock is not None:
        module = modules.get(first_stock["label"])
        input_label = first_stock["label"].removesuffix(".output") + ".input"
        if first_stock["label"] == "text.block0.attention.output":
            input_label = "text.block0.attention.q_proj.input"
        if module is not None and input_label in stock_b1:
            profile_report = _profile_module(
                module, [stock_b1[input_label][0], stock_b2[input_label][0]]
            )

    report = {
        "model": adapter.name,
        "comparison": "B=1 target versus first sample at unrelated B=2",
        "first_stock_difference": first_stock,
        "first_after_gemm_difference": _first_difference(gemm),
        "first_after_gemm_regular_convolution_difference": _first_difference(
            gemm_regular_conv
        ),
        "stock_trace": stock,
        "gemm_only_trace": gemm,
        "gemm_regular_convolution_trace": gemm_regular_conv,
        "fixed_trace": fixed,
        "fixed_all_exact": all(record.get("exact", False) for record in fixed),
        "first_divergent_operator_profile": profile_report,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Trace π0.5's first patch-projection boundary at request batches one and two."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import (
    _compose_batch,
    _first_sample,
    _run_once,
    compare_outputs,
    seed_everything,
)


def _comparison(reference, candidate) -> dict:
    records = compare_outputs(reference, candidate)
    return {
        "exact": all(record.exact for record in records),
        "tensors": [asdict(record) for record in records],
    }


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _target_like(reference: torch.Tensor, candidate: torch.Tensor):
    if candidate.shape == reference.shape:
        return candidate
    if (
        reference.ndim > 0
        and candidate.ndim == reference.ndim
        and candidate.shape[1:] == reference.shape[1:]
        and candidate.shape[0] >= reference.shape[0]
    ):
        return candidate[: reference.shape[0]]
    return None


def _leaf_trace(adapter, model, batch, seed: int, references=None):
    records = []
    handles = []
    occurrences = {}

    def register(name, module):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                return
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            record = {
                "name": name,
                "occurrence": occurrence,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            if references is None:
                record["hash"] = _tensor_hash(tensor)
            elif len(records) < len(references):
                reference = references[len(records)]
                comparable = (
                    reference["name"] == name
                    and reference["occurrence"] == occurrence
                )
                if comparable:
                    reference_shape = tuple(reference["shape"])
                    placeholder = torch.empty(reference_shape, dtype=tensor.dtype)
                    target = _target_like(placeholder, tensor)
                    comparable = target is not None
                record["comparable"] = comparable
                record["target_hash"] = _tensor_hash(target) if comparable else None
                record["exact"] = comparable and record["target_hash"] == reference["hash"]
            records.append(record)

        handles.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name and not any(module.children()):
            register(name, module)
    try:
        _run_once(adapter, model, batch, seed)
    finally:
        for handle in handles:
            handle.remove()
    return records


def _capture_leaf(adapter, model, batch, seed: int, module, occurrence: int, invariant: bool):
    captured = []

    def hook(_module, _inputs, output):
        tensor = _first_tensor(output)
        if tensor is not None:
            captured.append(tensor.detach().cpu())

    handle = module.register_forward_hook(hook)
    mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
    try:
        with mode:
            _run_once(adapter, model, batch, seed)
    finally:
        handle.remove()
    if occurrence >= len(captured):
        raise RuntimeError(f"missing occurrence {occurrence}; observed {len(captured)}")
    return captured[occurrence]


def _run(adapter, model, batch, seed: int, invariant: bool) -> dict:
    captured = {}
    patch_projection = model.vit.to_patch_embed[0]

    def capture_input(_module, inputs):
        captured["patch_projection_input"] = inputs[0][:3].detach().cpu()

    def capture_output(_module, _inputs, output):
        captured["patch_projection_output"] = output[:3].detach().cpu()

    handles = (
        patch_projection.register_forward_pre_hook(capture_input),
        patch_projection.register_forward_hook(capture_output),
    )
    mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
    try:
        with mode:
            output = _run_once(adapter, model, batch, seed)
    finally:
        for handle in handles:
            handle.remove()
    captured["final_action"] = _first_sample(output).detach().cpu()
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.pizero_pi05_reference",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)

    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    singleton = _compose_batch(adapter, examples[:1])
    paired = _compose_batch(adapter, examples)

    native_reference_trace = _leaf_trace(adapter, model, singleton, args.seed)
    native_candidate_trace = _leaf_trace(
        adapter, model, paired, args.seed, references=native_reference_trace
    )
    first_index = next(
        (
            index
            for index, record in enumerate(native_candidate_trace)
            if record.get("comparable") and not record.get("exact")
        ),
        None,
    )
    if first_index is None:
        raise RuntimeError("no divergent comparable leaf-module output was found")
    first = native_reference_trace[first_index]
    previous = next(
        (
            native_reference_trace[index]
            for index in range(first_index - 1, -1, -1)
            if native_candidate_trace[index].get("comparable")
            and native_candidate_trace[index].get("exact")
        ),
        None,
    )
    modules = dict(model.named_modules())
    first_module = modules[first["name"]]
    first_native_reference = _capture_leaf(
        adapter, model, singleton, args.seed, first_module, first["occurrence"], False
    )
    first_native_candidate_full = _capture_leaf(
        adapter, model, paired, args.seed, first_module, first["occurrence"], False
    )
    first_native_candidate = _target_like(first_native_reference, first_native_candidate_full)
    if first_native_candidate is None:
        raise RuntimeError("first divergent leaf cannot be sliced to the target request")
    first_fixed_reference = _capture_leaf(
        adapter, model, singleton, args.seed, first_module, first["occurrence"], True
    )
    first_fixed_candidate_full = _capture_leaf(
        adapter, model, paired, args.seed, first_module, first["occurrence"], True
    )
    first_fixed_candidate = _target_like(first_fixed_reference, first_fixed_candidate_full)
    if first_fixed_candidate is None:
        raise RuntimeError("fixed leaf cannot be sliced to the target request")

    modes = {}
    for name, invariant in (("native", False), ("batch_invariant", True)):
        reference = _run(adapter, model, singleton, args.seed, invariant)
        candidate = _run(adapter, model, paired, args.seed, invariant)
        modes[name] = {
            key: _comparison(reference[key], candidate[key])
            for key in ("patch_projection_input", "patch_projection_output", "final_action")
        }

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "batch_size": 2,
        "composition": "unrelated",
        "request_to_vision_batch_mapping": {
            "request_batch_1": 3,
            "request_batch_2": 6,
            "reason": "three camera views are flattened into the vision batch",
        },
        "boundary": "vit.to_patch_embed.0",
        "historical_boundary_operation": "nn.Conv2d -> aten::convolution",
        "host_leaf_scan": {
            "comparable_leaf_outputs": sum(
                bool(record.get("comparable")) for record in native_candidate_trace
            ),
            "first_divergent_index": first_index,
            "last_exact_leaf": previous,
            "first_divergent_leaf": {
                **first,
                "module_type": type(first_module).__name__,
                "native_comparison": _comparison(
                    first_native_reference, first_native_candidate
                ),
                "batch_invariant_comparison": _comparison(
                    first_fixed_reference, first_fixed_candidate
                ),
            },
        },
        "modes": modes,
        "native_first_divergence": (
            f"{first['name']} output occurrence {first['occurrence']}"
        ),
        "batch_invariant_boundary_exact": all(
            item["exact"] for item in modes["batch_invariant"].values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

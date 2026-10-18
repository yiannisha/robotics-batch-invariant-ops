"""Locate Pi0-FAST's first divergent vision-tower leaf on the current host."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, seed_everything


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    for attribute in ("pooler_output", "last_hidden_state"):
        tensor = getattr(value, attribute, None)
        if isinstance(tensor, torch.Tensor):
            return tensor
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


def _hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _target_like(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor | None:
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


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    result: dict[str, Any] = {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact": exact,
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }
    if reference.shape != candidate.shape:
        result["candidate_shape"] = list(candidate.shape)
        return result
    different = reference != candidate
    result["differing_elements"] = int(different.sum().item())
    locations = torch.nonzero(different, as_tuple=False)
    result["first_differing_index"] = (
        [int(item) for item in locations[0].tolist()] if locations.numel() else None
    )
    if reference.is_floating_point():
        absolute = (reference.float() - candidate.float()).abs()
        result["max_abs_difference"] = float(absolute.max().item())
        result["mean_abs_difference"] = float(absolute.mean().item())
    return result


def _vision_tower(model: torch.nn.Module) -> torch.nn.Module:
    return model.paligemma_with_expert.paligemma.model.vision_tower


def _run(adapter, model, samples, batch_size: int, seed: int) -> None:
    batch = _compose_batch(adapter, samples[:batch_size])
    seed_everything(seed)
    images = batch["images"].to(adapter.device)
    image_masks = batch["image_masks"].to(adapter.device)
    image_list = [images[:, index] * 2 - 1 for index in range(images.shape[1])]
    mask_list = [image_masks[:, index] for index in range(image_masks.shape[1])]
    with torch.inference_mode():
        # One generated token is enough to execute the complete vision prefix.
        # Skip the adapter's action decoder because an intentionally truncated
        # sequence does not yet contain its required FAST action prefix.
        model.sample_actions_fast_kv_cache(
            image_list,
            mask_list,
            batch["tokens"].to(adapter.device),
            batch["token_mask"].to(adapter.device),
            max_decoding_steps=1,
            temperature=0.0,
        )


def _scan(adapter, model, samples, batch_size: int, seed: int):
    records: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    handles = []

    def register(name: str, module: torch.nn.Module) -> None:
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                return
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            records.append(
                {
                    "name": name,
                    "occurrence": occurrence,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "tensor": tensor.detach().cpu(),
                }
            )

        handles.append(module.register_forward_hook(hook))

    for name, module in _vision_tower(model).named_modules():
        if name and not any(module.children()):
            register(name, module)
    try:
        _run(adapter, model, samples, batch_size, seed)
    finally:
        for handle in handles:
            handle.remove()
    return records


def _capture(
    adapter,
    model,
    samples,
    batch_size: int,
    seed: int,
    module: torch.nn.Module,
    occurrence: int,
    invariant: bool,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        tensor = _first_tensor(output)
        if tensor is not None:
            captured.append(tensor.detach().cpu())

    handle = module.register_forward_hook(hook)
    mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
    try:
        with mode:
            _run(adapter, model, samples, batch_size, seed)
    finally:
        handle.remove()
    if occurrence >= len(captured):
        raise RuntimeError(f"missing occurrence {occurrence}; observed {len(captured)}")
    return captured[occurrence]


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "tensor"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", default="scripts.model_invariance.adapters.lerobot_pi0fast"
    )
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)

    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    model = adapter.load_model().eval()
    samples = list(adapter.load_example_inputs(2))

    reference_trace = _scan(adapter, model, samples, 1, args.seed)
    candidate_trace = _scan(adapter, model, samples, 2, args.seed)
    comparisons = []
    for reference, candidate in zip(reference_trace, candidate_trace, strict=False):
        comparable = (
            reference["name"] == candidate["name"]
            and reference["occurrence"] == candidate["occurrence"]
        )
        target = _target_like(reference["tensor"], candidate["tensor"]) if comparable else None
        comparisons.append(
            {
                "comparable": target is not None,
                "exact": target is not None and torch.equal(reference["tensor"], target),
            }
        )
    first_index = next(
        (index for index, item in enumerate(comparisons) if item["comparable"] and not item["exact"]),
        None,
    )
    if first_index is None:
        raise RuntimeError("no divergent comparable vision leaf output was found")
    first = reference_trace[first_index]
    previous_index = next(
        (
            index
            for index in range(first_index - 1, -1, -1)
            if comparisons[index]["comparable"] and comparisons[index]["exact"]
        ),
        None,
    )
    modules = dict(_vision_tower(model).named_modules())
    module = modules[first["name"]]

    mode_records = {}
    for label, invariant in (("native", False), ("batch_invariant", True)):
        reference = _capture(
            adapter, model, samples, 1, args.seed, module, first["occurrence"], invariant
        )
        candidate_full = _capture(
            adapter, model, samples, 2, args.seed, module, first["occurrence"], invariant
        )
        candidate = _target_like(reference, candidate_full)
        if candidate is None:
            raise RuntimeError("first divergent output cannot be sliced to the target request")
        mode_records[label] = _comparison(reference, candidate)

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "batch_size": 2,
        "composition": "unrelated",
        "scope": "leaf modules within the first divergent coarse boundary: vision_tower",
        "vision_calls_per_request": 3,
        "comparable_leaf_outputs": sum(item["comparable"] for item in comparisons),
        "first_divergent_index": first_index,
        "last_exact_leaf": (
            _public(reference_trace[previous_index]) if previous_index is not None else None
        ),
        "first_divergent_leaf": {
            **_public(first),
            "module_type": type(module).__name__,
            "native_comparison": mode_records["native"],
            "batch_invariant_comparison": mode_records["batch_invariant"],
        },
        "native_first_divergence": f"vision_tower.{first['name']} output occurrence {first['occurrence']}",
        "batch_invariant_boundary_exact": mode_records["batch_invariant"]["exact"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

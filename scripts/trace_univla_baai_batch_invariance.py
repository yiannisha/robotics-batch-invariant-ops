"""Trace the autoregressive batch-dependent boundary in official BAAI UniVLA."""

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
    raise TypeError(f"cannot select tensor from {type(value).__name__}")


def _hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    full_shape = list(candidate.shape)
    if reference.ndim == candidate.ndim and all(
        left <= right for left, right in zip(reference.shape, candidate.shape)
    ):
        candidate = candidate[tuple(slice(0, size) for size in reference.shape)]
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    record: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_full_shape": full_shape,
        "dtype": str(reference.dtype),
        "exact": exact,
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }
    if reference.shape == candidate.shape:
        mask = reference != candidate
        locations = torch.nonzero(mask, as_tuple=False)
        record["differing_elements"] = int(mask.sum().item())
        record["first_differing_index"] = (
            [int(index) for index in locations[0].tolist()] if locations.numel() else None
        )
        if reference.is_floating_point():
            absolute = (reference.float() - candidate.float()).abs()
            record["max_abs_difference"] = float(absolute.max().item())
            record["mean_abs_difference"] = float(absolute.mean().item())
    return record


def _modules(policy) -> dict[str, torch.nn.Module]:
    layer = policy.model.model.layers[0]
    return {
        "layer0.input_norm": layer.input_layernorm,
        "layer0.q_proj": layer.self_attn.q_proj,
        "layer0.attention": layer.self_attn,
        "layer0.output": layer,
        "final_norm": policy.model.model.norm,
        "lm_head": policy.model.lm_head,
    }


def _run(adapter, policy, samples, invariant: bool, seed: int):
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []
    for label, module in _modules(policy).items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, label=label: captures[label].append(
                    _tensor(output)[:1, -1:].detach().contiguous().cpu()
                )
            )
        )
    batch = _compose_batch(adapter, samples)
    input_ids, attention_mask = adapter.prepare_batch(policy, batch)
    seed_everything(seed)
    try:
        with set_batch_invariant_mode(invariant), torch.inference_mode():
            output = policy.model.generate(
                input_ids,
                policy.generation_config,
                max_new_tokens=80,
                logits_processor=[policy.logits_processor],
                attention_mask=attention_mask,
            )
    finally:
        for handle in handles:
            handle.remove()
    return captures, output[:, input_ids.shape[1] :].detach().contiguous().cpu()


def _compare_runs(reference, candidate, reference_tokens, candidate_tokens):
    records = []
    labels = list(reference)
    call_count = min(len(reference[label]) for label in labels)
    for call in range(call_count):
        for label in labels:
            record = {"label": label, "call": call, "query_length": 1293 if call == 0 else 1}
            record.update(_comparison(reference[label][call], candidate[label][call]))
            records.append(record)
    token_record = {"label": "generation.tokens", "call": 0}
    token_record.update(_comparison(reference_tokens, candidate_tokens))
    records.append(token_record)
    return records


def _first_difference(records):
    return next((record for record in records if not record.get("exact", False)), None)


def _operator_comparison(policy, hidden, operator: str | None):
    library = None
    if operator is not None:
        from batch_invariant_ops.batch_invariant_ops import (
            addmm_batch_invariant,
            mm_batch_invariant,
        )

        implementation = {
            "aten::addmm": addmm_batch_invariant,
            "aten::mm": mm_batch_invariant,
        }[operator]
        library = torch.library.Library("aten", "IMPL")
        library.impl(operator, implementation, "CUDA")
    try:
        with torch.inference_mode():
            reference = policy.model.lm_head(hidden)
            candidate = policy.model.lm_head(torch.cat([hidden, hidden], dim=0))
    finally:
        if library is not None:
            library._destroy()
    return _comparison(reference.cpu(), candidate.cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.univla_baai")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    report = {}
    stock_hidden = None
    for mode, invariant in (("stock", False), ("fixed", True)):
        reference, reference_tokens = _run(adapter, policy, [examples[0]], invariant, args.seed)
        if mode == "stock":
            stock_hidden = reference["final_norm"][1].to("cuda")
        for composition, samples in (
            ("duplicate", [examples[0], examples[0]]),
            ("unrelated", examples),
        ):
            candidate, candidate_tokens = _run(adapter, policy, samples, invariant, args.seed)
            records = _compare_runs(reference, candidate, reference_tokens, candidate_tokens)
            report[f"{mode}_{composition}"] = {
                "first_difference": _first_difference(records),
                "records": records,
            }

    assert stock_hidden is not None
    report["operator_isolation"] = {
        "stock": _operator_comparison(policy, stock_hidden, None),
        "addmm_only": _operator_comparison(policy, stock_hidden, "aten::addmm"),
        "mm_only": _operator_comparison(policy, stock_hidden, "aten::mm"),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value["first_difference"]
                for key, value in report.items()
                if key != "operator_isolation"
            }
            | {"operator_isolation": report["operator_isolation"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

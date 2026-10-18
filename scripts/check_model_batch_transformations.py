"""Exercise position, permutation, partition, and singleton-fidelity checks."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import (
    _compose_batch,
    _run_once,
    compare_outputs,
    seed_everything,
)


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not sizes or any(size < 2 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be integers of at least two")
    return sizes


def _map_tensors(value: Any, function) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return type(value)((key, _map_tensors(item, function)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    raise TypeError(f"unsupported output type {type(value).__name__}")


def _select(value: Any, indices: list[int]) -> Any:
    def select_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor
        index = torch.tensor(indices, device=tensor.device)
        return torch.index_select(tensor, 0, index)

    return _map_tensors(value, select_tensor)


def _concat(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            if not all(torch.equal(first, value) for value in values[1:]):
                raise ValueError("scalar outputs differ across partitions")
            return first
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return type(first)((key, _concat([value[key] for value in values])) for key in first)
    if isinstance(first, tuple):
        return tuple(_concat([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        return [_concat([value[index] for value in values]) for index in range(len(first))]
    raise TypeError(f"unsupported output type {type(first).__name__}")


def _comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    tensors = compare_outputs(reference, candidate)
    return {
        "exact": all(item.exact for item in tensors),
        "tensors": [asdict(item) for item in tensors],
    }


def _run_mode(adapter, model, examples, batch_sizes, invariant: bool, seed: int):
    records = []
    mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
    with mode:
        target = examples[0]
        singleton = _run_once(adapter, model, _compose_batch(adapter, [target]), seed)
        for size in batch_sizes:
            request_set = examples[:size]
            canonical = _run_once(adapter, model, _compose_batch(adapter, request_set), seed)

            for position in sorted({0, size // 2, size - 1}):
                positioned = list(examples[1:size])
                positioned.insert(position, target)
                candidate = _run_once(
                    adapter, model, _compose_batch(adapter, positioned), seed
                )
                records.append(
                    {
                        "transformation": "target_position",
                        "batch_size": size,
                        "target_position": position,
                        **_comparison(singleton, _select(candidate, [position])),
                    }
                )

            permutation = list(reversed(range(size)))
            permuted = _run_once(
                adapter,
                model,
                _compose_batch(adapter, [request_set[index] for index in permutation]),
                seed,
            )
            restore = [permutation.index(index) for index in range(size)]
            records.append(
                {
                    "transformation": "permutation",
                    "batch_size": size,
                    "candidate_order": permutation,
                    **_comparison(canonical, _select(permuted, restore)),
                }
            )

            split = size // 2
            partitions = (request_set[:split], request_set[split:])
            partitioned = _concat(
                [
                    _run_once(adapter, model, _compose_batch(adapter, partition), seed)
                    for partition in partitions
                ]
            )
            records.append(
                {
                    "transformation": "partition",
                    "batch_size": size,
                    "partition_sizes": [len(partition) for partition in partitions],
                    **_comparison(canonical, partitioned),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--batch-sizes", type=_parse_sizes, default=(2, 4, 8, 16, 32))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes",
        default="native,batch_invariant",
        help="Comma-separated subset of native,batch_invariant",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected_modes = {item.strip() for item in args.modes.split(",") if item.strip()}
    valid_modes = {"native", "batch_invariant"}
    if not selected_modes or not selected_modes <= valid_modes:
        parser.error(f"--modes must be a comma-separated subset of {sorted(valid_modes)}")
    if args.output.exists():
        raise FileExistsError(args.output)

    module = importlib.import_module(args.adapter)
    adapter = module.adapter
    seed_everything(args.seed)
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    examples = list(adapter.load_example_inputs(max(args.batch_sizes)))

    modes = {
        name: _run_mode(adapter, model, examples, args.batch_sizes, invariant, args.seed)
        for name, invariant in (("native", False), ("batch_invariant", True))
        if name in selected_modes
    }
    native_singleton = _run_once(
        adapter, model, _compose_batch(adapter, [examples[0]]), args.seed
    )
    with set_batch_invariant_mode():
        invariant_singleton = _run_once(
            adapter, model, _compose_batch(adapter, [examples[0]]), args.seed
        )
    report = {
        "model": adapter.name,
        "seed": args.seed,
        "batch_sizes": list(args.batch_sizes),
        "modes": {
            name: {
                "all_exact": all(record["exact"] for record in records),
                "records": records,
            }
            for name, records in modes.items()
        },
        "singleton_fidelity": _comparison(native_singleton, invariant_singleton),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

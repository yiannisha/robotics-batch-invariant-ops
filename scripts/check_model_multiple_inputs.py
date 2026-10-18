"""Check several adapter examples under duplicate and unrelated composition."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

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


def _integers(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--target-indices", type=_integers, default=(0, 20, 40))
    parser.add_argument("--batch-sizes", type=_integers, default=(2, 4))
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
    if any(size < 2 for size in args.batch_sizes):
        raise ValueError("batch sizes must be at least two")

    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    example_count = max(max(args.target_indices) + 1, max(args.batch_sizes))
    examples = list(adapter.load_example_inputs(example_count))

    modes = {}
    for name, invariant in (("native", False), ("batch_invariant", True)):
        if name not in selected_modes:
            continue
        records = []
        mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
        with mode:
            for target_index in args.target_indices:
                target = examples[target_index]
                reference = _run_once(
                    adapter, model, _compose_batch(adapter, [target]), args.seed
                )
                companions = [
                    example for index, example in enumerate(examples) if index != target_index
                ]
                for composition in ("duplicate", "unrelated"):
                    for batch_size in args.batch_sizes:
                        samples = (
                            [target] * batch_size
                            if composition == "duplicate"
                            else [target, *companions[: batch_size - 1]]
                        )
                        candidate = _first_sample(
                            _run_once(
                                adapter,
                                model,
                                _compose_batch(adapter, samples),
                                args.seed,
                            )
                        )
                        comparisons = compare_outputs(reference, candidate)
                        records.append(
                            {
                                "target_index": target_index,
                                "composition": composition,
                                "batch_size": batch_size,
                                "exact": all(item.exact for item in comparisons),
                                "tensors": [asdict(item) for item in comparisons],
                            }
                        )
        modes[name] = {
            "all_exact": all(record["exact"] for record in records),
            "records": records,
        }

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "target_indices": list(args.target_indices),
        "batch_sizes": list(args.batch_sizes),
        "compositions": ["duplicate", "unrelated"],
        "modes": modes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

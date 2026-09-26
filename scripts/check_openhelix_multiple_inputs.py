"""Check all three public CALVIN targets through official PyTorch OpenHelix."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.openhelix_calvin")
    parser.add_argument("--batch-sizes", default="2,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    models = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(max(batch_sizes)))
    targets = examples[:3]
    report = {
        "model": adapter.name,
        "targets": "three real public CALVIN ABC transitions",
        "target_indices": list(range(len(targets))),
        "batch_sizes": list(batch_sizes),
        "compositions": ["duplicate", "unrelated"],
        "seed": args.seed,
        "modes": {},
    }
    for invariant_ops in (False, True):
        results = []
        with set_batch_invariant_mode(invariant_ops):
            for target_index, target in enumerate(targets):
                companions = [
                    value for index, value in enumerate(examples) if index != target_index
                ]
                reference = _run_once(adapter, models, _compose_batch(adapter, [target]), args.seed)
                for composition in ("duplicate", "unrelated"):
                    for batch_size in batch_sizes:
                        samples = (
                            [target] * batch_size
                            if composition == "duplicate"
                            else [
                                target,
                                *[
                                    companions[index % len(companions)]
                                    for index in range(batch_size - 1)
                                ],
                            ]
                        )
                        candidate = _first_sample(
                            _run_once(
                                adapter,
                                models,
                                _compose_batch(adapter, samples),
                                args.seed,
                            )
                        )
                        comparisons = compare_outputs(reference, candidate)
                        results.append(
                            {
                                "target_index": target_index,
                                "composition": composition,
                                "batch_size": batch_size,
                                "exact": all(item.exact for item in comparisons),
                                "tensors": [asdict(item) for item in comparisons],
                            }
                        )
        report["modes"]["fixed" if invariant_ops else "baseline"] = {
            "all_exact": all(item["exact"] for item in results),
            "passed": sum(item["exact"] for item in results),
            "total": len(results),
            "results": results,
        }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

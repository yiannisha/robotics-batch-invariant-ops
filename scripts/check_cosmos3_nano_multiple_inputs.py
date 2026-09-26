"""Check several real DROID targets through Cosmos 3 policy inference."""

from __future__ import annotations

import argparse
import importlib
import json
import os
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
    parser.add_argument(
        "--adapter", default="scripts.model_invariance.adapters.cosmos3_nano_policy"
    )
    parser.add_argument("--target-frames", default="0,20,40")
    parser.add_argument("--batch-sizes", default="2,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target_frames = tuple(int(item) for item in args.target_frames.split(","))
    batch_sizes = tuple(int(item) for item in args.batch_sizes.split(","))
    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    report = {
        "model": adapter.name,
        "target_frames": list(target_frames),
        "batch_sizes": list(batch_sizes),
        "compositions": ["duplicate", "unrelated"],
        "seed": args.seed,
        "modes": {},
    }

    for invariant_ops in (False, True):
        results = []
        with set_batch_invariant_mode(invariant_ops):
            for target_frame in target_frames:
                os.environ["COSMOS3_TARGET_FRAME"] = str(target_frame)
                examples = list(adapter.load_example_inputs(max(batch_sizes)))
                target = examples[0]
                reference = _run_once(adapter, model, _compose_batch(adapter, [target]), args.seed)
                for composition in ("duplicate", "unrelated"):
                    for batch_size in batch_sizes:
                        samples = (
                            [target] * batch_size
                            if composition == "duplicate"
                            else [target, *examples[1:batch_size]]
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
                        results.append(
                            {
                                "target_frame": target_frame,
                                "composition": composition,
                                "batch_size": batch_size,
                                "exact": all(item.exact for item in comparisons),
                                "tensors": [asdict(item) for item in comparisons],
                            }
                        )
        report["modes"]["fixed" if invariant_ops else "baseline"] = {
            "all_exact": all(item["exact"] for item in results),
            "results": results,
        }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

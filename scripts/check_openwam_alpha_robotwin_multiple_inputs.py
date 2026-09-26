"""Check repaired OpenWAM joint inference on several real episode frames."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import compare_outputs, seed_everything


def _run(adapter, model, samples, seed):
    seed_everything(seed)
    with torch.inference_mode():
        output = adapter.run_model(model, adapter.compose_batch(samples))
    return {key: value.detach().cpu().contiguous() for key, value in output.items()}


def _first(output):
    return {key: value[:1] for key, value in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.openwam_alpha_robotwin",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(4))
    checks = []
    with set_batch_invariant_mode():
        for target_index in (0, 1, 2):
            target = examples[target_index]
            reference = _run(adapter, model, [target], args.seed)
            for batch_size in (2, 4):
                duplicate = _first(_run(adapter, model, [target] * batch_size, args.seed))
                companions = [
                    examples[(target_index + offset + 1) % len(examples)]
                    for offset in range(batch_size - 1)
                ]
                unrelated = _first(_run(adapter, model, [target, *companions], args.seed))
                for composition, candidate in (
                    ("duplicate", duplicate),
                    ("unrelated", unrelated),
                ):
                    comparisons = compare_outputs(reference, candidate)
                    checks.append(
                        {
                            "target_frame": target_index,
                            "batch_size": batch_size,
                            "composition": composition,
                            "exact": all(item.exact for item in comparisons),
                            "tensors": [asdict(item) for item in comparisons],
                        }
                    )

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "targets": "three distinct frames from a real three-camera RoboTwin episode",
        "surfaces": ["physical actions", "joint video latent"],
        "all_exact": all(check["exact"] for check in checks),
        "checks": checks,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_exact"] else 1)


if __name__ == "__main__":
    main()

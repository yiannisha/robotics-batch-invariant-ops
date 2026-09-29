"""Check explicit-noise Xiaomi inference against the released B=1 path."""

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

from scripts.model_invariance.harness import compare_outputs, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.xiaomi_robotics_1_robocasa",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = adapter.load_example_inputs(3)
    checks = []
    for frame_index, example in enumerate(examples):
        batch = adapter.compose_batch([example])
        with torch.inference_mode():
            official = adapter.run_official_model(model, batch)
            explicit = adapter.run_model(model, batch)
        differences = compare_outputs(official, explicit)
        checks.append(
            {
                "frame_index": frame_index,
                "exact": all(item.exact for item in differences),
                "tensors": [asdict(item) for item in differences],
            }
        )

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "comparison": "released seeded forward versus explicit per-sample noise",
        "official_equivalent": all(check["exact"] for check in checks),
        "checks": checks,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["official_equivalent"] else 1)


if __name__ == "__main__":
    main()

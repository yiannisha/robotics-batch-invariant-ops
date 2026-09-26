#!/usr/bin/env python3
"""Check the adapter's B=1 path against released OpenWAM inference."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_invariance.harness import compare_outputs, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.openwam_alpha_robotwin",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["OPENWAM_KEEP_PREPROCESSORS"] = "1"
    seed_everything(0)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model()
    sample = adapter.load_example_inputs(1)[0]
    batch = adapter.compose_batch([sample])
    with torch.inference_mode():
        adapted = adapter.run_model(model, batch)
        official = adapter.run_official_model(model, batch)

    comparisons = compare_outputs(official, {"actions": adapted["actions"]})
    report = {
        "model": adapter.name,
        "batch_size": 1,
        "surface": "actions",
        "official_equivalent": all(item.exact for item in comparisons),
        "comparisons": [asdict(item) for item in comparisons],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["official_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

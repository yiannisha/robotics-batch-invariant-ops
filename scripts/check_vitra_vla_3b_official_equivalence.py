"""Check the VITRA adapter against the released private inference path."""

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
        default="scripts.model_invariance.adapters.vitra_vla_3b",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(2))
    checks = []
    for composition, samples in (
        ("single", [examples[0]]),
        ("unrelated_b2", [examples[0], examples[1]]),
    ):
        batch = adapter.compose_batch(samples)
        with torch.inference_mode():
            official = adapter.run_official_model(model, batch).detach().cpu().contiguous()
            traced = adapter.run_model(model, batch).detach().cpu().contiguous()
        comparison = compare_outputs(official, traced)[0]
        checks.append({"composition": composition, **asdict(comparison)})

    report = {
        "model": adapter.name,
        "seed": args.seed,
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

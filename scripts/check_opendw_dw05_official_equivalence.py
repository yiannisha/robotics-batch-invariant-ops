"""Compare the DW05 adapter with the released B=1 inference method."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.opendw_dw05",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # The released method recomputes both encoders, so retain them for this
    # equivalence check. Batch sweeps offload them after preprocessing.
    os.environ["OPENDW_KEEP_PREPROCESSORS"] = "1"
    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]
    batch = adapter.compose_batch([target])
    with torch.inference_mode():
        official = adapter.run_official_model(model, batch).detach().cpu().contiguous()
        adapted = adapter.run_model(model, batch).detach().cpu().contiguous()
    comparison = asdict(compare_outputs(official, adapted)[0])
    report = {
        "model": adapter.name,
        "seed": args.seed,
        "scope": "released infer_action B=1 path",
        "official_equivalent": comparison["exact"],
        "comparison": comparison,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["official_equivalent"] else 1)


if __name__ == "__main__":
    main()

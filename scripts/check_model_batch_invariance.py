"""Run a model adapter through the reusable batch-invariance harness."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_invariance.harness import DEFAULT_BATCH_SIZES, run_investigation


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        required=True,
        help="import path to a module exposing an `adapter` object",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="comma-separated sizes; must include 1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--batch-invariant-ops", action="store_true")
    args = parser.parse_args()

    module = importlib.import_module(args.adapter)
    adapter = module.adapter
    report = run_investigation(
        adapter,
        batch_sizes=args.batch_sizes,
        invariant_ops=args.batch_invariant_ops,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["end_to_end_exact"] else 1)


if __name__ == "__main__":
    main()

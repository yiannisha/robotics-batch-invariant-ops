"""Check fixed WSA invariance for several real LIBERO targets."""

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


def _run(adapter, policy, samples, seed):
    seed_everything(seed)
    policy.reset()
    with torch.inference_mode():
        return adapter.run_model(policy, adapter.compose_batch(samples)).detach().cpu().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="scripts.model_invariance.adapters.wsa_base_libero",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    examples = list(adapter.load_example_inputs(48))
    target_indices = (0, 17, 42)
    checks = []
    with set_batch_invariant_mode():
        for target_index in target_indices:
            target = examples[target_index]
            reference = _run(adapter, policy, [target], args.seed)
            for batch_size in (2, 4):
                duplicate = _run(adapter, policy, [target] * batch_size, args.seed)[:1]
                companions = [
                    examples[(target_index + offset + 1) % len(examples)]
                    for offset in range(batch_size - 1)
                ]
                unrelated = _run(adapter, policy, [target, *companions], args.seed)[:1]
                for composition, candidate in (
                    ("duplicate", duplicate),
                    ("unrelated", unrelated),
                ):
                    comparison = compare_outputs(reference, candidate)[0]
                    checks.append(
                        {
                            "target_index": target_index,
                            "batch_size": batch_size,
                            "composition": composition,
                            **asdict(comparison),
                        }
                    )

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "targets": "three real public LIBERO episode-zero observations",
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

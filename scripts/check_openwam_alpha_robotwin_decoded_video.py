"""Check decoded OpenWAM pixels before and after invariant operators."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
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
        pixels = adapter.decode_video_latents(model, output["video_latent"])
    return {"actions": output["actions"], "decoded_video": pixels}


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

    os.environ["OPENWAM_KEEP_PREPROCESSORS"] = "1"
    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model().eval()
    target, companion = adapter.load_example_inputs(2)
    report = {
        "model": adapter.name,
        "seed": args.seed,
        "surface": "physical actions and released Wan2.2 VAE uint8 pixels",
        "modes": {},
    }
    for name, mode in (
        ("baseline", contextlib.nullcontext),
        ("fixed", set_batch_invariant_mode),
    ):
        checks = []
        with mode():
            reference = _run(adapter, model, [target], args.seed)
            for composition, samples in (
                ("duplicate", [target, target]),
                ("unrelated", [target, companion]),
            ):
                candidate = _first(_run(adapter, model, samples, args.seed))
                comparisons = compare_outputs(reference, candidate)
                checks.append(
                    {
                        "batch_size": 2,
                        "composition": composition,
                        "exact": all(item.exact for item in comparisons),
                        "tensors": [asdict(item) for item in comparisons],
                    }
                )
        report["modes"][name] = {
            "all_exact": all(check["exact"] for check in checks),
            "checks": checks,
        }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["modes"]["fixed"]["all_exact"] else 1)


if __name__ == "__main__":
    main()

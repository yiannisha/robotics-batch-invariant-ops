"""Compare the SmolVLA adapter with the released ``sample_actions`` path."""

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

from scripts.model_invariance.harness import (
    _compose_batch,
    compare_outputs,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", default="scripts.model_invariance.adapters.lerobot_smolvla"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    policy = adapter.load_model().eval()
    batch = _compose_batch(adapter, adapter.load_example_inputs(1))

    with torch.inference_mode():
        adapter.reset_model(policy)
        adapted = adapter.run_model(policy, batch).detach()

        adapter.reset_model(policy)
        model_batch = {
            key: value.to(adapter.device) for key, value in batch.items() if key != "noise"
        }
        images, image_masks = policy.prepare_images(model_batch)
        state = policy.prepare_state(model_batch)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        official = policy.model.sample_actions(
            images,
            image_masks,
            model_batch[OBS_LANGUAGE_TOKENS],
            model_batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            noise=batch["noise"].to(adapter.device),
        )
        official = official[..., : policy.config.action_feature.shape[0]]
        official = adapter.postprocessor(official.detach().cpu()).to(adapter.device)

    comparisons = compare_outputs(official, adapted)
    report = {
        "model": adapter.name,
        "comparison": "adapter singleton versus released sample_actions singleton",
        "seed": args.seed,
        "exact": all(item.exact for item in comparisons),
        "tensors": [asdict(item) for item in comparisons],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["exact"] else 1)


if __name__ == "__main__":
    main()

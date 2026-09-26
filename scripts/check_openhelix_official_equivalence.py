"""Verify explicit-noise sampling against OpenHelix's released sampler."""

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
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.openhelix_calvin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    adapter = importlib.import_module(args.adapter).adapter
    models = adapter.load_model().eval()
    sample = adapter.load_example_inputs(1)[0]
    batch = adapter.compose_batch([sample])

    with torch.inference_mode():
        planner_features = adapter._planner_features(
            models, batch["planner_images"], batch["instructions"]
        ).float()
        rgb = batch["rgb"].to("cuda", dtype=torch.float32)[..., 20:180, 20:180]
        pcd = batch["pcd"].to("cuda", dtype=torch.float32)[..., 20:180, 20:180]
        proprio = batch["proprio"].to("cuda", dtype=torch.float32)[:, None]
        relative_pcd, relative_proprio = models.policy.convert2rel(pcd, proprio)
        relative_pcd = torch.permute(
            models.policy.normalize_pos(torch.permute(relative_pcd.clone(), [0, 1, 3, 4, 2])),
            [0, 1, 4, 2, 3],
        )
        relative_proprio = relative_proprio.clone()
        relative_proprio[..., :3] = models.policy.normalize_pos(relative_proprio[..., :3])
        relative_proprio = models.policy.convert_rot(relative_proprio)
        fixed_inputs = models.policy.encode_inputs(
            rgb, relative_pcd, planner_features, relative_proprio
        )

        condition = torch.zeros_like(batch["noise"], device="cuda")
        mask = torch.zeros_like(condition, dtype=torch.bool)
        torch.cuda.manual_seed(91_000)
        official = models.policy.conditional_sample(condition, mask, fixed_inputs)
        explicit = adapter._sample_policy(
            models.policy,
            fixed_inputs,
            batch["noise"].to("cuda"),
            batch["position_step_noise"].to("cuda"),
            batch["rotation_step_noise"].to("cuda"),
        )

    comparison = compare_outputs(official.cpu(), explicit.cpu())[0]
    report = {
        "model": adapter.name,
        "reference": "released DiffuserActorACTS.conditional_sample",
        "candidate": "adapter explicit per-sample noise path",
        "cuda_seed": 91_000,
        "diffusion_steps": 25,
        "exact": comparison.exact,
        "comparison": asdict(comparison),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if comparison.exact else 1)


if __name__ == "__main__":
    main()

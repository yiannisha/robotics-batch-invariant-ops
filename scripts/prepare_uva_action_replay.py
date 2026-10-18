"""Freeze UVA's real 16-frame action replay and three explicit noises/request.

The joint video/action path additionally needs 100 full video-diffusion noises
per request and is deliberately not claimed by this action-only manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import dill
import numpy as np
import torch


VAE_SHAPE = (4, 16, 16, 16)
ACTION_INITIAL_SHAPE = (16, 10)
ACTION_STEP_SHAPE = (100, 16, 10)
WIDTHS = (np.prod(VAE_SHAPE), np.prod(ACTION_INITIAL_SHAPE), np.prod(ACTION_STEP_SHAPE))
NOISE_WIDTH = int(sum(WIDTHS))
BASE_SEED = 188_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--context-inventory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--noise-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.noise_file.exists():
        raise FileExistsError("UVA action replay or noise already exists; preserve frozen evidence")
    source = json.loads(args.source_manifest.read_text())
    inventory = json.loads(args.context_inventory.read_text())
    rows = source.get("observations", [])
    windows = inventory.get("windows", [])
    if (len(rows) != 1000 or len(windows) != 1000
            or inventory.get("source_replay_manifest_sha256") != sha256(args.source_manifest)
            or any(row["id"] != window["id"] or row["replay_index"] != window["replay_index"]
                   for row, window in zip(rows, windows))):
        raise ValueError("UVA real-context inventory differs from the pinned 1,000-observation source")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True,
                            pickle_module=dill, weights_only=False)
    config = checkpoint["cfg"].model.policy.autoregressive_model_params
    if str(config.act_diff_testing_steps) != "100" or str(config.num_sampling_steps) != "100":
        raise ValueError("UVA checkpoint sampling schedules differ from pinned 100-step shapes")
    del checkpoint
    args.noise_file.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".uva_action_noise_", suffix=".npy", dir=args.noise_file.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    observations = []
    try:
        noises = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                           shape=(1000, 3, NOISE_WIDTH))
        for index, (source_row, window) in enumerate(zip(rows, windows)):
            row = {key: value for key, value in source_row.items()
                   if key not in ("noise", "request_ids")}
            row["context_window"] = window
            row["noise"] = []
            row["request_ids"] = []
            for noise_id in range(3):
                seed = BASE_SEED + index * 3 + noise_id
                generator = torch.Generator(device="cpu").manual_seed(seed)
                parts = (
                    torch.randn(VAE_SHAPE, generator=generator, dtype=torch.float32),
                    torch.randn(ACTION_INITIAL_SHAPE, generator=generator, dtype=torch.float32),
                    torch.randn(ACTION_STEP_SHAPE, generator=generator, dtype=torch.float32),
                )
                offset = 0
                for part in parts:
                    width = part.numel()
                    noises[index, noise_id, offset : offset + width] = part.numpy().reshape(-1)
                    offset += width
                noise_hash = hashlib.sha256(noises[index, noise_id].tobytes()).hexdigest()
                row["noise"].append({"noise_id": noise_id, "seed": seed, "sha256": noise_hash})
                row["request_ids"].append(f"{row['id']}-noise-{noise_id}")
            observations.append(row)
        noises.flush()
        del noises
        os.rename(temporary, args.noise_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        key: value for key, value in source.items()
        if key not in ("observations", "noise_file", "noise_file_sha256", "noise_shape",
                       "checkpoint_config_sha256")
    }
    manifest.update({
        "model_id": "uva",
        "source_replay_manifest_path": str(args.source_manifest.resolve()),
        "source_replay_manifest_sha256": sha256(args.source_manifest),
        "context_inventory_path": str(args.context_inventory.resolve()),
        "context_inventory_sha256": sha256(args.context_inventory),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "noise_file": str(args.noise_file.resolve()),
        "noise_file_sha256": sha256(args.noise_file),
        "noise_shape": [1000, 3, NOISE_WIDTH],
        "noise_layout": {
            "vae_noise": {"offset": 0, "shape": list(VAE_SHAPE)},
            "action_initial_noise": {"offset": int(WIDTHS[0]), "shape": list(ACTION_INITIAL_SHAPE)},
            "action_step_noise": {"offset": int(WIDTHS[0] + WIDTHS[1]),
                                  "shape": list(ACTION_STEP_SHAPE)},
        },
        "observations": observations,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model_id": "uva", "observations": 1000, "requests": 3000,
                      "noise_shape": manifest["noise_shape"],
                      "noise_bytes": args.noise_file.stat().st_size,
                      "replay_manifest_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()

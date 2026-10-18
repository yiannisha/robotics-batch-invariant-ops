"""Attach checkpoint-shaped explicit noises to a pinned real-observation replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--noise-shape", type=int, nargs="+", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--noise-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.noise_output.exists():
        raise FileExistsError("derived replay or noise output already exists")
    if not args.noise_shape or any(size < 1 for size in args.noise_shape):
        raise ValueError("noise shape must have positive dimensions")
    source = json.loads(args.source_manifest.read_text())
    if len(source.get("observations", [])) != 1000:
        raise ValueError("source replay needs 1,000 frozen real observations")
    checkpoint = json.loads(args.checkpoint_config.read_text())
    temporal = checkpoint.get("chunk_size", checkpoint.get("action_horizon"))
    if temporal is not None and args.noise_shape[0] != temporal:
        raise ValueError("noise temporal dimension differs from checkpoint action horizon")
    if checkpoint.get("max_action_dim") is not None and args.noise_shape[-1] != checkpoint["max_action_dim"]:
        raise ValueError("noise action dimension differs from checkpoint max_action_dim")
    args.noise_output.parent.mkdir(parents=True, exist_ok=True)
    noise = np.lib.format.open_memmap(
        args.noise_output, mode="w+", dtype=np.float32,
        shape=(1000, 3, *args.noise_shape),
    )
    observations = []
    for index, source_row in enumerate(source["observations"]):
        row = {key: value for key, value in source_row.items() if key not in ("noise", "request_ids")}
        noise_records = []
        for noise_id in range(3):
            seed = args.seed_base + index * 3 + noise_id
            sample = np.random.default_rng(seed).standard_normal(tuple(args.noise_shape), dtype=np.float32)
            noise[index, noise_id] = sample
            noise_records.append({
                "noise_id": noise_id,
                "seed": seed,
                "sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
            })
        row["noise"] = noise_records
        row["request_ids"] = [f"{row['id']}-{args.model_id}-noise-{noise_id}" for noise_id in range(3)]
        observations.append(row)
    noise.flush()
    del noise
    manifest = {
        key: value for key, value in source.items()
        if key not in (
            "observations", "noise_file", "noise_file_sha256", "noise_shape",
            "checkpoint_config_sha256",
        )
    }
    manifest.update({
        "model_id": args.model_id,
        "source_replay_manifest_path": str(args.source_manifest.resolve()),
        "source_replay_manifest_sha256": sha256(args.source_manifest),
        "source_variant_note": "Public HuggingFaceVLA/libero embedded-image frames; a model's other released data encoding is not claimed pixel-identical.",
        "checkpoint_config_sha256": sha256(args.checkpoint_config),
        "noise_file": str(args.noise_output.resolve()),
        "noise_file_sha256": sha256(args.noise_output),
        "noise_shape": [1000, 3, *args.noise_shape],
        "observations": observations,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model_id": args.model_id, "observations": 1000, "noise_shape": manifest["noise_shape"]}))


if __name__ == "__main__":
    main()

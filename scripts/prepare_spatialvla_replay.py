"""Freeze SpatialVLA's deterministic 1,000-observation real-image replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source_manifest.read_text())
    rows = source.get("observations") or []
    if (len(rows) != 1000
            or sum(row.get("split") == "diagnostic" for row in rows) != 200
            or sum(row.get("split") == "heldout" for row in rows) != 800):
        raise ValueError("SpatialVLA source replay is not the frozen 200/800 corpus")
    required_checkpoint = [
        "config.json", "model.safetensors.index.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "processor_config.json", "processing_spatialvla.py",
        "modeling_spatialvla.py", "action_tokenizer.py",
    ]
    missing = [name for name in required_checkpoint if not (args.checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"SpatialVLA checkpoint is incomplete: {missing}")
    if not args.adapter_source.is_file():
        raise FileNotFoundError(args.adapter_source)
    observations = []
    for source_row in rows:
        row = {
            key: value for key, value in source_row.items()
            if key not in ("noise", "request_ids")
        }
        row["request_ids"] = [f"{row['id']}-deterministic"]
        row["policy_setup"] = "widowx_bridge"
        row["unnorm_key"] = "bridge_orig/1.0.0"
        observations.append(row)
    manifest = {
        key: value for key, value in source.items()
        if key not in (
            "observations", "noise_file", "noise_file_sha256", "noise_shape",
            "checkpoint_config_sha256",
        )
    }
    manifest.update({
        "model_id": "spatialvla",
        "source_replay_manifest_path": str(args.source_manifest.resolve()),
        "source_replay_manifest_sha256": sha256(args.source_manifest),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_index_sha256": sha256(args.checkpoint / "model.safetensors.index.json"),
        "checkpoint_weight_shards": [
            str((args.checkpoint / name).resolve())
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        ],
        "checkpoint_weight_shards_sha256": [
            sha256(args.checkpoint / name)
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        ],
        "deterministic_decoding_contract": {
            "reason": (
                "The exact SpatialVLA policy uses greedy generation (do_sample=False); "
                "there is no sampled policy latent or request-local RNG input."
            ),
            "evidence_path": str(args.adapter_source.resolve()),
            "evidence_sha256": sha256(args.adapter_source),
            "max_new_tokens": 256,
            "do_sample": False,
        },
        "observations": observations,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "model_id": "spatialvla", "observations": len(observations),
        "requests": sum(len(row["request_ids"]) for row in observations),
        "replay_manifest_sha256": sha256(args.output),
    }))


if __name__ == "__main__":
    main()

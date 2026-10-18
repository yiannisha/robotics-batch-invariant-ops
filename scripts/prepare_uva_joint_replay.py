"""Freeze UVA joint replay with stored action and regenerated video noises.

The 19.7 GB of video diffusion draws are represented without loss by a pinned
CPU ``torch.Generator`` contract and one unique seed per request.  The action
noises remain byte-for-byte backed by the audited UVA action tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ACTION_WIDTH = 32544
TOTAL_WIDTH = 1687328
VIDEO_SEED_BASE = 288000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    action = json.loads(args.action_manifest.read_text())
    observations = action.get("observations") or []
    action_noise_path = Path(action.get("noise_file", ""))
    if (action.get("model_id") != "uva" or len(observations) != 1000
            or action.get("noise_shape") != [1000, 3, ACTION_WIDTH]
            or not action_noise_path.is_file()
            or sha256(action_noise_path) != action.get("noise_file_sha256")):
        raise ValueError("UVA action replay/noise is absent or changed")
    action_noise = np.load(action_noise_path, mmap_mode="r", allow_pickle=False)
    if list(action_noise.shape) != [1000, 3, ACTION_WIDTH] or action_noise.dtype != np.float32:
        raise ValueError("UVA action noise has another shape or dtype")
    contract = {
        "schema_version": 1,
        "algorithm": "torch_cpu_generator_manual_seed_randn",
        "torch_version": "2.8.0+cu128",
        "device": "cpu", "dtype": "float32",
        "seed_base": VIDEO_SEED_BASE,
        "seed_formula": "seed_base + replay_index * 3 + noise_id",
        "draw_order": ["video_initial_noise", "video_step_noise"],
        "components": {
            "video_initial_noise": [1024, 16],
            "video_step_noise": [100, 1024, 16],
        },
    }
    contract_hash = canonical_sha256(contract)
    joint_observations = []
    for row in observations:
        replay_index = row["replay_index"]
        result = {key: value for key, value in row.items() if key != "noise"}
        result["noise"] = []
        for noise_id in range(3):
            stored_hash = hashlib.sha256(
                np.asarray(action_noise[replay_index, noise_id]).tobytes()
            ).hexdigest()
            if stored_hash != row["noise"][noise_id]["sha256"]:
                raise ValueError(f"stored UVA action noise changed at {replay_index}/{noise_id}")
            video_contract = {
                "contract_sha256": contract_hash,
                "replay_index": replay_index, "noise_id": noise_id,
                "seed": VIDEO_SEED_BASE + replay_index * 3 + noise_id,
            }
            video_hash = canonical_sha256(video_contract)
            result["noise"].append({
                "noise_id": noise_id,
                "stored_action_sha256": stored_hash,
                "video_seed": video_contract["seed"],
                "video_contract_sha256": video_hash,
                "sha256": canonical_sha256({
                    "stored_action_sha256": stored_hash,
                    "video_contract_sha256": video_hash,
                }),
            })
        joint_observations.append(result)
    manifest = {
        key: value for key, value in action.items()
        if key not in (
            "model_id", "observations", "noise_file", "noise_file_sha256",
            "noise_shape", "noise_layout",
        )
    }
    manifest.update({
        "model_id": "uva_joint",
        "source_action_manifest_path": str(args.action_manifest.resolve()),
        "source_action_manifest_sha256": sha256(args.action_manifest),
        "stored_action_noise_file": str(action_noise_path.resolve()),
        "stored_action_noise_file_sha256": sha256(action_noise_path),
        "stored_action_noise_shape": [1000, 3, ACTION_WIDTH],
        "noise_regeneration_contract": contract,
        "noise_regeneration_contract_sha256": contract_hash,
        "noise_shape": [1000, 3, TOTAL_WIDTH],
        "noise_layout": {
            "vae_noise": {"offset": 0, "shape": [4, 16, 16, 16]},
            "action_initial_noise": {"offset": 16384, "shape": [16, 10]},
            "action_step_noise": {"offset": 16544, "shape": [100, 16, 10]},
            "video_initial_noise": {"offset": 32544, "shape": [1024, 16]},
            "video_step_noise": {"offset": 48928, "shape": [100, 1024, 16]},
        },
        "observations": joint_observations,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "model_id": "uva_joint", "observations": 1000, "requests": 3000,
        "logical_noise_bytes": 1000 * 3 * TOTAL_WIDTH * 4,
        "stored_noise_bytes": action_noise_path.stat().st_size,
        "replay_manifest_sha256": sha256(args.output),
    }))


if __name__ == "__main__":
    main()

"""Derive MolmoAct2's full real-input replay from the pinned LIBERO freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


SOURCE_DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
CHECKPOINT_REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
CHECKPOINT_FILES = [
    ("model-00001-of-00005.safetensors", 4_919_324_120, "5aa56113a3c1c95f0c03b90bada5b7ac2babf59c752a3d6201b8362a530c6809"),
    ("model-00002-of-00005.safetensors", 4_844_690_992, "1fd95aec9ad144d3f4eac42db88a000321e016beeae606ecfcb84fcbdb13ecf4"),
    ("model-00003-of-00005.safetensors", 4_844_691_024, "e6c455434915e74a8d750bedd0fc4c23b5a8ce37954aaff9c3e3f22085a26890"),
    ("model-00004-of-00005.safetensors", 4_877_619_536, "4e5bcf5bad81573153d6deaa3d25c3e53dc697658488b1cdbab4c8a3dffba1b6"),
    ("model-00005-of-00005.safetensors", 2_282_630_240, "76140ad00c225f95b80eb6b6a5fea2f985e4fdde3356a3c67c93d87d083d4ee2"),
]
NOISE_SEED_BASE = 371_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path,
        default=Path("results/cross_model/smolvla/full/replay_manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/cross_model/molmoact2/full/replay_manifest.json"),
    )
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    rows = source.get("observations") or []
    if (source.get("dataset_revision") != SOURCE_DATASET_REVISION
            or len(rows) != 1000
            or source.get("split_counts") != {"diagnostic": 200, "heldout": 800}
            or len({row.get("id") for row in rows}) != 1000
            or len({(row.get("episode_id"), row.get("observation_index")) for row in rows}) != 1000):
        raise ValueError("pinned LIBERO source replay is incomplete or changed")
    fixture = Path(source["fixture_root"])
    noise_path = args.output.parent / "request_noise.npy"
    noise_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_noise = noise_path.with_name(noise_path.name + f".tmp.{os.getpid()}")
    noise = np.lib.format.open_memmap(
        temporary_noise, mode="w+", dtype=np.float32, shape=(1000, 3, 10, 32)
    )
    derived_rows = []
    for replay_index, row in enumerate(rows):
        if row.get("replay_index") != replay_index:
            raise ValueError("source LIBERO replay indices are noncanonical")
        image_hashes = {}
        for name, relative in row["images"].items():
            path = fixture / relative
            if not path.is_file() or sha256(path) != row["image_sha256"][name]:
                raise ValueError(f"source LIBERO image changed: {path}")
            image_hashes[name] = row["image_sha256"][name]
        noise_records = []
        for noise_id in range(3):
            seed = NOISE_SEED_BASE + replay_index * 3 + noise_id
            generator = torch.Generator(device="cpu").manual_seed(seed)
            value = torch.randn((10, 32), generator=generator, dtype=torch.float32).numpy()
            noise[replay_index, noise_id] = value
            noise_records.append({
                "noise_id": noise_id, "seed": seed,
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            })
        derived = dict(row)
        derived.update({
            "noise": noise_records,
            "request_ids": [f"{row['id']}-noise-{noise_id}" for noise_id in range(3)],
            "image_sha256": image_hashes,
        })
        derived_rows.append(derived)
    noise.flush()
    del noise
    os.replace(temporary_noise, noise_path)
    manifest = {
        "schema_version": 1,
        "model_id": "molmoact2",
        "source_replay_manifest_path": str(args.source_manifest.resolve()),
        "source_replay_manifest_sha256": sha256(args.source_manifest),
        "dataset": source["dataset"],
        "dataset_revision": SOURCE_DATASET_REVISION,
        "source_root": source["source_root"],
        "fixture_root": str(fixture.resolve()),
        "source_sha256": source["source_sha256"],
        "tasks_sha256": source["tasks_sha256"],
        "checkpoint": {
            "repository": "allenai/MolmoAct2-LIBERO",
            "revision": CHECKPOINT_REVISION,
            "index_sha256": "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06",
            "files": [
                {"path": name, "bytes": size, "sha256": digest}
                for name, size, digest in CHECKPOINT_FILES
            ],
        },
        "noise_file": str(noise_path.resolve()),
        "noise_file_sha256": sha256(noise_path),
        "noise_shape": [1000, 3, 10, 32],
        "noise_dtype": "float32",
        "noise_generation": {
            "algorithm": "torch_cpu_generator_manual_seed_randn",
            "torch_version": torch.__version__,
            "seed_base": NOISE_SEED_BASE,
            "seed_formula": "seed_base + replay_index * 3 + noise_id",
            "draw_shape": [10, 32],
        },
        "split_counts": {"diagnostic": 200, "heldout": 800},
        "unique_physical_observations": 1000,
        "execution": {
            "flow_steps": 10,
            "max_action_horizon": 10,
            "max_action_dim": 32,
            "stochastic_input": "explicit_initial_action_trajectory",
        },
        "companion_schedule": source["companion_schedule"],
        "observations": derived_rows,
    }
    write_json_atomic(args.output, manifest)
    print(json.dumps({
        "observations": len(derived_rows), "split_counts": manifest["split_counts"],
        "noise_shape": manifest["noise_shape"], "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

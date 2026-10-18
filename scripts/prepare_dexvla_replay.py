"""Freeze every unique observation in DexVLA's only public example episode.

The public ``lesjie/dexvla_example_data`` dataset contains two byte-identical
HDF5 members.  The retained member has 396 real three-camera observations, so
this path uses all 396 without duplication: 200 diagnostic and 196 held out.
Each observation receives three explicit 50x14 ScaleDP initial noises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


BASE_SEED = 20261015
NOISE_DOMAIN = 0xD3A1A
EXPECTED_SOURCE_SHA256 = "abd1a0681df1a14e1fc09a799a7b5965e4a73bc7f233e2bbd5441303adb4203a"
EXPECTED_SOURCE_BYTES = 1_095_683_384
EXPECTED_FRAMES = 396
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CHECKPOINT_FILES = (
    "config2.json",
    "dataset_stats.pkl",
    "preprocessor_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def observation_sha256(images: list[np.ndarray], qpos: np.ndarray) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(np.ascontiguousarray(image).tobytes())
    digest.update(np.ascontiguousarray(qpos).tobytes())
    return digest.hexdigest()


def split_membership(frame_count: int = EXPECTED_FRAMES) -> dict[int, str]:
    if frame_count < 200:
        raise ValueError("DexVLA source must have at least 200 diagnostic observations")
    permutation = np.random.default_rng(BASE_SEED).permutation(frame_count)
    diagnostic = {int(index) for index in permutation[:200]}
    return {
        index: "diagnostic" if index in diagnostic else "heldout"
        for index in range(frame_count)
    }


def companion_indices(target: int, frame_count: int = EXPECTED_FRAMES) -> list[int]:
    """Return 31 distinct, episode-spanning source frames for a target."""
    if frame_count < 32 or not 0 <= target < frame_count:
        raise ValueError("B32 coverage requires at least 32 source observations")
    offsets = [(slot * frame_count) // 32 for slot in range(1, 32)]
    selected = [(target + offset) % frame_count for offset in offsets]
    if len(selected) != 31 or len(set(selected)) != 31 or target in selected:
        raise RuntimeError("source-limited companion construction is not distinct")
    return selected


def request_noise(frame_index: int, noise_id: int) -> np.ndarray:
    seed = np.random.SeedSequence([BASE_SEED, NOISE_DOMAIN, frame_index, noise_id])
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(
        (50, 14), dtype=np.float32
    )


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def save_npy_atomic(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_source(episode: h5py.File) -> str:
    if episode["observations/qpos"].shape != (EXPECTED_FRAMES, 14):
        raise ValueError("DexVLA qpos source shape differs from the pinned public episode")
    for camera in CAMERAS:
        if episode[f"observations/images/{camera}"].shape != (
            EXPECTED_FRAMES,
            480,
            640,
            3,
        ):
            raise ValueError(f"DexVLA {camera} source shape differs")
    if episode["language_raw"].shape != (1,):
        raise ValueError("DexVLA language source shape differs")
    task = episode["language_raw"][0]
    task = task.decode() if isinstance(task, bytes) else str(task)
    if task != "Cook rice.":
        raise ValueError(f"unexpected DexVLA public task: {task!r}")
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.output_root / "replay_manifest.json"
    noise_path = args.output_root / "request_noise.npy"
    if manifest_path.exists() or noise_path.exists():
        raise FileExistsError("DexVLA replay outputs already exist")
    if (
        not args.source.is_file()
        or args.source.stat().st_size != EXPECTED_SOURCE_BYTES
        or sha256(args.source) != EXPECTED_SOURCE_SHA256
    ):
        raise ValueError("DexVLA source differs from the retained public HDF5 member")
    inventory = json.loads(args.source_inventory.read_text())
    if (
        inventory.get("unique_public_episodes") != 1
        or inventory.get("unique_public_observations") != EXPECTED_FRAMES
        or inventory.get("retained_source", {}).get("sha256") != EXPECTED_SOURCE_SHA256
        or len(inventory.get("archive_members", [])) != 2
        or len({row.get("sha256") for row in inventory["archive_members"]}) != 1
    ):
        raise ValueError("DexVLA source-limit inventory does not prove the 396-frame limit")
    if not (args.official_repository / ".git").exists():
        raise ValueError("DexVLA official repository checkout is missing Git provenance")
    repository_revision = subprocess.run(
        ["git", "-C", str(args.official_repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repository_revision != "fc21a822f4c774e242eb6f1ab4a235788de7aba9":
        raise ValueError("DexVLA official repository revision differs")

    checkpoint_hashes = {}
    for name in CHECKPOINT_FILES:
        path = args.checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_hashes[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    noises = np.empty((EXPECTED_FRAMES, 3, 50, 14), dtype=np.float32)
    splits = split_membership()
    observations = []
    physical_hashes = set()
    with h5py.File(args.source, "r") as episode:
        task = validate_source(episode)
        for frame_index in range(EXPECTED_FRAMES):
            images = [
                np.asarray(episode[f"observations/images/{camera}"][frame_index])
                for camera in CAMERAS
            ]
            qpos = np.asarray(episode["observations/qpos"][frame_index])
            physical_hash = observation_sha256(images, qpos)
            if physical_hash in physical_hashes:
                raise ValueError(f"duplicate physical observation at frame {frame_index}")
            physical_hashes.add(physical_hash)
            noise_records = []
            for noise_id in range(3):
                noises[frame_index, noise_id] = request_noise(frame_index, noise_id)
                noise_records.append(
                    {
                        "noise_id": noise_id,
                        "seed_domain": [BASE_SEED, NOISE_DOMAIN, frame_index, noise_id],
                        "sha256": array_sha256(noises[frame_index, noise_id]),
                    }
                )
            identifier = f"dexvla-episode-47-frame-{frame_index:03d}"
            observations.append(
                {
                    "id": identifier,
                    "replay_index": frame_index,
                    "task_index": 0,
                    "task": task,
                    "episode_id": "episode_47",
                    "observation_index": frame_index,
                    "source_frame_index": frame_index,
                    "split": splits[frame_index],
                    "state": qpos.tolist(),
                    "state_sha256": array_sha256(qpos),
                    "images": {
                        camera: (
                            f"episode_47.hdf5::observations/images/{camera}"
                            f"[{frame_index}]"
                        )
                        for camera in CAMERAS
                    },
                    "image_sha256": {
                        camera: array_sha256(image)
                        for camera, image in zip(CAMERAS, images)
                    },
                    "physical_observation_sha256": physical_hash,
                    "noise": noise_records,
                    "request_ids": [
                        f"{identifier}-noise-{noise_id}" for noise_id in range(3)
                    ],
                    "companion_replay_indices": companion_indices(frame_index),
                }
            )

    save_npy_atomic(noise_path, noises)
    manifest = {
        "schema_version": 1,
        "model_id": "dexvla",
        "dataset": "lesjie/dexvla_example_data",
        "dataset_revision": "5e7f53a7d7d5198ffe68664f0426d3e5e3d0d366",
        "source_root": str(args.source.parent.resolve()),
        "source_file": args.source.name,
        "source_sha256": {args.source.name: EXPECTED_SOURCE_SHA256},
        "source_bytes": EXPECTED_SOURCE_BYTES,
        "source_inventory_path": str(args.source_inventory.resolve()),
        "source_inventory_sha256": sha256(args.source_inventory),
        "source_limit": {
            "reason": "The two public HDF5 members are byte-identical; the retained member contains every 396 unique observation frames.",
            "unique_public_episodes": 1,
            "unique_public_observations": EXPECTED_FRAMES,
            "all_unique_observations_used": True,
            "no_observation_duplication": True,
        },
        "official_repository": str(args.official_repository.resolve()),
        "official_repository_revision": repository_revision,
        "checkpoint_root": str(args.checkpoint.resolve()),
        "checkpoint_repository": "kuromivv/DexVLA",
        "checkpoint_revision": "f99bdd408e7dadf7ff1b960431c15781ce761b52",
        "checkpoint_files": checkpoint_hashes,
        "fixture_root": str(args.output_root.resolve()),
        "noise_file": noise_path.name,
        "noise_file_sha256": sha256(noise_path),
        "noise_shape": list(noises.shape),
        "noise_dtype": str(noises.dtype),
        "noise_contract": "NumPy PCG64 request-local explicit ScaleDP initial noise",
        "split_seed": BASE_SEED,
        "episode_split_method": "source-limited deterministic frame split; the only public episode necessarily spans splits",
        "split_counts": {"diagnostic": 200, "heldout": 196},
        "observations": observations,
        "companion_schedule": {
            "version": "distinct_source_observations_v1",
            "max_companions": 31,
            "description": "Thirty-one distinct episode-spanning real frames; the public source has no second unique episode.",
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()

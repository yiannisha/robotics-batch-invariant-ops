"""Freeze 1,000 unique real CALVIN observations for full OpenHelix replay.

The pinned public CALVIN_ABC release is a split tar archive that is much
larger than the available workspace.  This script reads the first archive
part as a stream and stops after 1,000 individual transition files.  It
retains the original NPZ bytes, validates the exact tensors consumed by the
OpenHelix adapter, and freezes three complete request-local DDPM noise sets
per transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ID = "openhelix"
DATASET_REPOSITORY = "KyujinL/CALVIN_ABC"
DATASET_REVISION = "0874cd64f0df8dcc3a3534360ca448705e44e954"
ARCHIVE_FILENAME = "validation.tar.part-000"
ARCHIVE_BYTES = 21_474_836_480
ARCHIVE_SHA256 = "aa735af43b268677c94d2c8edb32f78586bcd129f86acd4857f793f306d1ec41"
SOURCE_REVISION = "d88867fa56977f401a7c89d62b9ebce87d8078f0"
CHECKPOINT_REPOSITORY = "OpenHelix/openhelix"
CHECKPOINT_REVISION = "1a6fabbe148af045ed0710d007ed24c36fa83a53"
INSTRUCTION = "push the sliding door to the right side"
BASE_SEED = 20261015
NOISE_DOMAIN = 0x0A11CE
OBSERVATIONS = 1_000
DIAGNOSTIC = 200
POSITION_STEPS = 25
TRAJECTORY_STEPS = 19

CHECKPOINT_FILES = [
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00000.safetensors", 2_105_645_528, "7bc815c581781b9d864ff22fbfd87f84d45adda0b283609d42f5716f6b933ac7"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00001.safetensors", 2_114_018_192, "6f015b79cea364ea356c0e11414e9dbe71f4736b2b5b8523119a34a087e0e758"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00002.safetensors", 2_114_034_864, "2ec780a890f2554385765fe3f4beedeb6b953d7eaf6f3887ad1535e56976433f"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00003.safetensors", 2_124_504_248, "6d552bc92746187c7d1614186e924dee8c3b456c06be03868da9d2ac07410128"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00004.safetensors", 2_057_395_512, "247bc9b65065446a596a3c431077bdbe4126681821334c2019369aaa65f49620"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00005.safetensors", 2_114_018_256, "4172759f8b2e57b1e0d3d8daaddea4b5489adfa546f07b4a7ce1b53db55f2e62"),
    ("prompt_tuning_aux/llava_ckpt_safetensors/model-00006.safetensors", 1_504_179_160, "ee2df39c2fc516148d2d82e108f66c189f2cd769fb4e42cc333e6dca23d87810"),
    ("prompt_tuning_aux/policy.pth", 445_004_793, "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638"),
]

EXPECTED_ARRAYS = {
    "rgb_static": ((200, 200, 3), np.dtype("uint8")),
    "rgb_gripper": ((84, 84, 3), np.dtype("uint8")),
    "depth_static": ((200, 200), np.dtype("float32")),
    "depth_gripper": ((84, 84), np.dtype("float32")),
    "robot_obs": ((15,), np.dtype("float64")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def archive_url() -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
        f"{DATASET_REVISION}/{ARCHIVE_FILENAME}?download=true"
    )


def stream_transitions(destination: Path, count: int) -> list[Path]:
    """Extract exactly ``count`` regular transition members without the full tar."""
    destination.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    request = urllib.request.Request(archive_url(), headers={"User-Agent": "batch-invariance-replay/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = re.fullmatch(r"validation/(episode_[0-9]{7}\.npz)", member.name)
                if match is None:
                    raise ValueError(f"unexpected regular archive member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                target = destination / match.group(1)
                temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if temporary.stat().st_size != member.size:
                    raise ValueError(f"short archive member: {member.name}")
                os.replace(temporary, target)
                paths.append(target)
                if len(paths) == count:
                    break
    if len(paths) != count:
        raise ValueError(f"archive supplied only {len(paths)} transition files")
    if len({path.name for path in paths}) != count:
        raise ValueError("archive selection contains duplicate transition names")
    return paths


def validate_transition(path: Path) -> tuple[dict[str, Any], str]:
    array_hashes: dict[str, str] = {}
    physical = hashlib.sha256()
    with np.load(path, allow_pickle=False) as item:
        for name, (shape, dtype) in EXPECTED_ARRAYS.items():
            if name not in item:
                raise ValueError(f"{path.name} lacks {name}")
            value = item[name]
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"{path.name} has invalid {name}: {value.shape} {value.dtype}")
            contiguous = np.ascontiguousarray(value)
            digest = array_sha256(contiguous)
            array_hashes[name] = digest
            physical.update(name.encode() + b"\0")
            physical.update(contiguous.tobytes())
    return array_hashes, physical.hexdigest()


def request_noise(replay_index: int, noise_id: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([
        BASE_SEED, NOISE_DOMAIN, replay_index, noise_id,
    ])))
    initial = generator.standard_normal((TRAJECTORY_STEPS, 9), dtype=np.float32)
    position = generator.standard_normal(
        (POSITION_STEPS, TRAJECTORY_STEPS, 3), dtype=np.float32
    )
    rotation = generator.standard_normal(
        (POSITION_STEPS, TRAJECTORY_STEPS, 6), dtype=np.float32
    )
    return np.concatenate((initial.ravel(), position.ravel(), rotation.ravel()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-repository", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.output_root / "replay_manifest.json"
    noise_path = args.output_root / "request_noise.npy"
    fixture_root = args.source_root / "validation"
    if manifest_path.exists() or noise_path.exists() or fixture_root.exists():
        raise FileExistsError("OpenHelix full replay outputs already exist")

    import subprocess

    revision = subprocess.run(
        ["git", "-C", str(args.official_repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"official source revision differs: {revision}")

    temporary_root = fixture_root.with_name(f"validation.tmp.{os.getpid()}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    noise = np.empty((OBSERVATIONS, 3, 4_446), dtype=np.float32)
    observations: list[dict[str, Any]] = []
    physical_hashes: set[str] = set()
    try:
        paths = stream_transitions(temporary_root, OBSERVATIONS)
        for replay_index, path in enumerate(paths):
            array_hashes, physical_hash = validate_transition(path)
            if physical_hash in physical_hashes:
                raise ValueError(f"duplicate physical observation: {path.name}")
            physical_hashes.add(physical_hash)
            noise_records = []
            for noise_id in range(3):
                value = request_noise(replay_index, noise_id)
                noise[replay_index, noise_id] = value
                noise_records.append({
                    "noise_id": noise_id,
                    "seed_domain": [BASE_SEED, NOISE_DOMAIN, replay_index, noise_id],
                    "sha256": array_sha256(value),
                })
            identifier = f"openhelix-calvin-{path.stem}"
            observations.append({
                "id": identifier,
                "replay_index": replay_index,
                "episode_id": path.stem,
                "observation_index": 0,
                "split": "diagnostic" if replay_index < DIAGNOSTIC else "heldout",
                "task": "open_drawer",
                "instruction": INSTRUCTION,
                "source_member": f"validation/{path.name}",
                "source_file": path.name,
                "source_bytes": path.stat().st_size,
                "source_sha256": sha256(path),
                "consumed_array_sha256": array_hashes,
                "image_sha256": {
                    "static": array_hashes["rgb_static"],
                    "gripper": array_hashes["rgb_gripper"],
                },
                "physical_observation_sha256": physical_hash,
                "noise": noise_records,
                "request_ids": [f"{identifier}-noise-{noise_id}" for noise_id in range(3)],
            })

        for row in observations:
            index = row["replay_index"]
            row["companion_replay_indices"] = [
                (index + offset) % OBSERVATIONS for offset in range(1, 32)
            ]

        os.replace(temporary_root, fixture_root)
        save_npy_atomic(noise_path, noise)
        source_files = [{
            "path": row["source_member"],
            "bytes": row["source_bytes"],
            "sha256": row["source_sha256"],
        } for row in observations]
        manifest = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "dataset": {
                "repository": DATASET_REPOSITORY,
                "revision": DATASET_REVISION,
                "archive_filename": ARCHIVE_FILENAME,
                "archive_bytes": ARCHIVE_BYTES,
                "archive_sha256": ARCHIVE_SHA256,
                "selection": "first_1000_regular_transition_members_in_tar_order",
                "streamed_prefix_only": True,
            },
            "official_repository": str(args.official_repository.resolve()),
            "official_repository_revision": SOURCE_REVISION,
            "fixture_root": str(fixture_root.resolve()),
            "source_files": source_files,
            "source_files_canonical_sha256": canonical_sha256(source_files),
            "split_counts": {"diagnostic": DIAGNOSTIC, "heldout": OBSERVATIONS - DIAGNOSTIC},
            "unique_physical_observations": len(physical_hashes),
            "noise_file": str(noise_path.resolve()),
            "noise_file_sha256": sha256(noise_path),
            "noise_dtype": "float32",
            "noise_shape": list(noise.shape),
            "noise_layout": {
                "initial": [0, 171, [19, 9]],
                "position_steps": [171, 1596, [25, 19, 3]],
                "rotation_steps": [1596, 4446, [25, 19, 6]],
            },
            "noise_generation": {
                "algorithm": "numpy.PCG64.standard_normal_float32",
                "base_seed": BASE_SEED,
                "domain": NOISE_DOMAIN,
            },
            "checkpoint": {
                "repository": CHECKPOINT_REPOSITORY,
                "revision": CHECKPOINT_REVISION,
                "files": [
                    {"path": name, "bytes": size, "sha256": digest}
                    for name, size, digest in CHECKPOINT_FILES
                ],
            },
            "companion_schedule": {
                "version": "distinct_source_observations_v1",
                "max_companions": 31,
            },
            "observations": observations,
        }
        write_json_atomic(manifest_path, manifest)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "observations": len(observations),
        "unique_physical_observations": len(physical_hashes),
        "source_bytes": sum(row["source_bytes"] for row in observations),
        "noise_bytes": noise_path.stat().st_size,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

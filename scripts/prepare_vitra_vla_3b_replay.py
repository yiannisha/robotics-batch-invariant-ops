"""Freeze 1,000 real observations from the pinned VITRA-TeleData release.

The public release contains 50 synchronized real-robot episodes.  This
preparation selects 20 distinct frames per episode, preserves the recorded
XHand state and camera intrinsics, and losslessly stores the official
short-side-224 image transform consumed by VITRA.  Ten whole episodes form
the 200-observation diagnostic split; the remaining 40 form the 800 held-out
split.  Three request-local DDIM noise tensors are frozen per observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import av
import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


BASE_SEED = 20261015
NOISE_DOMAIN = 0x5172A
EPISODES = 50
OBSERVATIONS_PER_EPISODE = 20
DATASET_REVISION = "fcd93409b47b01e6075b8d7ced28ecb8a0327964"
REPOSITORY_REVISION = "b35517202b39d32a753fdd42014b2cc3c41fab58"
EXPECTED_SOURCE_BYTES = 91_473_237
EXPECTED_SOURCE_AGGREGATE_SHA256 = (
    "11fcb3067b16a3c7f2fccdac0c5ca799667a1ef7f4318afd6f2db57aebdd7f5f"
)
EXPECTED_CHECKPOINT = {
    "repository": "microsoft/VITRA-VLA-3B",
    "revision": "4bd47d5be2f664b9ae9636094a07a20e905d1ec1",
    "filename": "vitra-vla-3b.pt",
    "bytes": 15_067_578_877,
    "sha256": "7f57dfd4451d68f41accab8a06bb388d6fdc2cce87ee8b31fafa7c8f01d83796",
}
DIAGNOSTIC_EPISODES = {0, 1, 2, 3, 4, 6, 8, 10, 12, 47}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def frame_indices(length: int) -> list[int]:
    selected = np.linspace(0, length - 1, OBSERVATIONS_PER_EPISODE, dtype=np.int64)
    result = [int(value) for value in selected]
    if len(result) != len(set(result)) or len(result) != OBSERVATIONS_PER_EPISODE:
        raise ValueError(f"episode with {length} frames cannot provide 20 unique frames")
    return result


def resize_short_side_to_target(image: Image.Image, target: int = 224) -> Image.Image:
    width, height = image.size
    if width < height:
        new_width = target
        new_height = int(height * target / width)
    else:
        new_height = target
        new_width = int(width * target / height)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def request_noise(replay_index: int, noise_id: int) -> np.ndarray:
    seed = np.random.SeedSequence([BASE_SEED, NOISE_DOMAIN, replay_index, noise_id])
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(
        (16, 192), dtype=np.float32
    )


def task_class(instruction: str) -> str:
    """Collapse the four released phrasings into their two physical phases."""
    return "pick_mango" if instruction.startswith("Pick up") else "place_mango"


def source_inventory(source_root: Path) -> list[dict[str, Any]]:
    expected_names = [
        f"{folder}/{index:06d}.{extension}"
        for folder, extension in (("annotation", "h5"), ("videos", "mp4"))
        for index in range(EPISODES)
    ]
    actual_names = sorted(
        path.relative_to(source_root).as_posix()
        for path in source_root.glob("*/*")
        if path.is_file()
    )
    if actual_names != sorted(expected_names):
        raise ValueError("VITRA-TeleData source is not the exact 50 paired-file release")
    rows = []
    for name in actual_names:
        path = source_root / name
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    if sum(row["bytes"] for row in rows) != EXPECTED_SOURCE_BYTES:
        raise ValueError("VITRA-TeleData byte count differs from the pinned release")
    if canonical_json_sha256(rows) != EXPECTED_SOURCE_AGGREGATE_SHA256:
        raise ValueError("VITRA-TeleData content hashes differ from the pinned release")
    return rows


def decode_selected_frames(video_path: Path, selected: list[int]) -> dict[int, Image.Image]:
    wanted = set(selected)
    decoded: dict[int, Image.Image] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded[index] = frame.to_image().convert("RGB")
    if set(decoded) != wanted:
        raise ValueError(f"video {video_path.name} did not decode every selected frame")
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--official-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.output_root / "replay_manifest.json"
    noise_path = args.output_root / "request_noise.npy"
    fixture_root = args.output_root / "fixture/images"
    if manifest_path.exists() or noise_path.exists() or fixture_root.exists():
        raise FileExistsError("VITRA full replay outputs already exist")

    revision = subprocess.run(
        ["git", "-C", str(args.official_repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != REPOSITORY_REVISION:
        raise ValueError("VITRA official repository revision differs")
    sources = source_inventory(args.source_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary_images = fixture_root.with_name(f"images.tmp.{os.getpid()}")
    temporary_images.mkdir(parents=True)
    noises = np.empty((1000, 3, 16, 192), dtype=np.float32)
    observations: list[dict[str, Any]] = []
    physical_hashes: set[str] = set()
    episode_heads: dict[int, int] = {}
    instruction_counts: Counter[str] = Counter()

    try:
        for episode_index in range(EPISODES):
            annotation_path = args.source_root / f"annotation/{episode_index:06d}.h5"
            video_path = args.source_root / f"videos/{episode_index:06d}.mp4"
            with h5py.File(annotation_path, "r") as episode:
                frame_count = int(episode["meta/frame_count"][()])
                fps = float(episode["meta/fps"][()])
                instruction = episode["meta/instruction"][()].decode("utf-8")
                video_name = episode["meta/video_path"][()].decode("utf-8")
                if video_name != video_path.name or fps != 30.0:
                    raise ValueError("VITRA episode metadata differs from its paired video")
                if bool(episode["meta/has_left"][()]) or not bool(
                    episode["meta/has_right"][()]
                ):
                    raise ValueError("pinned VITRA-TeleData is expected to be right-arm only")
                selected = frame_indices(frame_count)
                frames = decode_selected_frames(video_path, selected)
                if episode["state/right_hand_mount_pose_in_cam"].shape != (frame_count, 6):
                    raise ValueError("VITRA camera-space wrist pose shape differs")
                if episode["state/right_hand_joint"].shape != (frame_count, 12):
                    raise ValueError("VITRA XHand joint shape differs")
                intrinsics = np.asarray(
                    episode["observation/camera/intrinsics"], dtype=np.float64
                )
                if intrinsics.shape != (3, 3):
                    raise ValueError("VITRA camera intrinsics are not 3x3")
                instruction_counts[instruction] += 1

                for local_index, frame_index in enumerate(selected):
                    replay_index = len(observations)
                    if local_index == 0:
                        episode_heads[episode_index] = replay_index
                    source_image = frames[frame_index]
                    source_rgb = np.asarray(source_image, dtype=np.uint8)
                    resized = resize_short_side_to_target(source_image, 224)
                    resized_rgb = np.asarray(resized, dtype=np.uint8)
                    image_name = (
                        f"episode_{episode_index:06d}_frame_{frame_index:06d}.png"
                    )
                    image_path = temporary_images / image_name
                    resized.save(image_path, format="PNG")

                    pose_axis_angle = np.asarray(
                        episode["state/right_hand_mount_pose_in_cam"][frame_index],
                        dtype=np.float64,
                    )
                    joints = np.asarray(
                        episode["state/right_hand_joint"][frame_index], dtype=np.float64
                    )
                    pose_euler = pose_axis_angle.copy()
                    pose_euler[3:6] = Rotation.from_rotvec(
                        pose_axis_angle[3:6]
                    ).as_euler("xyz", degrees=False)
                    xhand_state = np.concatenate([pose_euler, joints]).astype(np.float32)
                    if xhand_state.shape != (18,) or not np.isfinite(xhand_state).all():
                        raise ValueError("VITRA transformed XHand state is invalid")

                    physical = hashlib.sha256()
                    physical.update(np.ascontiguousarray(source_rgb).tobytes())
                    physical.update(np.ascontiguousarray(pose_axis_angle).tobytes())
                    physical.update(np.ascontiguousarray(joints).tobytes())
                    physical_hash = physical.hexdigest()
                    if physical_hash in physical_hashes:
                        raise ValueError("duplicate physical VITRA observation")
                    physical_hashes.add(physical_hash)

                    noise_records = []
                    for noise_id in range(3):
                        noise = request_noise(replay_index, noise_id)
                        noises[replay_index, noise_id] = noise
                        noise_records.append({
                            "noise_id": noise_id,
                            "seed_domain": [
                                BASE_SEED,
                                NOISE_DOMAIN,
                                replay_index,
                                noise_id,
                            ],
                            "sha256": array_sha256(noise),
                        })
                    identifier = (
                        f"vitra-teledata-episode-{episode_index:06d}-"
                        f"frame-{frame_index:06d}"
                    )
                    observations.append({
                        "id": identifier,
                        "replay_index": replay_index,
                        "episode_index": episode_index,
                        "episode_id": f"episode_{episode_index:06d}",
                        "observation_index": local_index,
                        "source_frame_index": frame_index,
                        "split": (
                            "diagnostic"
                            if episode_index in DIAGNOSTIC_EPISODES
                            else "heldout"
                        ),
                        "instruction": instruction,
                        "task": task_class(instruction),
                        "prompt": f"Left hand: None. Right hand: {instruction}",
                        "image": f"fixture/images/{image_name}",
                        "source_image_shape_hwc": list(source_rgb.shape),
                        "source_image_rgb_sha256": array_sha256(source_rgb),
                        "model_image_shape_hwc": list(resized_rgb.shape),
                        "model_image_rgb_sha256": array_sha256(resized_rgb),
                        "model_image_png_sha256": sha256(image_path),
                        "camera_intrinsics": intrinsics.tolist(),
                        "camera_intrinsics_sha256": array_sha256(intrinsics),
                        "source_pose_axis_angle": pose_axis_angle.tolist(),
                        "source_joints": joints.tolist(),
                        "xhand_state": xhand_state.tolist(),
                        "xhand_state_sha256": array_sha256(xhand_state),
                        "physical_observation_sha256": physical_hash,
                        "noise": noise_records,
                        "request_ids": [
                            f"{identifier}-noise-{noise_id}" for noise_id in range(3)
                        ],
                    })

        for row in observations:
            other_task_episodes = [
                episode
                for episode in range(EPISODES)
                if episode != row["episode_index"]
                and task_class(
                    observations[episode_heads[episode]]["instruction"]
                ) != row["task"]
            ]
            same_task_episodes = [
                episode
                for episode in range(EPISODES)
                if episode != row["episode_index"]
                and task_class(
                    observations[episode_heads[episode]]["instruction"]
                ) == row["task"]
            ]
            other_offset = row["replay_index"] % len(other_task_episodes)
            same_offset = row["replay_index"] % len(same_task_episodes)
            ordered = (
                other_task_episodes[other_offset:]
                + other_task_episodes[:other_offset]
                + same_task_episodes[same_offset:]
                + same_task_episodes[:same_offset]
            )
            selected_episodes = ordered[:31]
            row["companion_replay_indices"] = [
                episode_heads[episode] for episode in selected_episodes
            ]

        if len(observations) != 1000 or len(physical_hashes) != 1000:
            raise RuntimeError("VITRA replay did not retain 1,000 unique observations")
        if sum(row["split"] == "diagnostic" for row in observations) != 200:
            raise RuntimeError("VITRA diagnostic split does not contain 200 observations")
        fixture_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_images, fixture_root)
    except BaseException:
        shutil.rmtree(temporary_images, ignore_errors=True)
        raise

    save_npy_atomic(noise_path, noises)
    manifest = {
        "schema_version": 1,
        "model_id": "vitra_vla_3b",
        "dataset_revision": DATASET_REVISION,
        "official_repository_revision": REPOSITORY_REVISION,
        "checkpoint_revision": EXPECTED_CHECKPOINT["revision"],
        "checkpoint_root": "/workspace/assets/vitra/checkpoint/checkpoints",
        "source": {
            "repository": "microsoft/VITRA-TeleData",
            "revision": DATASET_REVISION,
            "root": str(args.source_root.resolve()),
            "files": sources,
            "file_count": len(sources),
            "bytes": EXPECTED_SOURCE_BYTES,
            "aggregate_sha256": EXPECTED_SOURCE_AGGREGATE_SHA256,
            "episodes": EPISODES,
            "decoded_frames": sum(
                int(row["source_image_shape_hwc"][0] > 0) for row in observations
            ),
        },
        "official_repository": {
            "path": str(args.official_repository.resolve()),
            "revision": REPOSITORY_REVISION,
        },
        "checkpoint_expected": EXPECTED_CHECKPOINT,
        "checkpoint_files": {
            EXPECTED_CHECKPOINT["filename"]: {
                "bytes": EXPECTED_CHECKPOINT["bytes"],
                "sha256": EXPECTED_CHECKPOINT["sha256"],
            }
        },
        "fixture_root": str(args.output_root.resolve()),
        "image_transform": {
            "source_decode": "PyAV RGB24",
            "operation": "resize_short_side_to_target",
            "target": 224,
            "aspect_ratio": "preserved",
            "resampling": "PIL.Image.Resampling.LANCZOS",
            "storage": "lossless PNG",
        },
        "state_transform": {
            "source": "right_hand_mount_pose_in_cam + right_hand_joint",
            "source_dim": 18,
            "rotation_source": "axis-angle rotation vector",
            "rotation_model": "Euler xyz radians",
            "rotation_implementation": "scipy.spatial.transform.Rotation",
            "xhand_to_human": "official XHAND_HUMAN_MAPPING at adapter load time",
            "model_state_dim": 212,
        },
        "selection": {
            "method": "20 integer-linspace frames from each of 50 episodes",
            "observations_per_episode": OBSERVATIONS_PER_EPISODE,
            "diagnostic_episodes": sorted(DIAGNOSTIC_EPISODES),
            "diagnostic_observations": 200,
            "heldout_observations": 800,
            "instruction_episode_counts": dict(sorted(instruction_counts.items())),
        },
        "split_counts": {"diagnostic": 200, "heldout": 800},
        "companion_schedule": {
            "version": "unrelated_episode_heads_v1",
            "max_companions": 31,
            "selection": "all other-task episode heads before same-task episode heads",
        },
        "observations": observations,
        "unique_physical_observations": len(physical_hashes),
        "noise_file": str(noise_path.resolve()),
        "noise_shape": list(noises.shape),
        "noise_dtype": str(noises.dtype),
        "noise_file_sha256": sha256(noise_path),
        "base_seed": BASE_SEED,
        "noise_domain": NOISE_DOMAIN,
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "observations": len(observations),
        "diagnostic": 200,
        "heldout": 800,
        "noise_sha256": manifest["noise_file_sha256"],
        "fixture_bytes": sum(path.stat().st_size for path in fixture_root.glob("*.png")),
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

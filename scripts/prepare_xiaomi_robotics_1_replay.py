"""Freeze 1,000 real observations from 100 pinned RoboCasa episodes.

The selection uses the first 100 episodes at the pinned public dataset
revision, exactly ten distinct frames per episode.  Both tasks present in
those episodes are stratified 20/80 at the episode level.  Images and the
complete real 25-D recorded state are retained without synthetic values; the
adapter pads that real state to the model's official 60-D input shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


BASE_SEED = 20261015
NOISE_DOMAIN = 0xA10A1
EPISODES = 100
OBSERVATIONS_PER_EPISODE = 10
DATASET_REVISION = "1f57c08d41d503c7a2c645c4181bae4a6d6ab997"
REPOSITORY_REVISION = "0dd7aef8dc87296246aae812a1f59ccb708e5546"
CHECKPOINT_REVISION = "82b7c2210c1a39fca4a87e681cffbbed134ccde0"
EXPECTED_INFO = {
    "codebase_version": "v2.0",
    "robot_type": "panda",
    "total_episodes": 70_513,
    "total_frames": 19_636_669,
    "total_tasks": 340,
    "total_chunks": 71,
    "chunks_size": 1000,
    "fps": 10,
}
CAMERAS = ("image_left", "image_right", "wrist_image")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def request_noise(replay_index: int, noise_id: int) -> np.ndarray:
    seed = np.random.SeedSequence([BASE_SEED, NOISE_DOMAIN, replay_index, noise_id])
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(
        (10, 60), dtype=np.float32
    )


def frame_indices(length: int) -> list[int]:
    indexes = np.linspace(0, length - 1, OBSERVATIONS_PER_EPISODE, dtype=np.int64)
    result = [int(value) for value in indexes]
    if len(result) != len(set(result)) or len(result) != OBSERVATIONS_PER_EPISODE:
        raise ValueError(f"episode with {length} frames cannot supply ten unique observations")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.output_root / "replay_manifest.json"
    noise_path = args.output_root / "request_noise.npy"
    image_root = args.output_root / "fixture/images"
    if manifest_path.exists() or noise_path.exists() or image_root.exists():
        raise FileExistsError("Xiaomi full replay outputs already exist")

    info_path = args.source_root / "meta/info.json"
    episodes_path = args.source_root / "meta/episodes.jsonl"
    tasks_path = args.source_root / "meta/tasks.jsonl"
    info = json.loads(info_path.read_text())
    if any(info.get(key) != value for key, value in EXPECTED_INFO.items()):
        raise ValueError("RoboCasa public dataset metadata differs from the pinned revision")
    episode_metadata = read_jsonl(episodes_path)[:EPISODES]
    if [row.get("episode_index") for row in episode_metadata] != list(range(EPISODES)):
        raise ValueError("first 100 pinned episode metadata rows are not contiguous")
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(tasks_path)}
    if len(tasks) != EXPECTED_INFO["total_tasks"]:
        raise ValueError("RoboCasa task metadata is incomplete")

    by_task: dict[str, list[int]] = defaultdict(list)
    for row in episode_metadata:
        if len(row.get("tasks", [])) != 1:
            raise ValueError("selected episode does not have exactly one task")
        by_task[row["tasks"][0]].append(row["episode_index"])
    if sorted(map(len, by_task.values())) != [40, 60]:
        raise ValueError("pinned first-100 task distribution differs from 40/60")
    diagnostic_episodes = {
        episode
        for task_episodes in by_task.values()
        for episode in task_episodes[: len(task_episodes) // 5]
    }
    if len(diagnostic_episodes) != 20:
        raise RuntimeError("stratified diagnostic split is not 20 episodes")

    repository_revision = subprocess.run(
        ["git", "-C", str(args.official_repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repository_revision != REPOSITORY_REVISION:
        raise ValueError("Xiaomi official repository revision differs")

    checkpoint_files = {}
    for path in sorted(args.checkpoint.iterdir()):
        if path.is_file():
            checkpoint_files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    shards = [name for name in checkpoint_files if name.endswith(".safetensors")]
    if len(shards) != 3 or sum(checkpoint_files[name]["bytes"] for name in shards) != 10_106_433_336:
        raise ValueError("Xiaomi exact three-shard public checkpoint is incomplete")

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary_images = image_root.with_name(f"images.tmp.{os.getpid()}")
    temporary_images.mkdir(parents=True)
    noises = np.empty((1000, 3, 10, 60), dtype=np.float32)
    observations: list[dict[str, Any]] = []
    source_hashes = {}
    physical_hashes: set[str] = set()
    episode_heads: dict[int, int] = {}
    try:
        for episode_index, metadata in enumerate(episode_metadata):
            relative_source = f"data/chunk-000/episode_{episode_index:06d}.parquet"
            source = args.source_root / relative_source
            if not source.is_file():
                raise FileNotFoundError(source)
            source_hashes[relative_source] = sha256(source)
            table = pq.read_table(source)
            if table.num_rows != metadata["length"]:
                raise ValueError(f"episode {episode_index} length differs from metadata")
            task = metadata["tasks"][0]
            selected_frames = frame_indices(table.num_rows)
            rows = table.take(selected_frames).to_pylist()
            for local_index, (frame_index, row) in enumerate(zip(selected_frames, rows)):
                replay_index = len(observations)
                if local_index == 0:
                    episode_heads[episode_index] = replay_index
                if row["episode_index"] != episode_index or row["frame_index"] != frame_index:
                    raise ValueError("selected parquet row indices differ from metadata")
                if tasks[row["task_index"]] != task:
                    raise ValueError("selected parquet task differs from episode metadata")
                state = np.asarray(row["state"], dtype=np.float32)
                if state.shape != (25,):
                    raise ValueError("Xiaomi real state is not 25-D")
                image_paths = {}
                image_hashes = {}
                physical = hashlib.sha256()
                for camera in CAMERAS:
                    encoded = bytes(row[camera]["bytes"])
                    if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError("Xiaomi fixture image is not lossless PNG")
                    name = f"episode_{episode_index:06d}_frame_{frame_index:06d}_{camera}.png"
                    path = temporary_images / name
                    path.write_bytes(encoded)
                    image_paths[camera] = f"fixture/images/{name}"
                    image_hashes[camera] = bytes_sha256(encoded)
                    physical.update(encoded)
                physical.update(np.ascontiguousarray(state).tobytes())
                physical_hash = physical.hexdigest()
                if physical_hash in physical_hashes:
                    raise ValueError("duplicate physical Xiaomi observation")
                physical_hashes.add(physical_hash)
                noise_records = []
                for noise_id in range(3):
                    noises[replay_index, noise_id] = request_noise(replay_index, noise_id)
                    noise_records.append({
                        "noise_id": noise_id,
                        "seed_domain": [BASE_SEED, NOISE_DOMAIN, replay_index, noise_id],
                        "sha256": array_sha256(noises[replay_index, noise_id]),
                    })
                identifier = (
                    f"xiaomi-robocasa-episode-{episode_index:06d}-frame-{frame_index:06d}"
                )
                observations.append({
                    "id": identifier,
                    "replay_index": replay_index,
                    "task_index": int(row["task_index"]),
                    "task": task,
                    "episode_id": f"episode_{episode_index:06d}",
                    "observation_index": local_index,
                    "source_frame_index": frame_index,
                    "split": (
                        "diagnostic" if episode_index in diagnostic_episodes else "heldout"
                    ),
                    "state": state.tolist(),
                    "state_sha256": array_sha256(state),
                    "images": image_paths,
                    "image_sha256": image_hashes,
                    "physical_observation_sha256": physical_hash,
                    "noise": noise_records,
                    "request_ids": [
                        f"{identifier}-noise-{noise_id}" for noise_id in range(3)
                    ],
                })

        for row in observations:
            episode_index = int(row["episode_id"].split("_")[-1])
            other = [
                candidate
                for task, candidates in by_task.items()
                if task != row["task"]
                for candidate in candidates
            ]
            offset = episode_index % len(other)
            selected = (other[offset:] + other[:offset])[:31]
            row["companion_replay_indices"] = [episode_heads[index] for index in selected]

        image_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_images, image_root)
    except BaseException:
        shutil.rmtree(temporary_images, ignore_errors=True)
        raise

    if len(observations) != 1000 or len(physical_hashes) != 1000:
        raise RuntimeError("Xiaomi replay did not retain 1,000 unique observations")
    save_npy_atomic(noise_path, noises)
    manifest = {
        "schema_version": 1,
        "model_id": "xiaomi_robotics_1",
        "dataset": "daixianjie/robocasa_mg_lerobot",
        "dataset_revision": DATASET_REVISION,
        "dataset_inventory": EXPECTED_INFO,
        "source_root": str(args.source_root.resolve()),
        "source_sha256": source_hashes,
        "source_metadata": {
            str(path.relative_to(args.source_root)): sha256(path)
            for path in (info_path, episodes_path, tasks_path, args.source_root / "meta/stats.json")
        },
        "official_repository": str(args.official_repository.resolve()),
        "official_repository_revision": repository_revision,
        "checkpoint_root": str(args.checkpoint.resolve()),
        "checkpoint_repository": "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_files": checkpoint_files,
        "fixture_root": str(args.output_root.resolve()),
        "noise_file": noise_path.name,
        "noise_file_sha256": sha256(noise_path),
        "noise_shape": list(noises.shape),
        "noise_dtype": str(noises.dtype),
        "noise_contract": "NumPy PCG64 request-local explicit five-step flow initial noise; converted to bfloat16 at policy input",
        "state_contract": "complete real 25-D dataset state padded with zeros to the official 60-D policy input",
        "split_seed": BASE_SEED,
        "episode_split_method": "first 100 pinned episodes; per-task first 20 percent diagnostic and remaining 80 percent held out",
        "split_counts": {"diagnostic": 200, "heldout": 800},
        "observations": observations,
        "companion_schedule": {
            "version": "unrelated_episode_heads_v1",
            "max_companions": 31,
            "description": "Thirty-one distinct real episode heads from the other selected task.",
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()

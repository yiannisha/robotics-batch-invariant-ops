"""Freeze 100 LIBERO episodes x 10 frames with three explicit noises each.

The source parquet files must be downloaded from the pinned
HuggingFaceVLA/libero revision before this script runs. This script never
silently substitutes synthetic observations or noise for missing source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
CAMERAS = ("observation.images.image", "observation.images.image2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    source_files = sorted((args.source / "data" / "chunk-000").glob("file-*.parquet"))
    if not source_files:
        raise FileNotFoundError("no pinned LIBERO parquet shards found")
    if args.manifest_output.exists() or (args.output / "replay_manifest.json").exists():
        raise FileExistsError("refusing to overwrite an existing replay manifest")

    config = json.loads(args.checkpoint_config.read_text())
    chunk_size = int(config["chunk_size"])
    max_action_dim = int(config["max_action_dim"])
    task_rows = pq.read_table(args.tasks).to_pylist()
    tasks = {int(row["task_index"]): row["__index_level_0__"] for row in task_rows}

    episodes: dict[int, list[dict]] = defaultdict(list)
    source_hashes = {}
    for path in source_files:
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "task_index", "observation.state"]
        )
        for local_index, row in enumerate(table.to_pylist()):
            episode_id = int(row["episode_index"])
            episodes[episode_id].append(
                {
                    "parquet": path,
                    "local_index": local_index,
                    "frame_index": int(row["frame_index"]),
                    "task_index": int(row["task_index"]),
                    "state": row["observation.state"],
                }
            )
    eligible = sorted(key for key, rows in episodes.items() if len(rows) >= 10)
    if len(eligible) < 100:
        raise ValueError(f"need 100 complete source episodes; found {len(eligible)}")
    selected_episodes = eligible[:100]
    episodes_by_task: dict[int, list[int]] = defaultdict(list)
    for episode_id in selected_episodes:
        task_ids = {row["task_index"] for row in episodes[episode_id]}
        if len(task_ids) != 1:
            raise ValueError(f"episode {episode_id} spans multiple tasks: {task_ids}")
        episodes_by_task[next(iter(task_ids))].append(episode_id)
    if len(episodes_by_task) != 10 or any(len(values) < 2 for values in episodes_by_task.values()):
        raise ValueError("cannot make a stratified 20/80 split across 10 source tasks")
    rng = random.Random(20260929)
    diagnostic_episodes = set()
    for task_id in sorted(episodes_by_task):
        diagnostic_episodes.update(rng.sample(episodes_by_task[task_id], 2))
    if len(diagnostic_episodes) != 20:
        raise AssertionError("diagnostic episode count")

    selected: list[dict] = []
    for episode_id in selected_episodes:
        frames = sorted(episodes[episode_id], key=lambda row: row["frame_index"])
        positions = [round(index * (len(frames) - 1) / 9) for index in range(10)]
        if len(set(positions)) != 10:
            raise AssertionError(f"repeated frame position in episode {episode_id}")
        for position in positions:
            selected.append(frames[position] | {"episode_id": episode_id})
    if len(selected) != 1000:
        raise AssertionError("observation count")

    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "frames"
    images_dir.mkdir(parents=True, exist_ok=True)
    noise_path = args.output / "initial_action_noise.npy"
    if noise_path.exists():
        raise FileExistsError(noise_path)
    noise = np.lib.format.open_memmap(
        noise_path, mode="w+", dtype=np.float32,
        shape=(1000, 3, chunk_size, max_action_dim),
    )
    by_file: dict[Path, list[tuple[int, dict]]] = defaultdict(list)
    observations = []
    for index, row in enumerate(selected):
        by_file[row["parquet"]].append((index, row))
        identifier = f"libero-episode-{row['episode_id']:04d}-frame-{row['frame_index']:04d}"
        noise_records = []
        for noise_id in range(3):
            seed = 93_000 + row["episode_id"] * 100_000 + row["frame_index"] * 3 + noise_id
            sample = np.random.default_rng(seed).standard_normal(
                (chunk_size, max_action_dim), dtype=np.float32
            )
            noise[index, noise_id] = sample
            noise_records.append(
                {"noise_id": noise_id, "seed": seed, "sha256": hashlib.sha256(sample.tobytes()).hexdigest()}
            )
        observations.append(
            {
                "id": identifier,
                "replay_index": index,
                "task_index": row["task_index"],
                "task": tasks[row["task_index"]],
                "episode_id": row["episode_id"],
                "observation_index": row["frame_index"],
                "split": "diagnostic" if row["episode_id"] in diagnostic_episodes else "heldout",
                "state": row["state"],
                "source_parquet": str(row["parquet"].relative_to(args.source)),
                "source_row": row["local_index"],
                "images": {
                    "image": f"frames/{identifier}-image.png",
                    "image2": f"frames/{identifier}-image2.png",
                },
                "noise": noise_records,
                "request_ids": [f"{identifier}-noise-{noise_id}" for noise_id in range(3)],
            }
        )
    noise.flush()
    del noise

    for path, requests in sorted(by_file.items()):
        table = pq.read_table(path, columns=list(CAMERAS))
        for index, row in requests:
            for column, camera in zip(CAMERAS, ("image", "image2"), strict=True):
                payload = table.column(column)[row["local_index"]].as_py()["bytes"]
                if payload is None:
                    raise ValueError(f"missing {column} bytes in {path}:{row['local_index']}")
                image_path = args.output / observations[index]["images"][camera]
                if image_path.exists():
                    if hashlib.sha256(payload).hexdigest() != sha256(image_path):
                        raise ValueError(f"existing image differs from source: {image_path}")
                else:
                    image_path.write_bytes(payload)
                observations[index].setdefault("image_sha256", {})[camera] = hashlib.sha256(payload).hexdigest()
        source_hashes[str(path.relative_to(args.source))] = sha256(path)
        print(f"extracted {path.name}: {len(requests)} observations", flush=True)

    split_counts = {name: sum(item["split"] == name for item in observations) for name in ("diagnostic", "heldout")}
    if split_counts != {"diagnostic": 200, "heldout": 800}:
        raise AssertionError(split_counts)
    manifest = {
        "schema_version": 1,
        "dataset": "HuggingFaceVLA/libero",
        "dataset_revision": REVISION,
        "source_root": str(args.source.resolve()),
        "fixture_root": str(args.output.resolve()),
        "source_sha256": source_hashes,
        "tasks_sha256": sha256(args.tasks),
        "checkpoint_config_sha256": sha256(args.checkpoint_config),
        "noise_file": noise_path.name,
        "noise_file_sha256": sha256(noise_path),
        "noise_shape": [1000, 3, chunk_size, max_action_dim],
        "episode_split_seed": 20260929,
        "selected_episodes": selected_episodes,
        "diagnostic_episodes": sorted(diagnostic_episodes),
        "episode_split_method": "two episodes per task across ten tasks; all remaining episodes held out",
        "split_counts": split_counts,
        "observations": observations,
    }
    serialized = json.dumps(manifest, indent=2) + "\n"
    (args.output / "replay_manifest.json").write_text(serialized)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(serialized)
    print(json.dumps({"observations": len(observations), "episodes": 100, "splits": split_counts}))


if __name__ == "__main__":
    main()

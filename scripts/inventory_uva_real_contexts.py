"""Prove 1,000 distinct real 16-frame UVA contexts in the pinned LIBERO source.

This is source inventory, not a stochastic replay manifest or model result.
Frame membership is pinned by exact parquet hashes and per-window local row
numbers; no frame is synthesized or repeated to fill a short episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(source_manifest_path: Path) -> dict:
    source = json.loads(source_manifest_path.read_text())
    observations = source.get("observations", [])
    if len(observations) != 1000 or len({item["id"] for item in observations}) != 1000:
        raise ValueError("the pinned LIBERO replay does not contain 1,000 distinct anchors")
    source_root = Path(source["source_root"])
    files = sorted({item["source_parquet"] for item in observations})
    rows_by_episode = defaultdict(list)
    file_hashes = {}
    for relative in files:
        path = source_root / relative
        digest = sha256(path)
        if digest != source["source_sha256"][relative]:
            raise ValueError(f"source parquet differs from the pinned replay: {path}")
        file_hashes[relative] = digest
        metadata = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"]).to_pylist()
        for local_row, item in enumerate(metadata):
            rows_by_episode[int(item["episode_index"])].append({
                "source_parquet": relative,
                "source_row": local_row,
                "frame_index": int(item["frame_index"]),
                "task_index": int(item["task_index"]),
            })
    for episode, rows in rows_by_episode.items():
        rows.sort(key=lambda row: row["frame_index"])
        if [row["frame_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"episode {episode} lacks a contiguous real frame sequence")
        if len({row["source_parquet"] for row in rows}) != 1:
            raise ValueError(f"episode {episode} crosses parquet files")
    windows = []
    for observation in observations:
        episode = int(observation["episode_id"])
        frames = rows_by_episode[episode]
        if len(frames) < 16:
            raise ValueError(f"episode {episode} has fewer than 16 real frames")
        anchor = int(observation["observation_index"])
        anchor_row = frames[anchor]
        if (anchor_row["source_parquet"] != observation["source_parquet"]
                or anchor_row["source_row"] != observation["source_row"]
                or anchor_row["task_index"] != observation["task_index"]):
            raise ValueError(f"replay anchor differs from source row: {observation['id']}")
        start = min(anchor, len(frames) - 16)
        selected = frames[start : start + 16]
        if ([row["source_row"] for row in selected]
                != list(range(selected[0]["source_row"], selected[0]["source_row"] + 16))):
            raise ValueError(f"UVA context is not 16 contiguous source rows: {observation['id']}")
        windows.append({
            "id": observation["id"],
            "replay_index": observation["replay_index"],
            "episode_id": episode,
            "task_index": observation["task_index"],
            "split": observation["split"],
            "source_parquet": selected[0]["source_parquet"],
            "source_row_start": selected[0]["source_row"],
            "start_frame_index": start,
            "end_frame_index": start + 15,
            "anchor_frame_index": anchor,
        })
    if (len({(row["episode_id"], row["start_frame_index"]) for row in windows}) != 1000
            or Counter(row["split"] for row in windows) != {"diagnostic": 200, "heldout": 800}):
        raise ValueError("UVA source inventory duplicates a context or changes the split")
    episode_lengths = {episode: len(rows_by_episode[episode]) for episode in {row["episode_id"] for row in windows}}
    return {
        "schema_version": 1,
        "evidence_tier": "real_context_source_inventory_not_model_replay",
        "model_paths": ["uva", "uva_joint"],
        "source_replay_manifest_path": str(source_manifest_path.resolve()),
        "source_replay_manifest_sha256": sha256(source_manifest_path),
        "source_root": str(source_root.resolve()),
        "source_parquet_sha256": file_hashes,
        "selected_episodes": len(episode_lengths),
        "episode_min_frames": min(episode_lengths.values()),
        "episode_max_frames": max(episode_lengths.values()),
        "distinct_16_frame_contexts": len(windows),
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inventory(args.source_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "evidence_tier", "selected_episodes", "episode_min_frames",
        "episode_max_frames", "distinct_16_frame_contexts",
    )}))


if __name__ == "__main__":
    main()

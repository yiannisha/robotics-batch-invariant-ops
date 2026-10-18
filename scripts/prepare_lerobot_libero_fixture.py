"""Extract the first frames of a LeRobot LIBERO parquet into adapter fixtures."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq


IMAGE_COLUMNS = {
    "observation.images.image": "image",
    "observation.images.image2": "image2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()

    if args.frames < 1:
        raise ValueError("--frames must be positive")
    table = pq.read_table(
        args.parquet,
        columns=[*IMAGE_COLUMNS, "observation.state", "task_index"],
    )
    if len(table) < args.frames:
        raise ValueError(f"requested {args.frames} frames but parquet has {len(table)}")

    frames_dir = args.output / "frames"
    data_dir = args.output / "data" / "chunk-000"
    frames_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(table.slice(0, args.frames).to_pylist(), start=1):
        for column, prefix in IMAGE_COLUMNS.items():
            payload = row[column]
            if payload["bytes"] is None:
                raise ValueError(f"{column} row {index - 1} has no embedded bytes")
            (frames_dir / f"{prefix}_{index:03d}.png").write_bytes(payload["bytes"])

    shutil.copyfile(args.parquet, data_dir / "file-000.parquet")
    shutil.copyfile(args.tasks, args.output / "tasks.parquet")
    print(args.output)


if __name__ == "__main__":
    main()

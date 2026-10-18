"""Extract synchronized LeRobot video frames into the adapter fixture layout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _video(value: str) -> tuple[str, Path]:
    prefix, separator, raw_path = value.partition("=")
    if not separator or not prefix or not raw_path:
        raise argparse.ArgumentTypeError("videos must use PREFIX=PATH")
    return prefix, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--video", type=_video, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")

    frames_dir = args.output / "frames"
    data_dir = args.output / "data" / "chunk-000"
    frames_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for prefix, video in args.video:
        if not video.is_file():
            raise FileNotFoundError(video)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-frames:v",
                str(args.frames),
                "-vsync",
                "0",
                str(frames_dir / f"{prefix}_%03d.png"),
            ],
            check=True,
        )
    shutil.copyfile(args.parquet, data_dir / "file-000.parquet")
    shutil.copyfile(args.tasks, args.output / "tasks.parquet")
    print(args.output)


if __name__ == "__main__":
    main()

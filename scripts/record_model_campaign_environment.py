"""Record frozen source, runtime, and artifact provenance for a model campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import triton


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "commit": subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "-C", str(path), "status", "--short"], text=True
        ).splitlines(),
    }


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("artifacts must use NAME=PATH")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--artifact", type=_named_path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)

    artifacts = {}
    for name, path in args.artifact:
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "schema_version": 1,
        "model": args.model,
        "adapter": args.adapter,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "triton": triton.__version__,
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "driver": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "repositories": [_git(path) for path in args.source],
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

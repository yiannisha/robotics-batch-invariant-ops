"""Record a fail-closed Xiaomi-Robotics-1 RoboCasa evaluator preflight.

This does not substitute a synthetic environment for RoboCasa.  It records
whether the pinned official evaluator's Python modules and simulator assets are
present and whether the current filesystem can hold the documented assets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


MODULES = ("robocasa", "robosuite", "mujoco", "tyro", "transformers", "torch")
MINIMUM_ASSET_BYTES = 10_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--official-source",
        type=Path,
        default=Path("/workspace/models/xiaomi-robotics-1/eval_robocasa/main.py"),
    )
    parser.add_argument(
        "--official-readme",
        type=Path,
        default=Path("/workspace/models/xiaomi-robotics-1/eval_robocasa/README.md"),
    )
    args = parser.parse_args()

    for required in (args.official_source, args.official_readme):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    modules = {name: importlib.util.find_spec(name) is not None for name in MODULES}
    candidate_assets = [
        Path("/workspace/models/robocasa"),
        Path("/workspace/assets/robocasa"),
        Path("/workspace/models/isaac-groot/external_dependencies/robocasa"),
        Path("/workspace/models/isaac-groot/examples/robocasa"),
        Path("/workspace/models/cosmos-framework/cosmos_framework/simulation/robocasa"),
    ]
    assets = {
        str(path): {
            "exists": path.exists(),
            "is_runtime_checkout": (path / "robocasa" / "__init__.py").is_file(),
        }
        for path in candidate_assets
    }
    capacity = shutil.disk_usage("/workspace")
    missing_modules = [name for name, present in modules.items() if not present]
    runtime_checkout = any(item["is_runtime_checkout"] for item in assets.values())
    blockers = []
    if missing_modules:
        blockers.append("missing Python modules: " + ", ".join(missing_modules))
    if not runtime_checkout:
        blockers.append("no RoboCasa runtime checkout/assets found")
    if capacity.free < MINIMUM_ASSET_BYTES:
        blockers.append(
            f"workspace free bytes {capacity.free} are below documented asset floor {MINIMUM_ASSET_BYTES}"
        )

    report = {
        "schema_version": 1,
        "model_id": "xiaomi_robotics_1",
        "probe": "official_robocasa_environment_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "modules": modules,
        "asset_candidates": assets,
        "filesystem": {
            "path": "/workspace",
            "total_bytes": capacity.total,
            "used_bytes": capacity.used,
            "free_bytes": capacity.free,
            "required_asset_floor_bytes": MINIMUM_ASSET_BYTES,
        },
        "official_source_path": str(args.official_source),
        "official_source_sha256": sha256(args.official_source),
        "official_readme_path": str(args.official_readme),
        "official_readme_sha256": sha256(args.official_readme),
        "blockers": blockers,
        "runnable": not blockers,
    }
    stdout = json.dumps(report, sort_keys=True, indent=2) + "\n"
    stderr = "\n".join(blockers) + ("\n" if blockers else "")
    write_once(args.output_dir / "preflight.json", stdout)
    write_once(args.output_dir / "preflight.stdout.log", stdout)
    write_once(args.output_dir / "preflight.stderr.log", stderr)
    print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    raise SystemExit(0 if not blockers else 1)


if __name__ == "__main__":
    main()

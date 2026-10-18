"""Record the shared OpenWAM-Alpha checkpoint and RoboTwin preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT = (
    "checkpoint_step_118655.safetensors",
    24_813_767_464,
    "d07ff6f8cfb627ebf47313e65af38773fabcb3861d52fdc7c3bc5e7bee33f6fe",
)
SUPPORT = (
    ("config.yaml", "d69c342a2d7e67dbb44058b20bfd0b0e986e8885b9462f7c7e85f8658fb822b0"),
    ("normalization_stats.npy", "3eb52f759ebd004feda2772925d587054f9c2f05e1b979c3ff8c94fdd40e5db5"),
)


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
    parser.add_argument("--asset-root", type=Path, default=Path("/workspace/assets/openwam_alpha"))
    parser.add_argument(
        "--checkpoint-manifest", type=Path,
        default=Path("research/openwam_alpha_robotwin/checkpoint.txt"),
    )
    parser.add_argument(
        "--official-source", type=Path,
        default=Path("/workspace/models/openwam/README.md"),
    )
    parser.add_argument(
        "--official-evaluator", type=Path,
        default=Path("/workspace/models/openwam/benchmarks/robotwin/eval_policy_wrapper.py"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.checkpoint_manifest, args.official_source, args.official_evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)

    name, expected_size, expected_hash = CHECKPOINT
    checkpoint = args.asset_root / name
    present = checkpoint.is_file()
    artifact = {
        "name": name, "path": str(checkpoint), "size_bytes": expected_size,
        "sha256": expected_hash, "present": present,
        "resident_size_bytes": checkpoint.stat().st_size if present else None,
        "resident_sha256": sha256(checkpoint) if present else None,
    }
    support = []
    for relative, expected in SUPPORT:
        path = args.asset_root / relative
        actual = sha256(path) if path.is_file() else None
        support.append({
            "path": str(path), "expected_sha256": expected,
            "resident_sha256": actual, "exact": actual == expected,
        })

    capacity = shutil.disk_usage("/workspace")
    robotwin_roots = [Path("/workspace/RoboTwin"), Path("/workspace/models/RoboTwin"), Path("/workspace/models/robotwin")]
    robotwin_runtime = any(
        all((root / part).is_dir() for part in ("envs", "assets", "task_config", "script"))
        for root in robotwin_roots
    )
    robotwin_env = any(
        (Path("/workspace/envs") / name / "bin" / "python").is_file()
        for name in ("RoboTwin", "robotwin")
    )
    blockers = []
    if not present:
        blockers.append(f"exact OpenWAM checkpoint is absent ({expected_size} bytes)")
    elif artifact["resident_size_bytes"] != expected_size or artifact["resident_sha256"] != expected_hash:
        blockers.append("resident OpenWAM checkpoint differs from pinned release")
    if not all(item["exact"] for item in support):
        blockers.append("OpenWAM support metadata is missing or differs")
    if capacity.free < expected_size and not present:
        blockers.append(
            f"workspace free bytes {capacity.free} cannot hold checkpoint bytes {expected_size}"
        )
    if not robotwin_runtime:
        blockers.append("RoboTwin 2.0 simulator runtime/assets checkout is absent")
    if not robotwin_env:
        blockers.append("RoboTwin simulator Python environment is absent")

    report = {
        "schema_version": 1, "model_family": "openwam_alpha",
        "paths": ["openwam_alpha_robotwin", "openwam_alpha_video"],
        "probe": "exact_checkpoint_and_robotwin_runtime_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": artifact, "support_files": support,
        "filesystem": {
            "path": "/workspace", "total_bytes": capacity.total,
            "used_bytes": capacity.used, "free_bytes": capacity.free,
        },
        "robotwin": {
            "candidate_roots": [str(path) for path in robotwin_roots],
            "runtime_assets_present": robotwin_runtime,
            "python_environment_present": robotwin_env,
        },
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest.resolve()), "sha256": sha256(args.checkpoint_manifest),
        },
        "official_source": {"path": str(args.official_source), "sha256": sha256(args.official_source)},
        "official_evaluator": {"path": str(args.official_evaluator), "sha256": sha256(args.official_evaluator)},
        "blockers": blockers, "runnable": not blockers, "python": sys.version,
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

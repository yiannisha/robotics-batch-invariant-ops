"""Record the shared OpenDW-DW05 checkpoint and RoboTwin evaluator preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ARTIFACTS = (
    ("model.pt", 12_041_813_433, "4ea55d5fd73aceab9917886d51717b8bdac21854a77b98e12dfed8ee34164529"),
    ("text_encoder/model.pth", 11_361_920_418, "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"),
    ("vae/model.pth", 2_818_839_170, "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"),
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
    parser.add_argument("--asset-root", type=Path, default=Path("/workspace/assets/opendw_dw05"))
    parser.add_argument(
        "--checkpoint-manifest", type=Path,
        default=Path("research/opendw_dw05/checkpoint.txt"),
    )
    parser.add_argument(
        "--official-source", type=Path,
        default=Path("/workspace/models/opendw/README.md"),
    )
    parser.add_argument(
        "--official-evaluator", type=Path,
        default=Path("/workspace/models/opendw/playground/benchmarks/robotwin2/eval_dw05_single.py"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.checkpoint_manifest, args.official_source, args.official_evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)

    artifacts = []
    for relative, expected_size, expected_hash in ARTIFACTS:
        path = args.asset_root / relative
        present = path.is_file()
        artifacts.append({
            "name": relative, "path": str(path), "size_bytes": expected_size,
            "sha256": expected_hash, "present": present,
            "resident_size_bytes": path.stat().st_size if present else None,
            "resident_sha256": sha256(path) if present else None,
        })
    missing = [item for item in artifacts if not item["present"]]
    missing_bytes = sum(item["size_bytes"] for item in missing)
    capacity = shutil.disk_usage("/workspace")
    robotwin_roots = [Path("/workspace/RoboTwin"), Path("/workspace/models/RoboTwin"), Path("/workspace/models/robotwin")]
    robotwin_runtime = any(
        all((root / name).is_dir() for name in ("envs", "assets", "task_config", "script"))
        for root in robotwin_roots
    )
    robotwin_env = any(
        (Path("/workspace/envs") / name / "bin" / "python").is_file()
        for name in ("RoboTwin", "robotwin")
    )
    blockers = []
    if missing:
        blockers.append(f"{len(missing)} exact DW05 artifacts are absent ({missing_bytes} bytes)")
    if capacity.free < missing_bytes:
        blockers.append(
            f"workspace free bytes {capacity.free} cannot hold missing artifact bytes {missing_bytes}"
        )
    if not robotwin_runtime:
        blockers.append("RoboTwin 2.0 simulator runtime/assets checkout is absent")
    if not robotwin_env:
        blockers.append("RoboTwin simulator Python environment is absent")

    report = {
        "schema_version": 1, "model_family": "opendw_dw05",
        "paths": ["opendw_dw05", "opendw_dw05_joint"],
        "probe": "exact_checkpoint_and_robotwin_runtime_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "filesystem": {
            "path": "/workspace", "total_bytes": capacity.total,
            "used_bytes": capacity.used, "free_bytes": capacity.free,
            "missing_checkpoint_bytes": missing_bytes,
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

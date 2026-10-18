"""Record UniVLA LIBERO evaluator checkpoint and capacity readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ARTIFACTS = (
    ("policy/model-00001-of-00004.safetensors", 4_884_766_656, "b7e6e472d974d1c572138839008e308486b526ffe1c9065306bd2107513b9e80"),
    ("policy/model-00002-of-00004.safetensors", 4_999_819_320, "c700932f380da894537a191b1b60f335a617efa01dc44ae6d8216a4f4e97f28d"),
    ("policy/model-00003-of-00004.safetensors", 4_915_916_184, "eeef8be71a34052ff2a590a2120f5d4067114f4f2f72b3c9b78d771968b207c3"),
    ("policy/model-00004-of-00004.safetensors", 2_183_554_760, "0932b91cf0b322993213a8fc5f0eb591171a130addd93d2bff701a3dc5e4259f"),
    ("vision/model.safetensors", 1_083_015_124, "89536431c69b08b10b449ec309f52dcea22f14b7647317f30f5715273392bbf1"),
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
    parser.add_argument("--asset-root", type=Path, default=Path("/workspace/assets/univla_baai"))
    parser.add_argument(
        "--checkpoint-manifest", type=Path,
        default=Path("research/univla_baai/checkpoint.txt"),
    )
    parser.add_argument(
        "--official-source", type=Path,
        default=Path("/workspace/models/univla/docs/libero.md"),
    )
    parser.add_argument(
        "--official-evaluator", type=Path,
        default=Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/evaluate_libero_emu.py"),
    )
    parser.add_argument(
        "--libero-python", type=Path,
        default=Path("/workspace/envs/smolvla_eval/bin/python"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.checkpoint_manifest, args.official_source, args.official_evaluator, args.libero_python):
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
    runtime = subprocess.run(
        [str(args.libero_python), "-c", "import libero, robosuite, mujoco"],
        check=False, capture_output=True, text=True,
    )
    blockers = []
    if missing:
        blockers.append(f"{len(missing)} exact policy/vision artifacts are absent ({missing_bytes} bytes)")
    if capacity.free < missing_bytes:
        blockers.append(
            f"workspace free bytes {capacity.free} cannot hold missing artifact bytes {missing_bytes}"
        )
    if runtime.returncode != 0:
        blockers.append("resident LIBERO simulator runtime import failed")

    report = {
        "schema_version": 1,
        "model_id": "univla_baai",
        "probe": "exact_policy_vision_and_libero_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "filesystem": {
            "path": "/workspace", "total_bytes": capacity.total,
            "used_bytes": capacity.used, "free_bytes": capacity.free,
            "missing_checkpoint_bytes": missing_bytes,
        },
        "libero_runtime": {
            "python": str(args.libero_python), "import_exit_code": runtime.returncode,
            "stderr": runtime.stderr,
        },
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest.resolve()),
            "sha256": sha256(args.checkpoint_manifest),
        },
        "official_source": {
            "path": str(args.official_source), "sha256": sha256(args.official_source),
        },
        "official_evaluator": {
            "path": str(args.official_evaluator), "sha256": sha256(args.official_evaluator),
        },
        "blockers": blockers,
        "runnable": not blockers,
        "python": sys.version,
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

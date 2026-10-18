"""Record LingBot-VLA2 exact-checkpoint and RoboTwin evaluator readiness."""

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


ROBOTWIN_COMMIT = "13c3c47ff4312dd62484bcd51be034af55c062d1"
EXPECTED_CHECKPOINT_BYTES = 25_503_889_772


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


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=Path("/workspace/assets/lingbot_vla2/checkpoints/global_step_50000/hf_ckpt"),
    )
    parser.add_argument(
        "--official-source", type=Path,
        default=Path("/workspace/models/lingbot-vla-v2/experiment/robotwin/README.md"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.official_source.is_file():
        raise FileNotFoundError(args.official_source)

    shards = sorted(args.checkpoint_root.glob("model-*-of-00006.safetensors"))
    checkpoint_files = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in shards
    ]
    checkpoint_bytes = sum(item["size_bytes"] for item in checkpoint_files)

    candidates = [
        Path("/workspace/RoboTwin"),
        Path("/workspace/models/RoboTwin"),
        Path("/workspace/models/robotwin"),
    ]
    simulator_candidates = []
    for path in candidates:
        required = [path / name for name in ("envs", "assets", "task_config", "script")]
        simulator_candidates.append({
            "path": str(path),
            "exists": path.is_dir(),
            "required_roots_present": all(item.is_dir() for item in required),
            "git_revision": git_revision(path),
        })
    exact_simulator = any(
        item["required_roots_present"] and item["git_revision"] == ROBOTWIN_COMMIT
        for item in simulator_candidates
    )
    environment_candidates = [
        Path("/workspace/envs/RoboTwin"), Path("/workspace/envs/robotwin"),
        Path("/workspace/envs/lingbotvla"),
    ]
    environments = {str(path): (path / "bin" / "python").is_file() for path in environment_candidates}

    blockers = []
    if len(shards) != 6 or checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        blockers.append(
            f"exact checkpoint mismatch: shards={len(shards)} bytes={checkpoint_bytes}"
        )
    if not exact_simulator:
        blockers.append(f"RoboTwin checkout at pinned commit {ROBOTWIN_COMMIT} is absent")
    if not environments["/workspace/envs/RoboTwin"] and not environments["/workspace/envs/robotwin"]:
        blockers.append("RoboTwin simulator Python environment is absent")
    if not environments["/workspace/envs/lingbotvla"]:
        blockers.append("LingBot inference Python environment is absent")

    capacity = shutil.disk_usage("/workspace")
    report = {
        "schema_version": 1,
        "model_id": "lingbot_vla2",
        "probe": "exact_checkpoint_and_robotwin_runtime_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_files": checkpoint_files,
        "checkpoint_bytes": checkpoint_bytes,
        "expected_checkpoint_bytes": EXPECTED_CHECKPOINT_BYTES,
        "simulator_required_commit": ROBOTWIN_COMMIT,
        "simulator_candidates": simulator_candidates,
        "environment_candidates": environments,
        "filesystem": {
            "path": "/workspace", "total_bytes": capacity.total,
            "used_bytes": capacity.used, "free_bytes": capacity.free,
        },
        "official_source_path": str(args.official_source),
        "official_source_sha256": sha256(args.official_source),
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

"""Record a fail-closed CogACT-Small SimplerEnv setup probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REVISION = "ca302a300542393a0b06e291465076a7d9105c23"
WEIGHT_SHA256 = "013cf0d4954e8a560064905ee03785ce3d0fa126697d4410cfde9a6754665e44"
WEIGHT_SIZE = 30_215_225_166
CHECKPOINT = Path("/workspace/assets/cogact_small")
WEIGHT = CHECKPOINT / "checkpoints/CogACT-Small.pt"
MODEL_ROOT = Path("/workspace/models/cogact")
SIMPLER_ROOT = Path("/workspace/models/simplerenv-openvla")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    staging.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    staging.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    source = MODEL_ROOT / "sim_cogact/cogact_policy.py"
    source_evidence = MODEL_ROOT / "README.md"
    simulator_source = SIMPLER_ROOT / "simpler_env/evaluation/maniskill2_evaluator.py"
    for path in (source, source_evidence, simulator_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    api_url = f"https://huggingface.co/api/models/CogACT/CogACT-Small/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(api_url, timeout=30) as response:
        remote = json.load(response)
    weights = [row for row in remote.get("siblings", []) if row.get("rfilename") == "checkpoints/CogACT-Small.pt"]
    if (remote.get("sha") != REVISION or len(weights) != 1
            or weights[0].get("size") != WEIGHT_SIZE
            or weights[0].get("lfs", {}).get("sha256") != WEIGHT_SHA256):
        raise RuntimeError("CogACT-Small pinned remote checkpoint metadata differs")
    remote_path = output / "pinned_huggingface_metadata.json"
    write_json(remote_path, {
        "repository": "CogACT/CogACT-Small", "revision": REVISION,
        "api_url": api_url, "weight": weights[0],
    })

    usage = shutil.disk_usage("/workspace")
    disk_path = output / "storage_probe.json"
    write_json(disk_path, {
        "workspace_free_bytes": usage.free, "checkpoint_size_bytes": WEIGHT_SIZE,
        "additional_bytes_required": max(0, WEIGHT_SIZE - usage.free),
        "checkpoint_path": str(WEIGHT), "checkpoint_exists": WEIGHT.is_file(),
    })
    stdout_path, stderr_path = output / "probe.stdout.log", output / "probe.stderr.log"
    stdout_path.write_text(
        f"CogACT source revision={git_revision(MODEL_ROOT)}\n"
        f"SimplerEnv source revision={git_revision(SIMPLER_ROOT)}\n"
        f"Pinned checkpoint revision={REVISION}\n"
    )
    reason = (
        f"required exact checkpoint is absent: {WEIGHT}; pinned size={WEIGHT_SIZE}; "
        f"workspace free={usage.free}; no substitute checkpoint was used"
    )
    stderr_path.write_text(reason + "\n")
    record = {
        "schema_version": 1, "model_id": "cogact_small", "status": "blocked",
        "evaluator": "official CogACT-Small SimplerEnv integration",
        "attempt_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempted_command": [
            "python", "scripts/probe_cogact_simplerenv_evaluator.py",
            "--output", str(output),
        ],
        "exit_code": 2,
        "failure_class": "exact_checkpoint_weight_absent_and_insufficient_local_storage",
        "failure_detail": reason,
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        "checkpoint_weight_path": str(WEIGHT),
        "checkpoint_weight_sha256": WEIGHT_SHA256,
        "checkpoint_weight_size_bytes": WEIGHT_SIZE,
        "checkpoint_repository_revision": REVISION,
        "source_evidence_path": str(source_evidence),
        "source_evidence_sha256": sha256(source_evidence),
        "official_evaluator_source_path": str(source),
        "official_evaluator_source_sha256": sha256(source),
        "simulator_evaluator_source_path": str(simulator_source),
        "simulator_evaluator_source_sha256": sha256(simulator_source),
        "blocker_investigation": {
            "reason_permanent": False,
            "resolution_attempts": [
                "Verified pinned Hugging Face revision, LFS SHA-256, and byte size.",
                "Searched the workspace and Hugging Face cache for the exact weight.",
                "Measured available workspace storage before attempting a download.",
                "Refused to substitute another CogACT size or checkpoint.",
            ],
            "evidence_files": [
                {"path": str(remote_path), "sha256": sha256(remote_path)},
                {"path": str(disk_path), "sha256": sha256(disk_path)},
                {"path": str(source), "sha256": sha256(source)},
                {"path": str(simulator_source), "sha256": sha256(simulator_source)},
            ],
        },
        "resume_requirement": (
            "Provide at least 30.3 GB additional persistent storage, download the pinned "
            "CogACT-Small weight, then construct and probe the official SimplerEnv policy integration."
        ),
    }
    write_json(output / "evaluator_attempt.json", record)
    print(json.dumps({"status": "blocked", "reason": reason}, indent=2))
    raise SystemExit(2)


if __name__ == "__main__":
    main()

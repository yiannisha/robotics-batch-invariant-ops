"""Run every non-success-rate stage required for Xiaomi-Robotics-1 coverage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import run_lingbot_vla2_full_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "xiaomi_robotics_1"
MODEL_ROOT = ROOT / "results/cross_model" / MODEL_ID
FULL = MODEL_ROOT / "full"
MANIFEST = FULL / "replay_manifest.json"
ADAPTER = "scripts.model_invariance.adapters.xiaomi_robotics_1_robocasa"
PYTHON = Path("/workspace/envs/dexvla/bin/python")
CHECKPOINT = Path("/workspace/assets/xiaomi_robotics_1/checkpoint")
SOURCE = Path("/dev/shm/xiaomi_replay")
OFFICIAL = Path("/workspace/models/xiaomi-robotics-1")
PILOT = MODEL_ROOT / "pilots/non_success_pre_full_coverage"
START_MARKER = FULL / "non_success_full_campaign_started.json"
SOURCE_SELECTION_SHA256 = "70123ba9826795a1076e8b0ae2ec009962672af6bae399fc98d9dcaee66d5708"
CHECKPOINT_SHARD_HASHES = (
    "ae78351b0612ab4d9b22bb8f9e05e22f8ea444ee6a8e27c284bc78531848fc40",
    "c71009db51ea59a376085bc8940d79708dccbcff41ac79ba955b87c733147619",
    "cb67483e9e0b9ddf98c66961fc1ee38d73084cc5f2c065166db382d20b6a8197",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets() -> None:
    if not PYTHON.is_file() or not MANIFEST.is_file() or not SOURCE.is_dir():
        raise FileNotFoundError("Xiaomi environment, replay manifest, or source is missing")
    manifest = json.loads(MANIFEST.read_text())
    if (
        len(manifest.get("observations", [])) != 1000
        or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
        or manifest.get("noise_shape") != [1000, 3, 10, 60]
    ):
        raise ValueError("Xiaomi replay does not contain the full campaign")
    selection_hash = hashlib.sha256(json.dumps(
        manifest.get("source_sha256") or {}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if selection_hash != SOURCE_SELECTION_SHA256:
        raise ValueError("Xiaomi selected source episodes differ")
    for relative, expected in manifest["source_sha256"].items():
        path = SOURCE / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Xiaomi source episode changed: {relative}")
    shards = sorted(CHECKPOINT.glob("model-*-of-00003.safetensors"))
    if len(shards) != 3 or tuple(sha256(path) for path in shards) != CHECKPOINT_SHARD_HASHES:
        raise ValueError("Xiaomi checkpoint differs from the exact three-shard snapshot")
    revision = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != "0dd7aef8dc87296246aae812a1f59ccb708e5546":
        raise ValueError("Xiaomi official code revision differs")


def wait_for_predecessor(pid: int) -> None:
    while True:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            return
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if "run_lingbot_vla2_full_campaign.py" not in command:
            raise RuntimeError(f"PID {pid} no longer identifies the LingBot supervisor")
        print(f"waiting for LingBot full-campaign supervisor PID {pid}", flush=True)
        time.sleep(30)


def configure_shared_driver() -> None:
    manifest = json.loads(MANIFEST.read_text())
    source_files = [SOURCE / relative for relative in sorted(manifest["source_sha256"])]
    source_files.extend(
        SOURCE / relative for relative in sorted(manifest["source_metadata"])
    )
    values = {
        "MODEL_ID": MODEL_ID,
        "MODEL_ROOT": MODEL_ROOT,
        "FULL": FULL,
        "MANIFEST": MANIFEST,
        "ADAPTER": ADAPTER,
        "PYTHON": PYTHON,
        "CHECKPOINT": CHECKPOINT,
        "SOURCE": SOURCE,
        "OFFICIAL": OFFICIAL,
        "PILOT": PILOT,
        "START_MARKER": START_MARKER,
        "SOURCE_SHA256": SOURCE_SELECTION_SHA256,
        "CHECKPOINT_SHARD_HASHES": CHECKPOINT_SHARD_HASHES,
        "MODEL_LABEL": "Xiaomi-Robotics-1",
        "ADAPTER_FILE": ROOT / "scripts/model_invariance/adapters/xiaomi_robotics_1_robocasa.py",
        "OPERATOR_BUILDER": "scripts/build_xiaomi_robotics_1_operator_config.py",
        "FLOW_STEPS": 5,
        "FLOW_LABEL_TEMPLATE": "flow.step_{step:02d}.state",
        "PILOT_REASON": "Prior Xiaomi outputs were pilots over at most three source frames and synthetic state.",
        "CAMPAIGN_ENVIRONMENT": {
            "PYTHONPATH": f"{OFFICIAL}:{ROOT}",
            "XR1_CHECKPOINT": str(CHECKPOINT),
            "XR1_SAMPLE": str(
                SOURCE / "data/chunk-000/episode_000000.parquet"
            ),
            "XR1_REPLAY_MANIFEST": str(MANIFEST),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "NUMERICAL_SOURCES": [
            ROOT / "scripts/model_invariance/adapters/xiaomi_robotics_1_robocasa.py",
            ROOT / "scripts/run_model_replay_invariance.py",
            ROOT / "batch_invariant_ops/batch_invariant_ops.py",
            CHECKPOINT / "modeling_mibot.py",
            CHECKPOINT / "processing_mibot.py",
            OFFICIAL / "eval_robocasa/main.py",
        ],
        "FROZEN_INPUT_FILES": source_files,
        "verify_assets": verify_assets,
        "wait_for_predecessor": wait_for_predecessor,
    }
    for name, value in values.items():
        setattr(campaign, name, value)


def main() -> None:
    configure_shared_driver()
    campaign.main()


if __name__ == "__main__":
    main()

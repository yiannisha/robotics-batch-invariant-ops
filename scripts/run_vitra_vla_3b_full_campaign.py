"""Run every non-success-rate stage required for VITRA-VLA-3B coverage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import run_lingbot_vla2_full_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "vitra_vla_3b"
MODEL_ROOT = ROOT / "results/cross_model" / MODEL_ID
FULL = MODEL_ROOT / "full"
MANIFEST = FULL / "replay_manifest.json"
ADAPTER = "scripts.model_invariance.adapters.vitra_vla_3b"
PYTHON = Path("/workspace/envs/spatialvla/bin/python")
CHECKPOINT_BUNDLE = Path("/workspace/assets/vitra/checkpoint")
CHECKPOINT = CHECKPOINT_BUNDLE / "checkpoints"
WEIGHTS = CHECKPOINT / "vitra-vla-3b.pt"
SOURCE = Path("/workspace/assets/vitra/teledata")
OFFICIAL = Path("/workspace/models/vitra")
PALIGEMMA_METADATA = Path("/workspace/assets/vitra/paligemma_metadata")
PILOT = MODEL_ROOT / "pilots/non_success_pre_full_coverage"
START_MARKER = FULL / "non_success_full_campaign_started.json"
SOURCE_SHA256 = "11fcb3067b16a3c7f2fccdac0c5ca799667a1ef7f4318afd6f2db57aebdd7f5f"
CHECKPOINT_SHARD_HASHES = (
    "7f57dfd4451d68f41accab8a06bb388d6fdc2cce87ee8b31fafa7c8f01d83796",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets() -> None:
    if not PYTHON.is_file() or not MANIFEST.is_file() or not SOURCE.is_dir():
        raise FileNotFoundError("VITRA environment, replay manifest, or source is missing")
    manifest = json.loads(MANIFEST.read_text())
    if (
        len(manifest.get("observations", [])) != 1000
        or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
        or manifest.get("noise_shape") != [1000, 3, 16, 192]
        or (manifest.get("source") or {}).get("aggregate_sha256") != SOURCE_SHA256
    ):
        raise ValueError("VITRA replay does not contain the full campaign")
    for record in (manifest.get("source") or {}).get("files", []):
        path = SOURCE / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"VITRA source changed: {record['path']}")
    if (
        not WEIGHTS.is_file()
        or WEIGHTS.stat().st_size != 15_067_578_877
        or sha256(WEIGHTS) != CHECKPOINT_SHARD_HASHES[0]
    ):
        raise ValueError("VITRA checkpoint differs from the exact public snapshot")
    revision = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != "b35517202b39d32a753fdd42014b2cc3c41fab58":
        raise ValueError("VITRA official code revision differs")


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
        if "run_xiaomi_robotics_1_full_campaign.py" not in command:
            raise RuntimeError(f"PID {pid} no longer identifies the Xiaomi supervisor")
        print(f"waiting for Xiaomi full-campaign supervisor PID {pid}", flush=True)
        time.sleep(30)


def configure_shared_driver() -> None:
    manifest = json.loads(MANIFEST.read_text())
    frozen_inputs = [SOURCE / row["path"] for row in manifest["source"]["files"]]
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
        "SOURCE_SHA256": SOURCE_SHA256,
        "CHECKPOINT_SHARD_HASHES": CHECKPOINT_SHARD_HASHES,
        "MODEL_LABEL": "VITRA-VLA-3B",
        "ADAPTER_FILE": ROOT / "scripts/model_invariance/adapters/vitra_vla_3b.py",
        "OPERATOR_BUILDER": "scripts/build_vitra_vla_3b_operator_config.py",
        "FLOW_STEPS": 10,
        "FLOW_LABEL_TEMPLATE": "ddim.step_{step:02d}.state",
        "PILOT_REASON": (
            "Prior VITRA outputs used three example images and synthetic zero state."
        ),
        "CAMPAIGN_ENVIRONMENT": {
            "PYTHONPATH": f"{OFFICIAL}:{ROOT}",
            "VITRA_SOURCE": str(OFFICIAL),
            "VITRA_CHECKPOINT": str(CHECKPOINT_BUNDLE),
            "VITRA_PALIGEMMA_METADATA": str(PALIGEMMA_METADATA),
            "VITRA_REPLAY_MANIFEST": str(MANIFEST),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "NUMERICAL_SOURCES": [
            ROOT / "scripts/model_invariance/adapters/vitra_vla_3b.py",
            ROOT / "scripts/run_model_replay_invariance.py",
            ROOT / "batch_invariant_ops/batch_invariant_ops.py",
            OFFICIAL / "vitra/models/vla_builder.py",
            OFFICIAL / "vitra/models/vlm_builder.py",
            OFFICIAL / "vitra/models/vla/vitra_paligemma.py",
            OFFICIAL / "vitra/models/action_model/dit.py",
            OFFICIAL / "vitra/models/action_model/gaussian_diffusion.py",
        ],
        "FROZEN_INPUT_FILES": frozen_inputs,
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

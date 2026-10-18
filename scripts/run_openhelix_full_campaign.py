"""Run every non-success-rate stage required for OpenHelix coverage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import run_lingbot_vla2_full_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openhelix"
MODEL_ROOT = ROOT / "results/cross_model" / MODEL_ID
FULL = MODEL_ROOT / "full"
MANIFEST = FULL / "replay_manifest.json"
RECEIPT = FULL / "checkpoint_stream_receipt.json"
ADAPTER = "scripts.model_invariance.adapters.openhelix_calvin"
PYTHON = Path("/workspace/envs/openhelix/bin/python")
LOCAL_CHECKPOINT = Path("/workspace/assets/openhelix/checkpoint/prompt_tuning_aux")
CHECKPOINT = FULL
SOURCE = Path("/workspace/assets/openhelix/calvin_abc_1000/validation")
OFFICIAL = Path("/workspace/models/openhelix")
BASE = Path("/workspace/assets/openhelix/llava_metadata")
VISION = Path("/workspace/assets/openhelix/clip_metadata")
PILOT = MODEL_ROOT / "pilots/non_success_pre_full_coverage"
START_MARKER = FULL / "non_success_full_campaign_started.json"
SOURCE_SHA256 = "6eaad2fe3e2452009e7109b34ec86fe0fdd3d1ca24e9c52301b3ddb23d1771b5"
CHECKPOINT_SHARD_HASHES = (
    "7bc815c581781b9d864ff22fbfd87f84d45adda0b283609d42f5716f6b933ac7",
    "6f015b79cea364ea356c0e11414e9dbe71f4736b2b5b8523119a34a087e0e758",
    "2ec780a890f2554385765fe3f4beedeb6b953d7eaf6f3887ad1535e56976433f",
    "6d552bc92746187c7d1614186e924dee8c3b456c06be03868da9d2ac07410128",
    "247bc9b65065446a596a3c431077bdbe4126681821334c2019369aaa65f49620",
    "4172759f8b2e57b1e0d3d8daaddea4b5489adfa546f07b4a7ce1b53db55f2e62",
    "ee2df39c2fc516148d2d82e108f66c189f2cd769fb4e42cc333e6dca23d87810",
    "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638",
)
EXPECTED_PACKAGES = {
    "torch": "2.8.0+cu128", "transformers": "4.40.1", "diffusers": "0.29.2",
    "peft": "0.11.1", "accelerate": "0.31.0", "safetensors": "0.4.5",
    "sentencepiece": "0.2.1", "einops": "0.8.1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        if "run_vitra_vla_3b_full_campaign.py" not in command:
            raise RuntimeError(f"PID {pid} no longer identifies the VITRA supervisor")
        print(f"waiting for VITRA full-campaign supervisor PID {pid}", flush=True)
        time.sleep(30)


def verify_assets() -> None:
    if not PYTHON.is_file() or not MANIFEST.is_file() or not SOURCE.is_dir():
        raise FileNotFoundError("OpenHelix environment, replay manifest, or source is missing")
    manifest = json.loads(MANIFEST.read_text())
    if (
        len(manifest.get("observations", [])) != 1000
        or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
        or manifest.get("unique_physical_observations") != 1000
        or manifest.get("noise_shape") != [1000, 3, 4446]
        or manifest.get("source_files_canonical_sha256") != SOURCE_SHA256
    ):
        raise ValueError("OpenHelix replay does not contain the pinned full campaign")
    for row in manifest["source_files"]:
        path = SOURCE / Path(row["path"]).name
        if (not path.is_file() or path.stat().st_size != row["bytes"]
                or sha256(path) != row["sha256"]):
            raise ValueError(f"OpenHelix source changed: {row['path']}")
    local_files = {
        LOCAL_CHECKPOINT / "llava_ckpt_safetensors/model-00000.safetensors":
            (2_105_645_528, CHECKPOINT_SHARD_HASHES[0]),
        LOCAL_CHECKPOINT / "policy.pth": (445_004_793, CHECKPOINT_SHARD_HASHES[-1]),
    }
    for path, (size, digest) in local_files.items():
        if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
            raise ValueError(f"OpenHelix resident checkpoint component differs: {path}")
    revision = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != "d88867fa56977f401a7c89d62b9ebce87d8078f0":
        raise ValueError("OpenHelix official code revision differs")
    package_code = (
        "import json; names=" + repr(list(EXPECTED_PACKAGES))
        + "; print(json.dumps({n:__import__(n).__version__ for n in names}))"
    )
    actual = json.loads(subprocess.run(
        [str(PYTHON), "-c", package_code], check=True, capture_output=True, text=True,
    ).stdout)
    if actual != EXPECTED_PACKAGES:
        raise ValueError(f"OpenHelix package versions differ: {actual}")
    # A range request proves each absent pinned object resolves without storing it.
    for filename in ("model-00001.safetensors", "model-00006.safetensors"):
        url = (
            "https://huggingface.co/OpenHelix/openhelix/resolve/"
            "1a6fabbe148af045ed0710d007ed24c36fa83a53/"
            f"prompt_tuning_aux/llava_ckpt_safetensors/{filename}?download=true"
        )
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status not in (200, 206):
                raise RuntimeError(f"OpenHelix pinned remote shard unavailable: {filename}")


def checkpoint_freeze_files() -> list[Path]:
    if not RECEIPT.is_file():
        raise FileNotFoundError("OpenHelix completed checkpoint stream receipt is missing")
    receipt = json.loads(RECEIPT.read_text())
    actual = tuple(record.get("sha256") for _, record in sorted(
        receipt.get("files", {}).items(),
        key=lambda item: (item[0] == "policy.pth", item[0]),
    ))
    planner = receipt.get("strict_planner_load") or {}
    policy = receipt.get("strict_policy_load") or {}
    if (
        receipt.get("all_pinned_files_verified") is not True
        or actual != CHECKPOINT_SHARD_HASHES
        or planner.get("missing_named_tensors") != []
        or planner.get("meta_parameters_remaining") is not False
        or policy.get("missing_keys") != []
        or policy.get("unexpected_keys") != []
    ):
        raise ValueError("OpenHelix checkpoint receipt does not prove an exact strict load")
    return [
        LOCAL_CHECKPOINT / "llava_ckpt_safetensors/model-00000.safetensors",
        LOCAL_CHECKPOINT / "policy.pth",
        RECEIPT,
    ]


def configure_shared_driver() -> None:
    manifest = json.loads(MANIFEST.read_text())
    frozen_inputs = [SOURCE / Path(row["path"]).name for row in manifest["source_files"]]
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
        "MODEL_LABEL": "OpenHelix",
        "ADAPTER_FILE": ROOT / "scripts/model_invariance/adapters/openhelix_calvin.py",
        "OPERATOR_BUILDER": "scripts/build_openhelix_operator_config.py",
        "FLOW_STEPS": 25,
        "FLOW_LABEL_TEMPLATE": "flow.step_{step:02d}.state",
        "PILOT_REASON": "Prior OpenHelix outputs used only three public CALVIN transitions.",
        "CAMPAIGN_ENVIRONMENT": {
            "PYTHONPATH": f"{OFFICIAL}:{ROOT}",
            "OPENHELIX_SOURCE": str(OFFICIAL),
            "OPENHELIX_CHECKPOINT": str(LOCAL_CHECKPOINT),
            "OPENHELIX_BASE": str(BASE),
            "OPENHELIX_VISION": str(VISION),
            "OPENHELIX_SAMPLE_DIR": str(SOURCE),
            "OPENHELIX_REPLAY_MANIFEST": str(MANIFEST),
            "OPENHELIX_STREAM_CHECKPOINT": "1",
            "OPENHELIX_STREAM_CACHE": "/workspace/assets/openhelix/stream_cache",
            "OPENHELIX_CHECKPOINT_RECEIPT": str(RECEIPT),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "NUMERICAL_SOURCES": [
            ROOT / "scripts/model_invariance/adapters/openhelix_calvin.py",
            ROOT / "scripts/run_model_replay_invariance.py",
            ROOT / "batch_invariant_ops/batch_invariant_ops.py",
            OFFICIAL / "planer.py",
            OFFICIAL / "diffuser_actor/trajectory_optimization/diffuser_actor_act_simple.py",
            OFFICIAL / "diffuser_actor/utils/multihead_custom_attention.py",
        ],
        "FROZEN_INPUT_FILES": frozen_inputs,
        "CHECKPOINT_FREEZE_FILES": checkpoint_freeze_files,
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

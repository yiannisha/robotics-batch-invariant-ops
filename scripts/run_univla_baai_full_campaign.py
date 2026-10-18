"""Run every non-success-rate stage required for full UniVLA coverage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import run_lingbot_vla2_full_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "univla_baai"
MODEL_ROOT = ROOT / "results/cross_model" / MODEL_ID
FULL = MODEL_ROOT / "full"
MANIFEST = FULL / "replay_manifest.json"
RECEIPT = FULL / "checkpoint_cache_receipt.json"
CACHE_ROOT = FULL / "runtime_checkpoint"
CHECKPOINT = CACHE_ROOT / "policy/UNIVLA_LIBERO_IMG_BS192_8K"
VISION = CACHE_ROOT / "vision"
ADAPTER = "scripts.model_invariance.adapters.univla_baai"
PYTHON = Path("/workspace/envs/dexvla/bin/python")
OFFICIAL = Path("/workspace/models/univla")
SOURCE = OFFICIAL / "reference/RoboVLMs/eval/libero/model_wrapper_emu.py"
PILOT = MODEL_ROOT / "pilots/non_success_pre_full_coverage"
START_MARKER = FULL / "non_success_full_campaign_started.json"
SOURCE_SHA256 = "80d93e906ea3d70506a620230f675c105beaa999ced4bbe0ab90fb48757c37ab"
POLICY_REVISION = "df6c94ff872f4fc056693e507c033ca390694118"
VISION_REVISION = "c81f916ad371289e205310a7539255e8a9396488"
CHECKPOINT_SHARD_HASHES = (
    "b7e6e472d974d1c572138839008e308486b526ffe1c9065306bd2107513b9e80",
    "c700932f380da894537a191b1b60f335a617efa01dc44ae6d8216a4f4e97f28d",
    "eeef8be71a34052ff2a590a2120f5d4067114f4f2f72b3c9b78d771968b207c3",
    "0932b91cf0b322993213a8fc5f0eb591171a130addd93d2bff701a3dc5e4259f",
)
SHARD_SIZES = (4_884_766_656, 4_999_819_320, 4_915_916_184, 2_183_554_760)
VISION_SIZE = 1_083_015_124
VISION_SHA256 = "89536431c69b08b10b449ec309f52dcea22f14b7647317f30f5715273392bbf1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_predecessor(pid: int) -> None:
    expected = os.environ.get(
        "UNIVLA_PREDECESSOR_COMMAND", "run_molmoact2_full_campaign.py"
    )
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
        if expected not in command:
            raise RuntimeError(f"PID {pid} no longer identifies {expected}")
        print(f"waiting for predecessor PID {pid}: {expected}", flush=True)
        time.sleep(30)


def ensure_memfd_cache() -> None:
    """Start the sealed anonymous-memory checkpoint holder when needed."""
    if RECEIPT.is_file():
        receipt = json.loads(RECEIPT.read_text())
        pid = receipt.get("daemon_pid")
        command = Path(f"/proc/{pid}/cmdline") if isinstance(pid, int) else None
        if command is not None and command.is_file():
            value = command.read_bytes().replace(b"\0", b" ").decode()
            if "serve_univla_checkpoint_memfd.py" in value:
                return
        RECEIPT.unlink()
    log = FULL / "checkpoint_memfd_daemon.log"
    command = [
        str(PYTHON), "-u", "scripts/serve_univla_checkpoint_memfd.py",
        "--runtime-root", str(CACHE_ROOT),
        "--policy-support", "/workspace/assets/univla/policy/UNIVLA_LIBERO_IMG_BS192_8K",
        "--vision-support", "/workspace/assets/univla/vision",
        "--output", str(RECEIPT),
    ]
    with log.open("ab") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"UniVLA checkpoint memory daemon exited {process.returncode}")
        if RECEIPT.is_file():
            receipt = json.loads(RECEIPT.read_text())
            if receipt.get("daemon_pid") == process.pid:
                return
        time.sleep(5)
    raise TimeoutError("UniVLA checkpoint memory daemon did not become ready in one hour")


def verify_assets() -> None:
    if not PYTHON.is_file() or not MANIFEST.is_file() or sha256(MANIFEST) != SOURCE_SHA256:
        raise FileNotFoundError("UniVLA environment or exact replay manifest is missing")
    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
        or len(manifest.get("observations", [])) != 1000
        or any(len(row.get("request_ids", [])) != 1 for row in manifest["observations"])
    ):
        raise ValueError("UniVLA replay is not the full deterministic 200/800 campaign")
    if not RECEIPT.is_file():
        raise FileNotFoundError("UniVLA exact checkpoint cache receipt is missing")
    receipt = json.loads(RECEIPT.read_text())
    files = receipt.get("files") or {}
    if receipt.get("all_pinned_files_verified") is not True or len(files) != 5:
        raise ValueError("UniVLA checkpoint cache receipt is incomplete")
    for index, (size, digest) in enumerate(zip(SHARD_SIZES, CHECKPOINT_SHARD_HASHES), 1):
        path = CHECKPOINT / f"model-{index:05d}-of-00004.safetensors"
        row = files.get(f"policy/{path.name}") or {}
        if (
            not path.is_file() or path.stat().st_size != size or sha256(path) != digest
            or row.get("bytes") != size or row.get("sha256") != digest
            or row.get("revision") != POLICY_REVISION
        ):
            raise ValueError(f"UniVLA policy shard differs: {path.name}")
    vision = VISION / "model.safetensors"
    row = files.get("vision/model.safetensors") or {}
    if (
        not vision.is_file() or vision.stat().st_size != VISION_SIZE
        or sha256(vision) != VISION_SHA256 or row.get("bytes") != VISION_SIZE
        or row.get("sha256") != VISION_SHA256 or row.get("revision") != VISION_REVISION
    ):
        raise ValueError("UniVLA vision tokenizer weight differs")
    revision = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != "a91ee5f269d7ad46d46635ee6481d2eb655b4af9":
        raise ValueError("UniVLA official source revision differs")


def checkpoint_freeze_files() -> list[Path]:
    return [
        *sorted(CHECKPOINT.glob("*")),
        *sorted(VISION.glob("*")),
        RECEIPT,
    ]


def configure_shared_driver() -> None:
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
        "MODEL_LABEL": "UniVLA",
        "ADAPTER_FILE": ROOT / "scripts/model_invariance/adapters/univla_baai.py",
        "OPERATOR_BUILDER": "scripts/build_univla_operator_config.py",
        "FLOW_STEPS": 80,
        "FLOW_LABEL_TEMPLATE": "flow.step_{step:03d}.scores",
        "FLOW_VARIABLE": True,
        "PILOT_REASON": (
            "Prior UniVLA outputs covered 64 fixture frames and pilot timing schedules."
        ),
        "CAMPAIGN_ENVIRONMENT": {
            "PYTHONPATH": f"{OFFICIAL}:{ROOT}",
            "UNIVLA_CHECKPOINT": str(CHECKPOINT),
            "UNIVLA_VISION_TOKENIZER": str(VISION),
            "UNIVLA_SOURCE": str(OFFICIAL),
            "UNIVLA_SAMPLE_DIR": "/workspace/assets/lerobot_libero/fixture",
            "UNIVLA_REPLAY_MANIFEST": str(MANIFEST),
            "UNIVLA_CHECKPOINT_RECEIPT": str(RECEIPT),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "NUMERICAL_SOURCES": [
            ROOT / "scripts/model_invariance/adapters/univla_baai.py",
            ROOT / "scripts/run_model_replay_invariance.py",
            ROOT / "batch_invariant_ops/batch_invariant_ops.py",
            OFFICIAL / "reference/Emu3/emu3/mllm/modeling_emu3.py",
            OFFICIAL / "reference/Emu3/emu3/mllm/processing_emu3.py",
            SOURCE,
        ],
        "FROZEN_INPUT_FILES": [SOURCE],
        "CHECKPOINT_FREEZE_FILES": checkpoint_freeze_files,
        "REQUEST_NOISE_FILE": None,
        "verify_assets": verify_assets,
        "wait_for_predecessor": wait_for_predecessor,
    }
    for name, value in values.items():
        setattr(campaign, name, value)
    campaign.__doc__ = __doc__


def main() -> None:
    ensure_memfd_cache()
    configure_shared_driver()
    campaign.main()


if __name__ == "__main__":
    main()

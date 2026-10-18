"""Run every non-success-rate stage required for full MolmoAct2 coverage."""

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
MODEL_ID = "molmoact2"
MODEL_ROOT = ROOT / "results/cross_model" / MODEL_ID
FULL = MODEL_ROOT / "full"
MANIFEST = FULL / "replay_manifest.json"
RECEIPT = FULL / "checkpoint_stream_receipt.json"
ADAPTER = "scripts.model_invariance.adapters.molmoact2"
PYTHON = Path("/workspace/envs/smolvla/bin/python")
CHECKPOINT = Path("/workspace/assets/molmoact2/checkpoint")
SOURCE = ROOT / "results/cross_model/smolvla/full/replay_manifest.json"
OFFICIAL = Path("/workspace/models/lerobot")
PILOT = MODEL_ROOT / "pilots/non_success_pre_full_coverage"
START_MARKER = FULL / "non_success_full_campaign_started.json"
SOURCE_SHA256 = "02a5e3f5837a5ae950ec093aa90e5e028bf4ceb79a3894c2d16c81d6fef37047"
CHECKPOINT_SHARD_HASHES = (
    "5aa56113a3c1c95f0c03b90bada5b7ac2babf59c752a3d6201b8362a530c6809",
    "1fd95aec9ad144d3f4eac42db88a000321e016beeae606ecfcb84fcbdb13ecf4",
    "e6c455434915e74a8d750bedd0fc4c23b5a8ce37954aaff9c3e3f22085a26890",
    "4e5bcf5bad81573153d6deaa3d25c3e53dc697658488b1cdbab4c8a3dffba1b6",
    "76140ad00c225f95b80eb6b6a5fea2f985e4fdde3356a3c67c93d87d083d4ee2",
)
SHARD_SIZES = (4_919_324_120, 4_844_690_992, 4_844_691_024, 4_877_619_536, 2_282_630_240)
SUPPORT_HASHES = {
    "config.json": "2e521b21ac0a74eb124e20242704957177bf32a7c4296c01d7180b74994c40b4",
    "model.safetensors.index.json": "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06",
    "norm_stats.json": "de7de428abfd1af0aff0e97ff499a4a27012106b3982d22e8b8d029de312db80",
    "modeling_molmoact2.py": "3e0090b8cc045d59911c0a579815ebd17ce0423f9fdd03be887412df5c7b8b49",
    "configuration_molmoact2.py": "ccc1b24db7ab42b30705414b14de9601b1d82fba6f6d85393b0e9cbe1052fac4",
    "processing_molmoact2.py": "61ec88c04cf70844dc98ab518d332911ec32cf5f24f052b7413ddd5d61483109",
    "image_processing_molmoact2.py": "bb623a013e7f897ffb938990bd02374b61dabcabd87f9f19d59111ec88ecf675",
    "inference.py": "14a328aa9c8094fe5bf6d99aa98d49d0337485ff48f828cc1944bb66318cc67d",
    "tokenizer.json": "d5395aefc9b1b7f0385d8c86a2f1775e5af81bdfbf9f2d97827ea37921d9f862",
    "tokenizer_config.json": "0ffb85ef66d5b354f53a341ff01edc5137c324981bf37928b86bcdee2b265415",
    "processor_config.json": "bd5ad3ddc456b7534005e075a5c3cb0644da5e4b1aa77f1d74268a9e4c7f071a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_predecessor(pid: int) -> None:
    expected_command = os.environ.get(
        "MOLMOACT2_PREDECESSOR_COMMAND", "run_openhelix_full_campaign.py"
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
        if expected_command not in command:
            raise RuntimeError(
                f"PID {pid} no longer identifies expected predecessor {expected_command}"
            )
        print(f"waiting for predecessor PID {pid}: {expected_command}", flush=True)
        time.sleep(30)


def verify_assets() -> None:
    if not PYTHON.is_file() or not MANIFEST.is_file() or sha256(SOURCE) != SOURCE_SHA256:
        raise FileNotFoundError("MolmoAct2 environment or pinned full replay is missing")
    manifest = json.loads(MANIFEST.read_text())
    if (len(manifest.get("observations", [])) != 1000
            or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 800}
            or manifest.get("noise_shape") != [1000, 3, 10, 32]
            or manifest.get("unique_physical_observations") != 1000):
        raise ValueError("MolmoAct2 replay does not contain the full campaign")
    for name, expected in SUPPORT_HASHES.items():
        path = CHECKPOINT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"MolmoAct2 support artifact differs: {name}")
    revision = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != "e595b7902714ba51f91e47523f66f89c5181b649":
        raise ValueError("official LeRobot source revision differs")
    for index, (size, digest) in enumerate(zip(SHARD_SIZES, CHECKPOINT_SHARD_HASHES), 1):
        name = f"model-{index:05d}-of-00005.safetensors"
        local = CHECKPOINT / name
        if local.is_file():
            if local.stat().st_size != size or sha256(local) != digest:
                raise ValueError(f"resident MolmoAct2 shard differs: {name}")
            continue
        url = (
            "https://huggingface.co/allenai/MolmoAct2-LIBERO/resolve/"
            f"0d24a92bd1faf321ef497c3bbd5681af97c65aa2/{name}?download=true"
        )
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120
        ) as response:
            if response.status not in (200, 206):
                raise RuntimeError(f"pinned MolmoAct2 shard unavailable: {name}")


def checkpoint_freeze_files() -> list[Path]:
    if not RECEIPT.is_file():
        raise FileNotFoundError("MolmoAct2 completed checkpoint stream receipt is missing")
    receipt = json.loads(RECEIPT.read_text())
    strict = receipt.get("strict_model_load") or {}
    actual = tuple(
        receipt.get("files", {}).get(f"model-{index:05d}-of-00005.safetensors", {}).get("sha256")
        for index in range(1, 6)
    )
    if (receipt.get("all_pinned_files_verified") is not True
            or actual != CHECKPOINT_SHARD_HASHES
            or strict.get("expected_state_tensors") != 1295
            or strict.get("loaded_state_tensors") != 1295
            or strict.get("missing_state_tensors") != []
            or strict.get("meta_parameters_remaining") != []
            or strict.get("meta_buffers_remaining") != []):
        raise ValueError("MolmoAct2 checkpoint receipt does not prove exact strict load")
    return [CHECKPOINT / name for name in SUPPORT_HASHES] + [RECEIPT]


def configure_shared_driver() -> None:
    values = {
        "MODEL_ID": MODEL_ID, "MODEL_ROOT": MODEL_ROOT, "FULL": FULL,
        "MANIFEST": MANIFEST, "ADAPTER": ADAPTER, "PYTHON": PYTHON,
        "CHECKPOINT": CHECKPOINT, "SOURCE": SOURCE, "OFFICIAL": OFFICIAL,
        "PILOT": PILOT, "START_MARKER": START_MARKER,
        "SOURCE_SHA256": SOURCE_SHA256,
        "CHECKPOINT_SHARD_HASHES": CHECKPOINT_SHARD_HASHES,
        "MODEL_LABEL": "MolmoAct2",
        "ADAPTER_FILE": ROOT / "scripts/model_invariance/adapters/molmoact2.py",
        "OPERATOR_BUILDER": "scripts/build_molmoact2_operator_config.py",
        "FLOW_STEPS": 10,
        "FLOW_LABEL_TEMPLATE": "flow.step_{step:02d}.state",
        "PILOT_REASON": "Prior MolmoAct2 outputs covered only 64 fixture frames and pilot schedules.",
        "CAMPAIGN_ENVIRONMENT": {
            "PYTHONPATH": str(ROOT),
            "MOLMOACT2_CHECKPOINT": str(CHECKPOINT),
            "MOLMOACT2_SAMPLE_DIR": "/workspace/assets/smolvla/full_replay",
            "MOLMOACT2_REPLAY_MANIFEST": str(MANIFEST),
            "MOLMOACT2_STREAM_CHECKPOINT": "1",
            "MOLMOACT2_CHECKPOINT_RECEIPT": str(RECEIPT),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "NUMERICAL_SOURCES": [
            ROOT / "scripts/model_invariance/adapters/molmoact2.py",
            ROOT / "scripts/run_model_replay_invariance.py",
            ROOT / "batch_invariant_ops/batch_invariant_ops.py",
            CHECKPOINT / "modeling_molmoact2.py",
            CHECKPOINT / "inference.py",
            CHECKPOINT / "processing_molmoact2.py",
        ],
        # The shared driver appends the model's request_noise.npy itself.
        "FROZEN_INPUT_FILES": [SOURCE],
        "CHECKPOINT_FREEZE_FILES": checkpoint_freeze_files,
        "verify_assets": verify_assets,
        "wait_for_predecessor": wait_for_predecessor,
    }
    for name, value in values.items():
        setattr(campaign, name, value)
    campaign.__doc__ = __doc__


def main() -> None:
    configure_shared_driver()
    campaign.main()


if __name__ == "__main__":
    main()

"""Record the exact MolmoAct2-LIBERO checkpoint/storage preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
INDEX_SHA256 = "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06"
SHARDS = (
    ("model-00001-of-00005.safetensors", 4_919_324_120, "5aa56113a3c1c95f0c03b90bada5b7ac2babf59c752a3d6201b8362a530c6809"),
    ("model-00002-of-00005.safetensors", 4_844_690_992, "1fd95aec9ad144d3f4eac42db88a000321e016beeae606ecfcb84fcbdb13ecf4"),
    ("model-00003-of-00005.safetensors", 4_844_691_024, "e6c455434915e74a8d750bedd0fc4c23b5a8ce37954aaff9c3e3f22085a26890"),
    ("model-00004-of-00005.safetensors", 4_877_619_536, "4e5bcf5bad81573153d6deaa3d25c3e53dc697658488b1cdbab4c8a3dffba1b6"),
    ("model-00005-of-00005.safetensors", 2_282_630_240, "76140ad00c225f95b80eb6b6a5fea2f985e4fdde3356a3c67c93d87d083d4ee2"),
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
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=Path("/workspace/assets/molmoact2/checkpoint"),
    )
    parser.add_argument(
        "--checkpoint-manifest", type=Path,
        default=Path("research/molmoact2/checkpoint.txt"),
    )
    parser.add_argument(
        "--official-source", type=Path,
        default=Path("/workspace/models/lerobot/docs/source/policy_molmoact2_README.md"),
    )
    parser.add_argument(
        "--python", type=Path, default=Path("/workspace/envs/smolvla/bin/python"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    index = args.checkpoint_root / "model.safetensors.index.json"
    for required in (index, args.checkpoint_manifest, args.official_source):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256(index) != INDEX_SHA256:
        raise RuntimeError("resident checkpoint index differs from pinned MolmoAct2 revision")

    artifacts = []
    for name, expected_size, expected_hash in SHARDS:
        path = args.checkpoint_root / name
        present = path.is_file()
        artifacts.append({
            "name": name,
            "path": str(path),
            "size_bytes": expected_size,
            "sha256": expected_hash,
            "present": present,
            "resident_size_bytes": path.stat().st_size if present else None,
            "resident_sha256": sha256(path) if present else None,
        })
    missing = [item for item in artifacts if not item["present"]]
    mismatched = [
        item for item in artifacts if item["present"] and (
            item["resident_size_bytes"] != item["size_bytes"]
            or item["resident_sha256"] != item["sha256"]
        )
    ]
    missing_bytes = sum(item["size_bytes"] for item in missing)
    capacity = shutil.disk_usage("/workspace")
    remote = []
    for item in missing:
        url = (
            "https://huggingface.co/allenai/MolmoAct2-LIBERO/resolve/"
            f"{REVISION}/{item['name']}?download=true"
        )
        probe = {"name": item["name"], "url": url, "range_probe": False, "error": None}
        try:
            request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                probe["range_probe"] = response.status in (200, 206)
        except Exception as error:
            probe["error"] = f"{type(error).__name__}: {error}"
        remote.append(probe)
    mem_available = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    meta_code = """
import json, torch
from transformers import AutoConfig, AutoModelForImageTextToText
p='/workspace/assets/molmoact2/checkpoint'
c=AutoConfig.from_pretrained(p, trust_remote_code=True)
with torch.device('meta'):
 m=AutoModelForImageTextToText.from_config(c, trust_remote_code=True, dtype=torch.bfloat16)
i=json.load(open(p+'/model.safetensors.index.json'))['weight_map']
s=set(m.state_dict())
print(json.dumps({'state_tensors':len(s),'index_tensors':len(i),'exact_inventory':s==set(i),'meta_parameters':sum(x.is_meta for x in m.parameters()),'meta_buffers':sum(x.is_meta for x in m.buffers())}))
"""
    meta_probe = subprocess.run(
        [str(args.python), "-c", meta_code], check=False, capture_output=True, text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    try:
        meta_result = json.loads(meta_probe.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        meta_result = {}
    blockers = []
    if mismatched:
        blockers.append(f"{len(mismatched)} resident checkpoint shards differ from pinned bytes")
    if any(not item["range_probe"] for item in remote):
        blockers.append("one or more absent pinned shards cannot be range-read")
    largest_missing = max((item["size_bytes"] for item in missing), default=0)
    if mem_available < largest_missing * 3:
        blockers.append(
            f"available RAM {mem_available} cannot hold a streamed shard and tensor decode headroom"
        )
    if (meta_probe.returncode != 0 or meta_result.get("state_tensors") != 1295
            or meta_result.get("index_tensors") != 1295
            or meta_result.get("exact_inventory") is not True
            or meta_result.get("meta_parameters") != 1294
            or meta_result.get("meta_buffers") != 3):
        blockers.append("official MolmoAct2 meta-device model/index inventory probe failed")

    report = {
        "schema_version": 1,
        "model_id": "molmoact2",
        "probe": "exact_checkpoint_and_capacity_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "allenai/MolmoAct2-LIBERO",
        "revision": REVISION,
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_index": {"path": str(index), "sha256": sha256(index)},
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest.resolve()),
            "sha256": sha256(args.checkpoint_manifest),
        },
        "official_source": {
            "path": str(args.official_source), "sha256": sha256(args.official_source),
        },
        "artifacts": artifacts,
        "filesystem": {
            "path": "/workspace", "total_bytes": capacity.total,
            "used_bytes": capacity.used, "free_bytes": capacity.free,
            "missing_checkpoint_bytes": missing_bytes,
        },
        "memory_streaming": {
            "enabled": True,
            "strategy": "one pinned shard downloaded to RAM, full SHA-256, safetensors decode, direct meta-to-GPU assignment",
            "mem_available_bytes": mem_available,
            "largest_missing_shard_bytes": largest_missing,
            "minimum_required_bytes": largest_missing * 3,
            "remote_exact_shards": remote,
            "meta_model_probe": {
                "python": str(args.python), "exit_code": meta_probe.returncode,
                "result": meta_result, "stderr": meta_probe.stderr,
            },
        },
        "blockers": blockers,
        "runnable": not blockers,
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

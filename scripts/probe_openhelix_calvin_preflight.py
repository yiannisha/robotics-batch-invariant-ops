"""Record OpenHelix exact-checkpoint and CALVIN evaluator readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ARTIFACTS = (
    ("llava_ckpt_safetensors/model-00000.safetensors", 2_105_645_528, "7bc815c581781b9d864ff22fbfd87f84d45adda0b283609d42f5716f6b933ac7"),
    ("llava_ckpt_safetensors/model-00001.safetensors", 2_114_018_192, "6f015b79cea364ea356c0e11414e9dbe71f4736b2b5b8523119a34a087e0e758"),
    ("llava_ckpt_safetensors/model-00002.safetensors", 2_114_034_864, "2ec780a890f2554385765fe3f4beedeb6b953d7eaf6f3887ad1535e56976433f"),
    ("llava_ckpt_safetensors/model-00003.safetensors", 2_124_504_248, "6d552bc92746187c7d1614186e924dee8c3b456c06be03868da9d2ac07410128"),
    ("llava_ckpt_safetensors/model-00004.safetensors", 2_057_395_512, "247bc9b65065446a596a3c431077bdbe4126681821334c2019369aaa65f49620"),
    ("llava_ckpt_safetensors/model-00005.safetensors", 2_114_018_256, "4172759f8b2e57b1e0d3d8daaddea4b5489adfa546f07b4a7ce1b53db55f2e62"),
    ("llava_ckpt_safetensors/model-00006.safetensors", 1_504_179_160, "ee2df39c2fc516148d2d82e108f66c189f2cd769fb4e42cc333e6dca23d87810"),
    ("policy.pth", 445_004_793, "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638"),
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
        default=Path("/workspace/assets/openhelix/checkpoint/prompt_tuning_aux"),
    )
    parser.add_argument(
        "--checkpoint-manifest", type=Path,
        default=Path("research/openhelix/checkpoint.txt"),
    )
    parser.add_argument(
        "--official-source", type=Path,
        default=Path("/workspace/models/openhelix/README.md"),
    )
    parser.add_argument(
        "--official-evaluator", type=Path,
        default=Path("/workspace/models/openhelix/online_evaluation_calvin/evaluate_policy_lcb_pt_act_simple_asy10.py"),
    )
    parser.add_argument(
        "--runtime-probe", type=Path,
        default=Path("results/cross_model/openhelix/full/closed_loop/runtime_probe.json"),
    )
    parser.add_argument(
        "--evaluator-assets", type=Path,
        default=Path("results/cross_model/openhelix/full/closed_loop/evaluator_assets.json"),
    )
    parser.add_argument(
        "--stream-receipt", type=Path,
        default=Path("results/cross_model/openhelix/full/checkpoint_stream_receipt.json"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.checkpoint_manifest, args.official_source, args.official_evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)

    artifacts = []
    for relative, expected_size, expected_hash in ARTIFACTS:
        path = args.checkpoint_root / relative
        present = path.is_file()
        resident_size = path.stat().st_size if present else None
        resident_hash = sha256(path) if present else None
        artifacts.append({
            "name": relative, "path": str(path), "size_bytes": expected_size,
            "sha256": expected_hash, "present": present,
            "resident_size_bytes": resident_size, "resident_sha256": resident_hash,
            "resident_exact": present and resident_size == expected_size and resident_hash == expected_hash,
        })
    missing = [item for item in artifacts if not item["present"]]
    mismatched = [item for item in artifacts if item["present"] and not item["resident_exact"]]
    missing_bytes = sum(item["size_bytes"] for item in missing)
    remote = []
    for item in missing:
        name = Path(item["name"]).name
        url = (
            "https://huggingface.co/OpenHelix/openhelix/resolve/"
            "1a6fabbe148af045ed0710d007ed24c36fa83a53/"
            f"prompt_tuning_aux/llava_ckpt_safetensors/{name}?download=true"
        )
        record = {"name": name, "url": url, "range_probe": False, "error": None}
        try:
            request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                record["range_probe"] = response.status in (200, 206)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        remote.append(record)
    capacity = shutil.disk_usage("/workspace")
    calvin_roots = [Path("/workspace/CALVIN"), Path("/workspace/models/CALVIN"), Path("/workspace/models/calvin")]
    calvin_runtime = (Path("/workspace/models/calvin/calvin_env")).is_dir()
    calvin_env = Path("/workspace/envs/openhelix/bin/python").is_file()
    runtime_probe = json.loads(args.runtime_probe.read_text()) if args.runtime_probe.is_file() else {}
    evaluator_assets = json.loads(args.evaluator_assets.read_text()) if args.evaluator_assets.is_file() else {}
    stream_receipt = json.loads(args.stream_receipt.read_text()) if args.stream_receipt.is_file() else {}
    blockers = []
    if any(not item["range_probe"] for item in remote):
        blockers.append("one or more absent exact planner shards cannot be range-read from the pinned revision")
    if mismatched:
        blockers.append(f"{len(mismatched)} resident checkpoint artifacts differ from pinned hashes")
    largest_missing = max((item["size_bytes"] for item in missing), default=0)
    if capacity.free < largest_missing + 128 * 1024 * 1024:
        blockers.append(
            f"workspace free bytes {capacity.free} cannot hold the largest streamed shard plus headroom"
        )
    if not calvin_runtime:
        blockers.append("CALVIN benchmark runtime/assets checkout is absent")
    if not calvin_env:
        blockers.append("CALVIN/OpenHelix evaluator Python environment is absent")
    if runtime_probe.get("status") != "passed" or len(runtime_probe.get("resets", [])) != 2:
        blockers.append("CALVIN headless two-reset/two-step runtime probe has not passed")
    if len(evaluator_assets.get("files", [])) != 5:
        blockers.append("exact minimal CALVIN ABC-D evaluator assets are absent")

    report = {
        "schema_version": 1, "model_id": "openhelix",
        "probe": "exact_checkpoint_and_calvin_runtime_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "filesystem": {
            "path": "/workspace", "total_bytes": capacity.total,
            "used_bytes": capacity.used, "free_bytes": capacity.free,
            "missing_checkpoint_bytes": missing_bytes,
            "largest_missing_checkpoint_bytes": largest_missing,
        },
        "calvin": {
            "candidate_roots": [str(path) for path in calvin_roots],
            "runtime_assets_present": calvin_runtime,
            "python_environment_present": calvin_env,
            "runtime_probe_path": str(args.runtime_probe.resolve()),
            "runtime_probe_sha256": sha256(args.runtime_probe) if args.runtime_probe.is_file() else None,
            "evaluator_assets_path": str(args.evaluator_assets.resolve()),
            "evaluator_assets_sha256": sha256(args.evaluator_assets) if args.evaluator_assets.is_file() else None,
        },
        "remote_exact_shards": remote,
        "streaming": {
            "enabled": True,
            "strategy": "one pinned shard at a time with full SHA-256 before tensor load",
            "receipt_path": str(args.stream_receipt.resolve()),
            "receipt_sha256": sha256(args.stream_receipt) if args.stream_receipt.is_file() else None,
            "all_pinned_files_verified_in_a_completed_load": stream_receipt.get("all_pinned_files_verified", False),
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

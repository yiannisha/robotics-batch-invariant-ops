"""Record Cosmos3 Nano/Edge exact-checkpoint and RoboLab readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


VAE = ("Wan2.2_VAE.pth", 2_818_839_170, "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36")
ARTIFACTS = {
    "nano": (
        ("transformer/diffusion_pytorch_model-00001-of-00007.safetensors", 4_902_249_008, "f260af4e9c877c22ba9a44246767ff4a0bc9cd8f62fd6d683bedfcac75378ddd"),
        ("transformer/diffusion_pytorch_model-00002-of-00007.safetensors", 4_999_863_656, "bce7ee18cc05ad291f57095d2705cb5ed26d6eca4469a44a0c40f8d507d724e9"),
        ("transformer/diffusion_pytorch_model-00003-of-00007.safetensors", 4_932_719_448, "885c1fc451c28ca34809df3b975b63f58adbaad29da53d6c4ad33613135526c2"),
        ("transformer/diffusion_pytorch_model-00004-of-00007.safetensors", 4_983_084_656, "3a2639189fb96cdf40fba25be3ecee60a42a27431daccdd40f049bbd37ff7244"),
        ("transformer/diffusion_pytorch_model-00005-of-00007.safetensors", 4_949_498_576, "067fca5fbee766cd6828c96478a7ab93ac4bd9595662cae71787420ba4f27f14"),
        ("transformer/diffusion_pytorch_model-00006-of-00007.safetensors", 4_261_644_464, "c3e56f83740a353e44f8342c997fe3cffdcb5cc801d7e54c9bd17c7c016f8a7c"),
        ("transformer/diffusion_pytorch_model-00007-of-00007.safetensors", 1_317_303_952, "5e23d7dbafab591bce1543b55f3ade34231085261d8a1754dfc26822bc691566"),
        ("vision_encoder/model.safetensors", 1_152_811_416, "b76056ebf9067cdb36e77554b2c1855aa1b662a9265c8ed26468b490e40b4adb"),
    ),
    "edge": (
        ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 5_000_041_944, "99b0ff1f9dbdcb497fac16cbc0ebe33dee19653b4c182fa0d474433dcc845206"),
        ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 1_748_779_008, "1842fefdacd0dcbb8ab543b59848a0b696e0c9634676bc76db9e0c971fe328d5"),
        ("vision_encoder/model.safetensors", 978_739_880, "2180ad739ecc96b5c1e9386892d3c5c08bfa42b9cdab9aabc53b028671db89b3"),
    ),
}


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
    parser.add_argument("--variant", choices=("nano", "edge"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument(
        "--official-source", type=Path,
        default=Path("/workspace/models/cosmos-framework/docs/action_policy_droid_server.md"),
    )
    parser.add_argument(
        "--official-evaluator", type=Path,
        default=Path("/workspace/models/cosmos-framework/cosmos_framework/scripts/action_policy_server_robolab.py"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    asset_root = args.asset_root or Path(f"/workspace/assets/cosmos3_{args.variant}")
    manifest = args.checkpoint_manifest or Path(f"research/cosmos3_{args.variant}/checkpoint.txt")
    for path in (manifest, args.official_source, args.official_evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)

    artifacts = []
    for relative, expected_size, expected_hash in (*ARTIFACTS[args.variant], VAE):
        path = asset_root / relative
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
    robolab_roots = [Path("/workspace/RoboLab"), Path("/workspace/models/RoboLab"), Path("/workspace/models/robolab")]
    robolab_runtime = any((root / "robolab").is_dir() or (root / "envs").is_dir() for root in robolab_roots)
    robolab_env = any(
        (Path("/workspace/envs") / name / "bin" / "python").is_file()
        for name in ("RoboLab", "robolab", "cosmos_robolab")
    )
    blockers = []
    if missing:
        blockers.append(f"{len(missing)} exact policy/VAE artifacts are absent ({missing_bytes} bytes)")
    if capacity.free < missing_bytes:
        blockers.append(
            f"workspace free bytes {capacity.free} cannot hold missing artifact bytes {missing_bytes}"
        )
    if not robolab_runtime:
        blockers.append("RoboLab simulator runtime/assets checkout is absent")
    if not robolab_env:
        blockers.append("RoboLab simulator Python environment is absent")

    family = f"cosmos3_{args.variant}"
    report = {
        "schema_version": 1, "model_family": family,
        "paths": [family, f"{family}_video"],
        "probe": "exact_checkpoint_vae_and_robolab_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "wan_vae_revision": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        "filesystem": {
            "path": "/workspace", "total_bytes": capacity.total,
            "used_bytes": capacity.used, "free_bytes": capacity.free,
            "missing_checkpoint_bytes": missing_bytes,
        },
        "robolab": {
            "candidate_roots": [str(path) for path in robolab_roots],
            "runtime_assets_present": robolab_runtime,
            "python_environment_present": robolab_env,
        },
        "checkpoint_manifest": {"path": str(manifest.resolve()), "sha256": sha256(manifest)},
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

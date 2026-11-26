"""Acquire the four requested campaign checkpoints at immutable revisions.

Run with the campaign policy environment. No inference or simulation is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

CHECKPOINTS = {
    "pi05_libero": ("lerobot/pi05_libero_finetuned_v044", "7152f13e34c6c115d808aea0e5d08aca74eac14b", ["*.json", "*.safetensors", "README.md"]),
    "lingbot_vla2": ("robbyant/lingbot-vla-v2-6b-robotwin", "0451855729ec904f970600e0aec8b84661423afe", ["checkpoints/global_step_50000/hf_ckpt/*", "lingbotvla_cli.yaml", "README.md"]),
    "rdt_1b": ("robotics-diffusion-transformer/maniskill-model", "9622afab2b7ce2312a6cf1febc526928589b77eb", ["rdt/*", "lang_embeds/*", "README.md"]),
    "cogact_small": ("CogACT/CogACT-Small", "ca302a300542393a0b06e291465076a7d9105c23", ["checkpoints/*", "config.*", "dataset_statistics.json", "README.md"]),
}


def acquire(root: Path, name: str) -> None:
    repo, revision, patterns = CHECKPOINTS[name]
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"revision mismatch: {name}")
    snapshot_download(repo, revision=revision, local_dir=destination,
                      allow_patterns=patterns, max_workers=2)
    files = []
    for entry in info.siblings:
        path = destination / entry.rfilename
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        expected = entry.lfs.sha256 if entry.lfs else None
        if path.stat().st_size != entry.size or (expected and actual != expected):
            raise RuntimeError(f"download identity mismatch: {path}")
        files.append({"path": entry.rfilename, "bytes": entry.size,
                      "sha256": actual, "upstream_lfs_sha256": expected})
    record = {"model": name, "repository": repo, "revision": revision,
              "status": "checkpoint_acquired_and_hashed", "files": files,
              "inference_qualified": False, "closed_loop_qualified": False}
    (destination / "acquisition.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"ACQUIRED {name}: {len(files)} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(acquire, args.asset_root, name) for name in args.models]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()

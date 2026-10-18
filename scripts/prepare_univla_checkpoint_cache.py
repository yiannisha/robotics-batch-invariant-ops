"""Materialize the exact UniVLA policy and vision weights in a temporary cache.

The workspace cannot hold the 18 GB snapshot.  This downloader writes directly
to a caller-selected filesystem, resumes verified partial downloads, and emits
a content-addressed receipt only after every byte and SHA256 digest matches the
pinned public revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


POLICY_REPOSITORY = "Yuqi1997/UniVLA"
POLICY_REVISION = "df6c94ff872f4fc056693e507c033ca390694118"
VISION_REPOSITORY = "BAAI/Emu3-VisionTokenizer"
VISION_REVISION = "c81f916ad371289e205310a7539255e8a9396488"
FILES = (
    (
        "policy", "UNIVLA_LIBERO_IMG_BS192_8K/model-00001-of-00004.safetensors",
        4_884_766_656, "b7e6e472d974d1c572138839008e308486b526ffe1c9065306bd2107513b9e80",
        POLICY_REPOSITORY, POLICY_REVISION,
    ),
    (
        "policy", "UNIVLA_LIBERO_IMG_BS192_8K/model-00002-of-00004.safetensors",
        4_999_819_320, "c700932f380da894537a191b1b60f335a617efa01dc44ae6d8216a4f4e97f28d",
        POLICY_REPOSITORY, POLICY_REVISION,
    ),
    (
        "policy", "UNIVLA_LIBERO_IMG_BS192_8K/model-00003-of-00004.safetensors",
        4_915_916_184, "eeef8be71a34052ff2a590a2120f5d4067114f4f2f72b3c9b78d771968b207c3",
        POLICY_REPOSITORY, POLICY_REVISION,
    ),
    (
        "policy", "UNIVLA_LIBERO_IMG_BS192_8K/model-00004-of-00004.safetensors",
        2_183_554_760, "0932b91cf0b322993213a8fc5f0eb591171a130addd93d2bff701a3dc5e4259f",
        POLICY_REPOSITORY, POLICY_REVISION,
    ),
    (
        "vision", "model.safetensors", 1_083_015_124,
        "89536431c69b08b10b449ec309f52dcea22f14b7647317f30f5715273392bbf1",
        VISION_REPOSITORY, VISION_REVISION,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_support(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            target = destination / path.name
            if not target.is_file() or sha256(target) != sha256(path):
                temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
                shutil.copy2(path, temporary)
                os.replace(temporary, target)


def download(
    destination: Path, *, repository: str, revision: str, remote_path: str,
    expected_size: int, expected_hash: str,
) -> str:
    if destination.is_file():
        if destination.stat().st_size == expected_size and sha256(destination) == expected_hash:
            return "verified_existing_cache"
        raise ValueError(f"existing UniVLA cache artifact differs: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    digest = hashlib.sha256()
    if offset:
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    quoted = urllib.parse.quote(remote_path, safe="/")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{quoted}?download=true"
    headers = {"User-Agent": "batch-invariance-univla-cache/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
            digest = hashlib.sha256()
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                digest.update(block)
            stream.flush()
            os.fsync(stream.fileno())
    if partial.stat().st_size != expected_size or digest.hexdigest() != expected_hash:
        raise RuntimeError(
            f"downloaded UniVLA artifact failed size/hash verification: {remote_path}"
        )
    os.replace(partial, destination)
    return "pinned_remote_resumable_download"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--policy-support", type=Path, required=True)
    parser.add_argument("--vision-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    policy = cache_root / "policy/UNIVLA_LIBERO_IMG_BS192_8K"
    vision = cache_root / "vision"
    copy_support(args.policy_support, policy)
    copy_support(args.vision_support, vision)
    records = {}
    for kind, remote_path, size, digest, repository, revision in FILES:
        destination = (policy if kind == "policy" else vision) / Path(remote_path).name
        source = download(
            destination, repository=repository, revision=revision,
            remote_path=remote_path, expected_size=size, expected_hash=digest,
        )
        records[f"{kind}/{destination.name}"] = {
            "path": str(destination), "bytes": size, "sha256": digest,
            "repository": repository, "revision": revision,
            "remote_path": remote_path, "source": source,
        }
        print(json.dumps({"verified": str(destination), "bytes": size}), flush=True)
    receipt = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "policy_directory": str(policy),
        "vision_directory": str(vision),
        "files": records,
        "all_pinned_files_verified": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "files": len(records)}))


if __name__ == "__main__":
    main()

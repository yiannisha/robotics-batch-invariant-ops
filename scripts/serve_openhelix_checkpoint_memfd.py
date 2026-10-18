"""Cache the exact remote OpenHelix planner shards in sealed anonymous memory."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_invariance.adapters.openhelix_calvin import (  # noqa: E402
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHARDS,
)


def _download_once(name: str, expected_size: int, expected_hash: str) -> int:
    descriptor = os.memfd_create(
        f"openhelix-cache-{name}", flags=getattr(os, "MFD_ALLOW_SEALING", 0)
    )
    remote_path = f"prompt_tuning_aux/llava_ckpt_safetensors/{name}"
    quoted = urllib.parse.quote(remote_path, safe="/")
    url = (
        f"https://huggingface.co/{CHECKPOINT_REPOSITORY}/resolve/"
        f"{CHECKPOINT_REVISION}/{quoted}?download=true"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "batch-invariance-openhelix-memfd/1"}
    )
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                digest.update(block)
                received += len(block)
        if received != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError(
                f"OpenHelix memory download failed size/hash verification: {name}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def download_to_memfd(
    name: str, expected_size: int, expected_hash: str, attempts: int = 5,
) -> int:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(name, expected_size, expected_hash)
        except Exception as error:
            last_error = error
            if attempt == attempts:
                break
            print(
                json.dumps({
                    "retrying": name,
                    "attempt": attempt + 1,
                    "error": f"{type(error).__name__}: {error}",
                }),
                flush=True,
            )
            time.sleep(min(2 ** attempt, 16))
    assert last_error is not None
    raise last_error


def link_descriptor(descriptor: int, destination: Path) -> str:
    source = f"/proc/{os.getpid()}/fd/{descriptor}"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.is_symlink():
        temporary.unlink()
    if destination.exists() and not destination.is_symlink():
        raise FileExistsError(f"refusing to replace resident checkpoint file: {destination}")
    temporary.symlink_to(source)
    os.replace(temporary, destination)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    shard_root = args.checkpoint.resolve() / "llava_ckpt_safetensors"
    shard_root.mkdir(parents=True, exist_ok=True)
    # Shard zero is already an exact resident file; cache only the six remote shards.
    selected = list(CHECKPOINT_SHARDS[1:])
    descriptors: list[int] = []
    records: dict[str, dict[str, object]] = {}
    lock = threading.Lock()

    def fetch(row: tuple[str, int, str]) -> None:
        name, size, digest = row
        descriptor = download_to_memfd(name, size, digest)
        try:
            source = link_descriptor(descriptor, shard_root / name)
        except BaseException:
            os.close(descriptor)
            raise
        with lock:
            descriptors.append(descriptor)
            records[name] = {
                "bytes": size,
                "sha256": digest,
                "memfd_path": source,
                "verification": "full_stream_sha256_then_memfd_sealed_immutable",
            }
        print(json.dumps({"verified": name, "bytes": size}), flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, row) for row in selected]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        receipt = {
            "schema_version": 1,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "daemon_pid": os.getpid(),
            "repository": CHECKPOINT_REPOSITORY,
            "revision": CHECKPOINT_REVISION,
            "checkpoint_directory": str(args.checkpoint.resolve()),
            "files": records,
            "all_selected_files_verified": set(records) == {row[0] for row in selected},
            "storage": "sealed_memfd",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
        print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
        while True:
            signal.pause()
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


if __name__ == "__main__":
    main()

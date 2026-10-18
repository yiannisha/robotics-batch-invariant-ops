"""Hold the exact 18 GB UniVLA snapshot in sealed anonymous memory files.

Each verified weight is exposed through a small symlink to ``/proc/PID/fd/N``.
This keeps the checkpoint reusable by the campaign's subprocesses without
consuming the nearly-full workspace filesystem or the busy /dev/shm mount.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from prepare_univla_checkpoint_cache import FILES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_support(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        target = destination / path.name
        if target.is_file() and sha256(target) == sha256(path):
            continue
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        shutil.copy2(path, temporary)
        os.replace(temporary, target)


def download_to_memfd(
    *, repository: str, revision: str, remote_path: str,
    expected_size: int, expected_hash: str,
) -> int:
    flags = getattr(os, "MFD_ALLOW_SEALING", 0)
    descriptor = os.memfd_create(f"univla-{Path(remote_path).name}", flags=flags)
    digest = hashlib.sha256()
    received = 0
    quoted = urllib.parse.quote(remote_path, safe="/")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{quoted}?download=true"
    try:
        # Xet may end a valid 206 response early without raising an exception.
        # Reconnect at the exact verified offset and keep one continuous digest.
        for segment in range(64):
            if received == expected_size:
                break
            request = urllib.request.Request(url, headers={
                "User-Agent": "batch-invariance-univla-memfd/1",
                "Range": f"bytes={received}-",
            })
            before = received
            with urllib.request.urlopen(request, timeout=600) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range", "")
                if status != 206 or not content_range.startswith(f"bytes {received}-"):
                    raise RuntimeError(
                        f"unexpected range response for {remote_path}: "
                        f"status={status} content_range={content_range!r}"
                    )
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    if received + len(block) > expected_size:
                        raise RuntimeError(
                            f"range response exceeded pinned size for {remote_path}"
                        )
                    view = memoryview(block)
                    while view:
                        written = os.write(descriptor, view)
                        view = view[written:]
                    digest.update(block)
                    received += len(block)
            if received == before:
                print(json.dumps({
                    "artifact": remote_path,
                    "segment": segment + 1,
                    "received": received,
                    "event": "empty_range_response_retry",
                }), flush=True)
                time.sleep(min(segment + 1, 5))
        else:
            raise RuntimeError(
                f"too many partial range responses for {remote_path}: {received} bytes"
            )
        actual_hash = digest.hexdigest()
        if received != expected_size or actual_hash != expected_hash:
            raise RuntimeError(
                "UniVLA memory download failed size/hash verification: "
                f"{remote_path}; received={received} expected_size={expected_size} "
                f"sha256={actual_hash} expected_sha256={expected_hash}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def link_descriptor(descriptor: int, destination: Path) -> str:
    source = f"/proc/{os.getpid()}/fd/{descriptor}"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(source)
    os.replace(temporary, destination)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--policy-support", type=Path, required=True)
    parser.add_argument("--vision-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download-attempts", type=int, default=3,
        help="Attempts per pinned artifact after a size/hash or transport failure.",
    )
    args = parser.parse_args()
    if args.download_attempts < 1:
        parser.error("--download-attempts must be at least 1")
    root = args.runtime_root.resolve()
    policy = root / "policy/UNIVLA_LIBERO_IMG_BS192_8K"
    vision = root / "vision"
    copy_support(args.policy_support, policy)
    copy_support(args.vision_support, vision)

    descriptors = []
    records = {}
    try:
        # Verify the smaller vision tokenizer first. A transient failure there
        # must not discard ~17 GB of already-verified policy memfds.
        ordered_files = sorted(FILES, key=lambda item: item[0] != "vision")
        for kind, remote_path, size, digest, repository, revision in ordered_files:
            descriptor = None
            for attempt in range(1, args.download_attempts + 1):
                try:
                    descriptor = download_to_memfd(
                        repository=repository, revision=revision,
                        remote_path=remote_path, expected_size=size,
                        expected_hash=digest,
                    )
                    break
                except (OSError, RuntimeError) as error:
                    print(json.dumps({
                        "artifact": remote_path,
                        "attempt": attempt,
                        "attempts": args.download_attempts,
                        "retrying": attempt < args.download_attempts,
                        "error": str(error),
                    }), flush=True)
                    if attempt == args.download_attempts:
                        raise
                    time.sleep(min(5 * attempt, 30))
            assert descriptor is not None
            descriptors.append(descriptor)
            destination = (policy if kind == "policy" else vision) / Path(remote_path).name
            source = link_descriptor(descriptor, destination)
            records[f"{kind}/{destination.name}"] = {
                "path": str(destination), "memfd_path": source,
                "bytes": size, "sha256": digest, "repository": repository,
                "revision": revision, "remote_path": remote_path,
                "verification": "full_stream_sha256_then_memfd_sealed_immutable",
            }
            print(json.dumps({"verified": str(destination), "bytes": size}), flush=True)
        receipt = {
            "schema_version": 1,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "daemon_pid": os.getpid(),
            "runtime_root": str(root),
            "policy_directory": str(policy),
            "vision_directory": str(vision),
            "files": records,
            "all_pinned_files_verified": True,
            "storage": "sealed_memfd",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
        print(json.dumps({"ready": True, "pid": os.getpid(), "files": len(records)}), flush=True)
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

"""Extract only the official CALVIN files required to instantiate ABC-D eval.

The complete pinned CALVIN ABC-D archive is 555 GB, but online evaluation
does not read demonstration frames.  It needs the validation Hydra scene
configuration and statistics.  This script uses HTTP byte ranges through the
ZIP central directory and retains those exact five files without downloading
or pretending to possess the full dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import requests
from remotezip import RemoteZip


ARCHIVE_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
ARCHIVE_BYTES = 555_309_812_705
ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
CALVIN_REVISION = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_REVISION = "1431a46bd36bde5903fb6345e68b5ccc30def666"
MEMBERS = (
    "task_ABC_D/validation/statistics.yaml",
    "task_ABC_D/validation/.hydra/config.yaml",
    "task_ABC_D/validation/.hydra/hydra.yaml",
    "task_ABC_D/validation/.hydra/overrides.yaml",
    "task_ABC_D/validation/.hydra/merged_config.yaml",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.evidence.exists():
        raise FileExistsError("CALVIN evaluator assets/evidence already exist")
    if git_revision(args.calvin_root) != CALVIN_REVISION:
        raise ValueError("CALVIN source revision differs")
    if git_revision(args.calvin_root / "calvin_env") != CALVIN_ENV_REVISION:
        raise ValueError("CALVIN environment revision differs")

    response = requests.head(ARCHIVE_URL, allow_redirects=True, timeout=120)
    response.raise_for_status()
    if int(response.headers.get("Content-Length", -1)) != ARCHIVE_BYTES:
        raise ValueError("CALVIN ABC-D archive byte count differs")
    if response.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise ValueError("CALVIN server does not advertise byte-range support")

    temporary_root = args.output_root.with_name(args.output_root.name + f".tmp.{os.getpid()}")
    records = []
    try:
        with RemoteZip(ARCHIVE_URL) as archive:
            infos = {info.filename: info for info in archive.infolist()}
            for member in MEMBERS:
                if member not in infos:
                    raise FileNotFoundError(f"official archive lacks {member}")
                value = archive.read(member)
                info = infos[member]
                if len(value) != info.file_size:
                    raise ValueError(f"short CALVIN archive member: {member}")
                relative = Path(member).relative_to("task_ABC_D")
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
                records.append({
                    "archive_member": member,
                    "path": relative.as_posix(),
                    "bytes": len(value),
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(value),
                })
        os.replace(temporary_root, args.output_root)
    except BaseException:
        import shutil
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    evidence = {
        "schema_version": 1,
        "purpose": "minimal exact official assets for CALVIN ABC-D online environment instantiation",
        "full_demonstration_archive_retained": False,
        "demonstration_frames_required_for_online_reset_step": False,
        "archive": {
            "url": ARCHIVE_URL,
            "bytes": ARCHIVE_BYTES,
            "published_sha256": ARCHIVE_SHA256,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "selection_method": "HTTP byte-range reads through the ZIP central directory",
        },
        "calvin_source": str(args.calvin_root.resolve()),
        "calvin_revision": CALVIN_REVISION,
        "calvin_env_revision": CALVIN_ENV_REVISION,
        "output_root": str(args.output_root.resolve()),
        "files": records,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.evidence.with_name(args.evidence.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.evidence)
    print(json.dumps({
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "output_root": str(args.output_root),
        "evidence": str(args.evidence),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

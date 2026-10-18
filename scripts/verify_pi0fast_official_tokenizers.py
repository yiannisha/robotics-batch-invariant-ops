"""Verify local PI0-FAST tokenizers byte-for-byte against pinned upstream files.

Google's PaliGemma file downloads are gated, but its public repository metadata
exposes the immutable LFS SHA-256 or Git blob SHA-1 of every tokenizer file.
This checks the local mirror against those official hashes; no model weights or
tokenizer bytes are downloaded from the gated repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


TEXT_FILES = (
    "added_tokens.json", "config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
)
TEXT_REPO = "google/paligemma-3b-pt-224"
TEXT_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
ACTION_REPO = "jadechoghari/tokenizer-lib-mean"
ACTION_REVISION = "79ae83e3cbd8786dcb84b628569f8d076ca8151e"


def check_files(repo: str, revision: str, root: Path, names: tuple[str, ...]) -> dict:
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"{repo} resolved to {info.sha}, not pinned {revision}")
    siblings = {item.rfilename: item for item in info.siblings}
    checked = []
    for name in names:
        source = siblings.get(name)
        local = root / name
        if source is None or not local.is_file():
            checked.append({"file": name, "match": False, "reason": "source or local file missing"})
            continue
        content = local.read_bytes()
        local_sha256 = hashlib.sha256(content).hexdigest()
        git_blob_sha1 = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content
        ).hexdigest()
        lfs_sha256 = source.lfs.sha256 if source.lfs else None
        expected = lfs_sha256 or source.blob_id
        actual = local_sha256 if lfs_sha256 else git_blob_sha1
        checked.append({
            "file": name, "bytes": len(content),
            "hash_kind": "lfs_sha256" if lfs_sha256 else "git_blob_sha1",
            "official_hash": expected, "local_hash": actual,
            "local_sha256": local_sha256, "match": actual == expected,
        })
    return {
        "repository": repo, "revision": revision,
        "source_url": f"https://huggingface.co/{repo}/tree/{revision}",
        "local_root": str(root.resolve()),
        "files": checked,
        "complete_byte_match": all(item["match"] for item in checked),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-root", type=Path, default=Path("/workspace/assets/pi_fast/text_tokenizer"))
    parser.add_argument("--action-root", type=Path, default=Path("/workspace/assets/pi_fast/action_tokenizer"))
    parser.add_argument("--output", type=Path, default=Path("results/cross_model/pi_fast/full/tokenizer_provenance.json"))
    args = parser.parse_args()
    action_info = HfApi().model_info(ACTION_REPO, revision=ACTION_REVISION, files_metadata=True)
    action_files = tuple(sorted(item.rfilename for item in action_info.siblings))
    results = [
        check_files(TEXT_REPO, TEXT_REVISION, args.text_root, TEXT_FILES),
        check_files(ACTION_REPO, ACTION_REVISION, args.action_root, action_files),
    ]
    report = {
        "schema_version": 1,
        "evidence_tier": "upstream_byte_hash_verification",
        "model_id": "pi_fast",
        "passed": all(item["complete_byte_match"] for item in results),
        "tokenizers": results,
    }
    serialized = json.dumps(report, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file() and args.output.read_text() != serialized:
        raise FileExistsError("existing tokenizer provenance differs from pinned upstream metadata")
    args.output.write_text(serialized)
    print(json.dumps({"passed": report["passed"], "files": sum(len(item["files"]) for item in results)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Losslessly gzip a preserved rollout directory with a hash manifest.

This is intended for terminal evidence sets, never a live/resumable evaluator
cache.  Each source is removed only after its deterministic gzip replacement
has been decompressed and checked byte-for-byte.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-terminal-marker", default="reset_pairing_audit.json")
    parser.add_argument("--allow-fully-paired", action="store_true")
    parser.add_argument("--expected-files", type=int, default=1260)
    args = parser.parse_args()
    root = args.root.resolve()
    marker = root / args.require_terminal_marker
    if not root.is_dir() or not marker.is_file():
        raise ValueError(f"refusing non-terminal rollout directory: {root}")
    marker_record = json.loads(marker.read_text())
    invalid_archive = (
        "paired_invalid" in root.name
        or marker_record.get("classification")
        == "superseded_provenance_preserved_not_active_campaign_evidence"
    )
    verified_complete = args.allow_fully_paired and marker_record.get("fully_paired") is True
    if not invalid_archive and not verified_complete:
        raise ValueError(
            "refusing rollout compaction unless this is an invalid archive or a fully paired set"
        )

    records = []
    sources = sorted(root.glob("*/rollouts/*.json"))
    existing = sorted(root.glob("*/rollouts/*.json.gz"))
    original_paths = [path.with_suffix("") for path in existing] + sources
    if (args.expected_files < 1 or len(original_paths) != args.expected_files
            or len(set(original_paths)) != args.expected_files):
        raise RuntimeError(
            f"expected exactly {args.expected_files} preserved raw rollouts, "
            f"found {len(original_paths)}"
        )
    for compressed in existing:
        compressed_bytes = compressed.read_bytes()
        raw = gzip.decompress(compressed_bytes)
        records.append({
            "original_path": str(compressed.with_suffix("")),
            "compressed_path": str(compressed),
            "original_bytes": len(raw),
            "compressed_bytes": len(compressed_bytes),
            "original_sha256": digest(raw),
            "compressed_sha256": digest(compressed_bytes),
        })
    for source in sources:
        raw = source.read_bytes()
        compressed = source.with_suffix(source.suffix + ".gz")
        payload = gzip.compress(raw, compresslevel=6, mtime=0)
        if gzip.decompress(payload) != raw:
            raise RuntimeError(f"gzip verification failed for {source}")
        temporary = compressed.with_suffix(compressed.suffix + ".tmp")
        temporary.write_bytes(payload)
        if gzip.decompress(temporary.read_bytes()) != raw:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"on-disk gzip verification failed for {source}")
        temporary.replace(compressed)
        source.unlink()
        records.append({
            "original_path": str(source),
            "compressed_path": str(compressed),
            "original_bytes": len(raw),
            "compressed_bytes": len(payload),
            "original_sha256": digest(raw),
            "compressed_sha256": digest(payload),
        })

    records.sort(key=lambda item: item["original_path"])
    if (len(records) != args.expected_files
            or len({item["original_path"] for item in records}) != args.expected_files):
        raise RuntimeError("post-compression rollout cardinality changed")
    report = {
        "schema_version": 1,
        "evidence_tier": "lossless_storage_transform_not_experiment_result",
        "terminal_pairing_status": marker_record.get("fully_paired"),
        "root": str(root),
        "files": len(records),
        "original_bytes": sum(item["original_bytes"] for item in records),
        "compressed_bytes": sum(item["compressed_bytes"] for item in records),
        "all_decompressed_sha256_verified": True,
        "records": records,
    }
    output = root / "rollout_compression_manifest.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()

"""Freeze a deterministic model's real-input replay from the pinned LIBERO corpus.

The source observation membership, states, task strings, episode split, and
camera pixels are inherited unchanged. No random samples are fabricated for a
policy whose exercised action path consumes no request-local noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(source_path: Path, evidence_path: Path, model_id: str, reason: str) -> dict:
    source = json.loads(source_path.read_text())
    rows = source.get("observations", [])
    if (len(rows) != 1000 or len({row.get("id") for row in rows}) != 1000
            or Counter(row.get("split") for row in rows) != {"diagnostic": 200, "heldout": 800}
            or {row.get("replay_index") for row in rows} != set(range(1000))):
        raise ValueError("source is not the frozen 1,000-unique-observation LIBERO replay")
    if not evidence_path.is_file() or not reason.strip():
        raise ValueError("deterministic inference needs a real pinned implementation and reason")
    root = Path(source["fixture_root"])
    for row in rows:
        for camera, relative in row["images"].items():
            path = root / relative
            if not path.is_file() or sha256(path) != row["image_sha256"][camera]:
                raise ValueError(f"real replay image is missing or changed: {path}")
    observations = []
    for item in rows:
        row = {key: value for key, value in item.items() if key not in ("noise", "request_ids")}
        row["request_ids"] = [f"{row['id']}-deterministic"]
        observations.append(row)
    manifest = {
        key: value for key, value in source.items()
        if key not in ("observations", "noise_file", "noise_file_sha256", "noise_shape",
                       "checkpoint_config_sha256")
    }
    manifest.update({
        "model_id": model_id,
        "source_replay_manifest_path": str(source_path.resolve()),
        "source_replay_manifest_sha256": sha256(source_path),
        "source_variant_note": (
            "Public HuggingFaceVLA/libero embedded-image frames; the LeRobot MP4 "
            "release is separately compressed and is not claimed pixel-identical."
        ),
        "deterministic_decoding_contract": {
            "reason": reason,
            "evidence_path": str(evidence_path.resolve()),
            "evidence_sha256": sha256(evidence_path),
        },
        "observations": observations,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--inference-evidence", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = derive(args.source_manifest, args.inference_evidence, args.model_id, args.reason)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model_id": args.model_id, "observations": 1000,
                      "deterministic_requests_per_observation": 1,
                      "output_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()

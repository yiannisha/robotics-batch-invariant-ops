"""Derive a real, deterministic π0-FAST replay from the frozen LIBERO corpus.

The selected episodes, states, task strings, and camera frames are inherited
unchanged from the pinned public embedded-image LIBERO source. This is not a
claim of pixel identity with LeRobot's separately compressed MP4 release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--decoder-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source_manifest.read_text())
    if len(source.get("observations", [])) != 1000:
        raise ValueError("source replay has not frozen 1,000 real observations")
    if not args.decoder_evidence.is_file():
        raise FileNotFoundError(args.decoder_evidence)
    observations = []
    for item in source["observations"]:
        row = {key: value for key, value in item.items() if key not in ("noise", "request_ids")}
        row["request_ids"] = [f"{row['id']}-deterministic"]
        observations.append(row)
    manifest = {
        key: value for key, value in source.items()
        if key not in (
            "observations", "noise_file", "noise_file_sha256", "noise_shape",
            "checkpoint_config_sha256",
        )
    }
    manifest.update({
        "model_id": "pi_fast",
        "source_replay_manifest_path": str(args.source_manifest.resolve()),
        "source_replay_manifest_sha256": sha256(args.source_manifest),
        "source_variant_note": "Public HuggingFaceVLA/libero embedded-image frames; the LeRobot MP4 release is separately compressed and is not claimed pixel-identical.",
        "deterministic_decoding_contract": {
            "reason": "The executed π0-FAST adapter sets temperature=0.0 for greedy action-token decoding and consumes no request-local flow noise.",
            "evidence_path": str(args.decoder_evidence.resolve()),
            "evidence_sha256": sha256(args.decoder_evidence),
        },
        "observations": observations,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"observations": len(observations), "request_multiplicity": 1}))


if __name__ == "__main__":
    main()

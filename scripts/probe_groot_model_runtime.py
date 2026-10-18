"""Probe exact official GR00T N1.7 LIBERO policy construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from groot_runtime import BASE_MODEL, LOCAL_METADATA, load_official_libero_policy


CHECKPOINT_INDEX_SHA256 = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
SHARD_HASHES = {
    "model-00001-of-00002.safetensors": "39b7beaadd9a06c87502f2b741a36f939b84941f37fef757c6177e54e34a5eff",
    "model-00002-of-00002.safetensors": "8c4fa56f1b4f25a1a842c811e8a12d2bf6d77d73942f9a04f5b876a11bd6d703",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/workspace/assets/groot_n17/repo/libero_10"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/cross_model/groot_n17/full/closed_loop/model_runtime_probe.json"),
    )
    args = parser.parse_args()
    index = args.checkpoint / "model.safetensors.index.json"
    if sha256(index) != CHECKPOINT_INDEX_SHA256:
        raise RuntimeError("GR00T checkpoint index differs")
    inventory = []
    checkpoint_keys: set[str] = set()
    for name, expected_hash in SHARD_HASHES.items():
        path = args.checkpoint / name
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"GR00T checkpoint shard differs: {name}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        checkpoint_keys.update(keys)
        inventory.append(
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual_hash,
             "tensor_count": len(keys)}
        )

    torch.cuda.reset_peak_memory_stats()
    policy = load_official_libero_policy(args.checkpoint)
    model_keys = set(policy.model.state_dict())
    missing = sorted(checkpoint_keys - model_keys)
    unexpected = sorted(model_keys - checkpoint_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"GR00T checkpoint/model key mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    report = {
        "schema_version": 1,
        "evidence_tier": "exact_checkpoint_model_runtime_probe_not_episode_evidence",
        "model_id": "groot_n17",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "model_load_succeeded",
        "official_policy_class": type(policy).__module__ + "." + type(policy).__qualname__,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_index_sha256": CHECKPOINT_INDEX_SHA256,
        "checkpoint_files": inventory,
        "checkpoint_tensor_count": len(checkpoint_keys),
        "model_tensor_count": len(model_keys),
        "missing_checkpoint_keys": missing,
        "unexpected_model_keys": unexpected,
        "nested_base_identifier": BASE_MODEL,
        "nested_base_download_used": False,
        "local_metadata": str(LOCAL_METADATA.resolve()),
        "local_metadata_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(LOCAL_METADATA.iterdir()) if path.is_file()
        ],
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "final_model_dtype": str(next(policy.model.parameters()).dtype),
        "runtime_helper_path": str(Path(__file__).with_name("groot_runtime.py").resolve()),
        "runtime_helper_sha256": sha256(Path(__file__).with_name("groot_runtime.py")),
        "probe_source_sha256": sha256(Path(__file__)),
        "pid": os.getpid(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"checkpoint_files", "local_metadata_files"}}, indent=2))


if __name__ == "__main__":
    main()

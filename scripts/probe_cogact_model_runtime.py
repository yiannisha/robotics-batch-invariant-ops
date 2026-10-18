"""Load the exact CogACT-Small checkpoint and run one pinned model request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


EXPECTED_WEIGHT_SHA256 = "013cf0d4954e8a560064905ee03785ce3d0fa126697d4410cfde9a6754665e44"
EXPECTED_WEIGHT_BYTES = 30_215_225_166


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    weight = checkpoint / "checkpoints" / "CogACT-Small.pt"
    if not weight.is_file() or weight.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(f"CogACT-Small checkpoint is absent or truncated: {weight}")
    observed_weight_sha256 = sha256(weight)
    if observed_weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("CogACT-Small checkpoint SHA-256 differs")

    os.environ.update(
        {
            "COGACT_CHECKPOINT": str(checkpoint),
            "COGACT_SOURCE": "/workspace/models/cogact",
            "COGACT_OPENVLA_SOURCE": "/workspace/models/cogact-openvla",
            "COGACT_SAMPLE_DIR": "/workspace/assets/smolvla/fixture",
            "COGACT_LLAMA_METADATA": "/workspace/assets/cogact_llama2_metadata",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    from scripts.model_invariance.adapters.cogact_small import adapter

    from scripts.cogact_runtime import load_model_with_bf16_llama_construction

    policy = load_model_with_bf16_llama_construction(adapter)
    batch = adapter.compose_batch(adapter.load_example_inputs(1))
    with torch.inference_mode():
        output = adapter.run_model(policy, batch)
    array = np.ascontiguousarray(output.detach().cpu().numpy())
    if array.shape != (1, 16, 7) or not np.isfinite(array).all():
        raise RuntimeError(f"CogACT-Small output contract differs: {array.shape}")

    adapter_source = Path(__file__).resolve().parent / "model_invariance/adapters/cogact_small.py"
    official_source = Path("/workspace/models/cogact/vla/cogactvla.py")
    record = {
        "schema_version": 1,
        "model_id": "cogact_small",
        "status": "model_probe_succeeded",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_weight_path": str(weight),
        "checkpoint_weight_bytes": weight.stat().st_size,
        "checkpoint_weight_sha256": observed_weight_sha256,
        "request_local_noise": True,
        "llama_construction_dtype": "torch.bfloat16",
        "llama_random_weight_initialization": "skipped_before_strict_checkpoint_load",
        "final_vlm_dtype": "torch.bfloat16",
        "final_action_head_dtype": str(next(policy.model.action_model.parameters()).dtype),
        "noise_sha256": hashlib.sha256(
            np.ascontiguousarray(batch["noise"].numpy()).tobytes()
        ).hexdigest(),
        "output_shape": list(array.shape),
        "output_dtype": str(array.dtype),
        "output_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "adapter_source_path": str(adapter_source),
        "adapter_source_sha256": sha256(adapter_source),
        "official_model_source_path": str(official_source),
        "official_model_source_sha256": sha256(official_source),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".staging")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

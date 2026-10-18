"""Probe optional large-batch capacity on frozen real requests.

This is a capacity gate, not a substitute for all diagnostic/held-out
arrangements or the required 5x20x200 timing series at a feasible size.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, _run_once
from scripts.run_model_replay_invariance import metrics, sample_at, unrelated_companion_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(16, 32))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (sorted(set(args.batch_sizes)) != sorted(args.batch_sizes)
            or any(size not in (8, 16, 32) for size in args.batch_sizes)):
        raise ValueError("capacity probe accepts unique B8/B16/B32 sizes only")
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    observations = manifest["observations"]
    target_row = next(row for row in observations if row["split"] == "diagnostic")
    target_index = target_row["replay_index"]
    adapter = importlib.import_module(args.adapter).adapter
    provider_manifest_path = getattr(adapter, "replay_manifest_path", None)
    if provider_manifest_path is None and hasattr(adapter, "sample_dir"):
        provider_manifest_path = adapter.sample_dir / "replay_manifest.json"
    if provider_manifest_path is None or not Path(provider_manifest_path).is_file() or hashlib.sha256(Path(provider_manifest_path).read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("adapter's replay fixture differs from the frozen manifest")
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    adapter_execution_configuration = (
        adapter.execution_configuration() if hasattr(adapter, "execution_configuration") else None
    )
    target = adapter.load_replay_input(target_index, 0)
    schedule = (manifest.get("companion_schedule") or {}).get("version")
    if schedule == "distinct_source_observations_v1":
        companion_indexes = target_row.get("companion_replay_indices", [])[:max(args.batch_sizes) - 1]
        if (len(companion_indexes) != max(args.batch_sizes) - 1
                or len(set(companion_indexes)) != len(companion_indexes)
                or target_index in companion_indexes):
            raise ValueError("capacity probe has an invalid source-limited companion schedule")
        companion_rows = [observations[index] for index in companion_indexes]
    else:
        companion_rows = unrelated_companion_rows(
            observations, target_row, target_index, max(args.batch_sizes) - 1
        )
        if target_row.get("companion_replay_indices", [])[: len(companion_rows)] != [row["replay_index"] for row in companion_rows]:
            raise ValueError("capacity probe does not match frozen companion schedule")
    companions = [adapter.load_replay_input(row["replay_index"], 0) for row in companion_rows]
    records = []
    for size in args.batch_sizes:
        for implementation in ("native", "batch_invariant"):
            invariant = implementation == "batch_invariant"
            mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
            with mode:
                reference = _run_once(adapter, model, _compose_batch(adapter, [target]), args.seed)
                torch.cuda.synchronize()
                failed = False
                for companion_type in ("duplicate", "diverse"):
                    selected = [target] * (size - 1) if companion_type == "duplicate" else companions[: size - 1]
                    companion_ids = (
                        [target_row["request_ids"][0]] * (size - 1)
                        if companion_type == "duplicate"
                        else [row["request_ids"][0] for row in companion_rows[: size - 1]]
                    )
                    for position_name, target_position in (("first", 0), ("middle", size // 2), ("last", size - 1)):
                        batch_inputs = list(selected)
                        batch_inputs.insert(target_position, target)
                        try:
                            batch = _compose_batch(adapter, batch_inputs)
                            torch.cuda.reset_peak_memory_stats()
                            baseline = torch.cuda.memory_allocated()
                            output = _run_once(adapter, model, batch, args.seed)
                            torch.cuda.synchronize()
                            evidence = {
                                "status": "ran",
                                "peak_incremental_bytes": torch.cuda.max_memory_allocated() - baseline,
                                "output": metrics(reference, sample_at(output, target_position)),
                            }
                            del output, batch
                        except torch.cuda.OutOfMemoryError as error:
                            evidence = {
                                "status": "cuda_oom", "error": str(error),
                                "free_bytes_after_failure": torch.cuda.mem_get_info()[0],
                            }
                            gc.collect()
                            torch.cuda.empty_cache()
                            failed = True
                        records.append({
                            "schema_version": 1,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "adapter": args.adapter,
                            "adapter_execution_configuration": adapter_execution_configuration,
                            "replay_manifest_sha256": manifest_hash,
                            "request_id": target_row["request_ids"][0],
                            "gpu": torch.cuda.get_device_name(),
                            "torch": torch.__version__,
                            "implementation": implementation,
                            "batch_size": size,
                            "companion_type": companion_type,
                            "companion_ids": companion_ids,
                            "target_position": position_name,
                            **evidence,
                        })
                        if failed:
                            break
                    if failed:
                        break
            print(f"capacity {implementation} B={size}: {'OOM' if failed else 'ran all six'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(args.output)


if __name__ == "__main__":
    main()

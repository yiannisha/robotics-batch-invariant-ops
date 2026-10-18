"""Probe one frozen unrelated-companion replay pair without step tracing.

This isolates backend execution from the full runner's trace instrumentation.
Its output is exploratory and cannot satisfy the full diagnostic gate.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, _run_once
from scripts.run_model_replay_invariance import metrics, run_with_trace, sample_at


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--implementation", choices=("native", "batch_invariant"), required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeat-singleton", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 2:
        raise ValueError("output must be new and batch size must be at least two")
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = manifest["observations"]
    target_row = next(row for row in rows if row["split"] == "diagnostic")
    indexes = target_row["companion_replay_indices"][: args.batch_size - 1]
    if len(indexes) != args.batch_size - 1 or any(
        rows[index]["episode_id"] == target_row["episode_id"] for index in indexes
    ):
        raise ValueError("frozen unrelated-companion schedule is incomplete")
    adapter = importlib.import_module(args.adapter).adapter
    provider = getattr(adapter, "replay_manifest_path", None)
    if provider is None and hasattr(adapter, "sample_dir"):
        provider = adapter.sample_dir / "replay_manifest.json"
    if provider is None or hashlib.sha256(Path(provider).read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("adapter and selected frozen manifest differ")
    model = adapter.load_model()
    model.eval()
    target = adapter.load_replay_input(target_row["replay_index"], 0)
    companions = [adapter.load_replay_input(index, 0) for index in indexes]
    mode = set_batch_invariant_mode() if args.implementation == "batch_invariant" else contextlib.nullcontext()
    with mode:
        if args.trace:
            singleton, singleton_trace = run_with_trace(adapter, model, _compose_batch(adapter, [target]), 0)
            batched, batched_trace = run_with_trace(adapter, model, _compose_batch(adapter, [target, *companions]), 0)
        else:
            singleton = _run_once(adapter, model, _compose_batch(adapter, [target]), 0)
            batched = _run_once(adapter, model, _compose_batch(adapter, [target, *companions]), 0)
            singleton_trace = batched_trace = {}
        torch.cuda.synchronize()
        output = metrics(singleton, sample_at(batched, 0))
        repeated = None
        if args.repeat_singleton:
            if args.trace:
                repeated_output, _ = run_with_trace(adapter, model, _compose_batch(adapter, [target]), 0)
            else:
                repeated_output = _run_once(adapter, model, _compose_batch(adapter, [target]), 0)
            torch.cuda.synchronize()
            repeated = metrics(singleton, repeated_output)
    report = {
        "schema_version": 1,
        "evidence_tier": "probe_only",
        "adapter": args.adapter,
        "replay_manifest_sha256": manifest_hash,
        "request_id": target_row["request_ids"][0],
        "companion_replay_indices": indexes,
        "implementation": args.implementation,
        "batch_size": args.batch_size,
        "trace": args.trace,
        "repeat_singleton": args.repeat_singleton,
        "repeated_singleton_output": repeated,
        "singleton_trace_labels": sorted(singleton_trace),
        "batched_trace_labels": sorted(batched_trace),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "output": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"exact": output["exact"], "max_absolute_error": output["max_absolute_error"]}))


if __name__ == "__main__":
    main()

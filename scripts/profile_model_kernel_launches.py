"""Profile one trace-free representative call per policy and serving mode."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, _compose_fixed_batch, _run_once


def _parse_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item) for item in value.split(",") if item)
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("replay indices must be unique nonnegative integers")
    return indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile(function, *, kind: str, configuration: str, batch_size: int | None) -> dict:
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        function()
        torch.cuda.synchronize()
    kernels = [
        item.name for item in trace.events()
        if getattr(getattr(item, "device_type", None), "name", "") == "CUDA"
    ]
    if not kernels:
        raise RuntimeError(f"profiler captured no CUDA kernels for {kind} {configuration} B={batch_size}")
    aten = {
        item.key: int(item.count)
        for item in trace.key_averages() if item.key.startswith("aten::")
    }
    return {
        "kind": kind,
        "configuration": configuration,
        "batch_size": batch_size,
        "kernel_launch_count": len(kernels),
        "cuda_kernels": dict(Counter(kernels)),
        "aten_operators": aten,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--logical-batch-size", type=int, default=4)
    parser.add_argument("--fixed-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--replay-indices", type=_parse_indices, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not {1, 2, 4}.issubset(args.batch_sizes) or args.logical_batch_size > args.fixed_batch_size:
        raise ValueError("invalid full campaign batch-size matrix")
    needed = max([*args.batch_sizes, args.fixed_batch_size])
    if len(args.replay_indices) != needed:
        raise ValueError("replay indices must match the largest profiled batch")
    manifest = json.loads(args.replay_manifest.read_text())
    observations = manifest["observations"]
    if max(args.replay_indices) >= len(observations):
        raise ValueError("replay index exceeds frozen manifest")
    selected = [observations[index] for index in args.replay_indices]
    if any(row.get("replay_index") != index or row.get("split") != "diagnostic"
           for index, row in zip(args.replay_indices, selected)):
        raise ValueError("kernel profile inputs must use frozen diagnostic rows")
    input_selection = {
        "replay_manifest_path": str(args.replay_manifest.resolve()),
        "replay_manifest_sha256": _sha256(args.replay_manifest),
        "replay_indices": list(args.replay_indices),
        "episode_ids": [row.get("episode_id") for row in selected],
        "task_indices": [row.get("task_index") for row in selected],
        "request_ids": [row["request_ids"][0] for row in selected],
    }
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    adapter_execution_configuration = (
        adapter.execution_configuration() if hasattr(adapter, "execution_configuration") else None
    )
    examples = [adapter.load_replay_input(index, 0) for index in args.replay_indices]
    profiles = []
    for implementation in ("native", "batch_invariant"):
        for size in args.batch_sizes:
            mode = set_batch_invariant_mode() if implementation == "batch_invariant" else contextlib.nullcontext()
            with mode:
                batch = _compose_batch(adapter, examples[:size])
                profiles.append(profile(
                    lambda: _run_once(adapter, model, batch, args.seed),
                    kind="policy", configuration=implementation, batch_size=size,
                ))
            print(f"profiled policy {implementation} B={size}", flush=True)
    logical = args.logical_batch_size
    fixed = args.fixed_batch_size
    dynamic = _compose_batch(adapter, examples[:logical])
    padded = _compose_fixed_batch(adapter, examples[:logical] + [examples[0]] * (fixed - logical))
    fixed_shape_metadata = {"physical_batch_size": fixed}
    if hasattr(adapter, "fixed_batch_shape_metadata"):
        fixed_shape_metadata.update(adapter.fixed_batch_shape_metadata(padded))
    singletons = [_compose_batch(adapter, [example]) for example in examples[:logical]]
    functions = (
        ("native_dynamic", False, lambda: _run_once(adapter, model, dynamic, args.seed)),
        ("batch_invariant_dynamic", True, lambda: _run_once(adapter, model, dynamic, args.seed)),
        ("serial_singleton", False, lambda: [_run_once(adapter, model, item, args.seed) for item in singletons]),
        ("fixed_size_padded", False, lambda: _run_once(adapter, model, padded, args.seed)),
    )
    for name, invariant, function in functions:
        mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
        with mode:
            if invariant:
                invariant_dynamic = _compose_batch(adapter, examples[:logical])
                function = lambda: _run_once(adapter, model, invariant_dynamic, args.seed)
            profiles.append(profile(function, kind="serving", configuration=name, batch_size=None))
        print(f"profiled serving {name}", flush=True)
    report = {
        "schema_version": 1, "adapter": args.adapter,
        "adapter_execution_configuration": adapter_execution_configuration,
        "input_selection": input_selection,
        "fixed_shape_metadata": fixed_shape_metadata,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

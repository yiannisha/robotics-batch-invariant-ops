"""Benchmark dynamic, serial-singleton, and fixed-padding serving strategies."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path

import torch
import triton

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, _compose_fixed_batch, _run_once, seed_everything


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


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _existing_counts(path: Path) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    if not path.exists():
        return counts
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") == "timing":
                key = (record["configuration"], record["session"])
                counts[key] = counts.get(key, 0) + 1
    return counts


def _existing_environment(path: Path) -> dict | None:
    if not path.exists():
        return None
    environments = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") == "environment":
                environments.append(record)
    if len(environments) != 1:
        raise ValueError(f"expected exactly one environment record in {path}")
    return environments[0]


def _existing_fixed_shape(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as stream:
        records = [record for record in map(json.loads, stream) if record.get("record_type") == "fixed_shape"]
    if len(records) > 1:
        raise ValueError(f"expected at most one fixed-shape record in {path}")
    return records[0] if records else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--logical-batch-size", type=int, default=4)
    parser.add_argument("--fixed-batch-size", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-indices", type=_parse_indices)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-campaign", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.logical_batch_size <= args.fixed_batch_size:
        raise ValueError("require 1 <= logical batch size <= fixed batch size")
    if min(args.sessions, args.warmup, args.repetitions) < 1:
        raise ValueError("sessions, warmup, and repetitions must be positive")
    if args.full_campaign and (args.sessions, args.warmup, args.repetitions) != (5, 20, 200):
        raise ValueError("full campaign requires exactly 5 sessions, 20 warm-ups, 200 timed calls")
    if args.full_campaign and args.replay_manifest is None:
        raise ValueError("full campaign requires hash-pinned frozen replay inputs")
    if (args.replay_manifest is None) != (args.replay_indices is None):
        raise ValueError("--replay-manifest and --replay-indices must be supplied together")
    input_selection = None
    if args.replay_manifest is not None:
        if len(args.replay_indices) != args.fixed_batch_size:
            raise ValueError("replay indices must have exactly fixed_batch_size entries")
        manifest = json.loads(args.replay_manifest.read_text())
        observations = manifest["observations"]
        if max(args.replay_indices) >= len(observations):
            raise ValueError("replay index exceeds frozen manifest")
        selected = [observations[index] for index in args.replay_indices]
        if any(row.get("replay_index") != index or row.get("split") != "diagnostic"
               for index, row in zip(args.replay_indices, selected)):
            raise ValueError("performance replay selection must use frozen diagnostic rows")
        input_selection = {
            "replay_manifest_path": str(args.replay_manifest.resolve()),
            "replay_manifest_sha256": _sha256(args.replay_manifest),
            "replay_indices": list(args.replay_indices),
            "episode_ids": [row.get("episode_id") for row in selected],
            "task_indices": [row.get("task_index") for row in selected],
            "request_ids": [row["request_ids"][0] for row in selected],
        }
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)

    prior_environment = _existing_environment(args.output)
    if prior_environment is not None:
        expected = {
            "adapter": args.adapter,
            "logical_batch_size": args.logical_batch_size,
            "fixed_batch_size": args.fixed_batch_size,
            "sessions": args.sessions,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "campaign_protocol": "full_v1" if args.full_campaign else "pilot",
            "input_selection": input_selection,
        }
        for key, value in expected.items():
            if prior_environment.get(key, "pilot" if key == "campaign_protocol" else None) != value:
                raise ValueError(f"resume setting {key} differs from existing environment record")
    existing = _existing_counts(args.output)
    if not args.output.exists():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        _append(
            args.output,
            {
                "record_type": "environment",
                "adapter": args.adapter,
                "gpu": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "triton": triton.__version__,
                "logical_batch_size": args.logical_batch_size,
                "fixed_batch_size": args.fixed_batch_size,
                "sessions": args.sessions,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "seed": args.seed,
                "campaign_protocol": "full_v1" if args.full_campaign else "pilot",
                "input_selection": input_selection,
            },
        )

    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    adapter_execution_configuration = (
        adapter.execution_configuration() if hasattr(adapter, "execution_configuration") else None
    )
    examples = (
        [adapter.load_replay_input(index, 0) for index in args.replay_indices]
        if args.replay_indices is not None else
        list(adapter.load_example_inputs(args.fixed_batch_size))
    )
    dynamic = _compose_batch(adapter, examples[: args.logical_batch_size])
    singletons = [
        _compose_batch(adapter, [example])
        for example in examples[: args.logical_batch_size]
    ]
    padded_examples = examples[: args.logical_batch_size] + [examples[0]] * (
        args.fixed_batch_size - args.logical_batch_size
    )
    padded = _compose_fixed_batch(adapter, padded_examples)
    fixed_shape_metadata = {"physical_batch_size": args.fixed_batch_size}
    if hasattr(adapter, "fixed_batch_shape_metadata"):
        fixed_shape_metadata.update(adapter.fixed_batch_shape_metadata(padded))
    fixed_shape_record = {
        "record_type": "fixed_shape",
        "adapter_execution_configuration": adapter_execution_configuration,
        "metadata": fixed_shape_metadata,
    }
    previous_fixed_shape = _existing_fixed_shape(args.output)
    if previous_fixed_shape is not None and previous_fixed_shape != fixed_shape_record:
        raise ValueError("resume fixed-padded shape differs from recorded shape")
    if previous_fixed_shape is None:
        _append(args.output, fixed_shape_record)

    configurations = (
        ("native_dynamic", False, lambda: _run_once(adapter, model, dynamic, args.seed)),
        (
            "batch_invariant_dynamic",
            True,
            lambda: _run_once(adapter, model, dynamic, args.seed),
        ),
        (
            "serial_singleton",
            False,
            lambda: [
                _run_once(adapter, model, singleton, args.seed) for singleton in singletons
            ],
        ),
        ("fixed_size_padded", False, lambda: _run_once(adapter, model, padded, args.seed)),
    )

    for name, invariant, function in configurations:
        mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
        with mode:
            if invariant:
                # Some model-specific invariant repairs canonicalize input
                # shapes during composition, so compose inside the active
                # mode rather than reusing the native batch object.
                invariant_dynamic = _compose_batch(adapter, examples[: args.logical_batch_size])
                function = lambda: _run_once(adapter, model, invariant_dynamic, args.seed)
            for session in range(args.sessions):
                observed = existing.get((name, session), 0)
                if observed == args.repetitions:
                    continue
                if observed > args.repetitions:
                    raise RuntimeError(
                        f"session {(name, session)} has {observed}/{args.repetitions} records"
                    )
                attempt_id = uuid.uuid4().hex
                for warmup_index in range(args.warmup):
                    function()
                    if args.full_campaign:
                        torch.cuda.synchronize()
                        _append(
                            args.output,
                            {
                                "record_type": "warmup",
                                "adapter_execution_configuration": adapter_execution_configuration,
                                "configuration": name,
                                "session": session,
                                "warmup_index": warmup_index,
                                "resume_from_repetition": observed,
                                "attempt_id": attempt_id,
                            },
                        )
                torch.cuda.synchronize()
                for repetition in range(observed, args.repetitions):
                    torch.cuda.reset_peak_memory_stats()
                    baseline_bytes = torch.cuda.memory_allocated()
                    started = time.perf_counter_ns()
                    output = function()
                    torch.cuda.synchronize()
                    elapsed_ns = time.perf_counter_ns() - started
                    peak_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
                    del output
                    _append(
                        args.output,
                        {
                            "record_type": "timing",
                            "adapter_execution_configuration": adapter_execution_configuration,
                            "model": adapter.name,
                            "configuration": name,
                            "session": session,
                            "repetition": repetition,
                            "logical_batch_size": args.logical_batch_size,
                            "physical_batch_size": (
                                args.fixed_batch_size
                                if name == "fixed_size_padded"
                                else 1
                                if name == "serial_singleton"
                                else args.logical_batch_size
                            ),
                            "latency_seconds": elapsed_ns / 1_000_000_000,
                            "requests_per_second": (
                                args.logical_batch_size * 1_000_000_000 / elapsed_ns
                            ),
                            "peak_incremental_bytes": peak_bytes,
                        },
                    )
                print(
                    f"completed {name} session={session + 1}/{args.sessions}", flush=True
                )
    print(args.output)


if __name__ == "__main__":
    main()

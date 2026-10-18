"""Benchmark a model adapter's complete native and batch-invariant policy paths."""

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
from scripts.model_invariance.harness import _compose_batch, _run_once, seed_everything


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def _parse_implementations(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"native", "batch_invariant"}:
        raise argparse.ArgumentTypeError("implementations must be a unique subset of native,batch_invariant")
    return modes


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


def _existing_counts(path: Path) -> dict[tuple[str, int, int], int]:
    counts: dict[tuple[str, int, int], int] = {}
    if not path.exists():
        return counts
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "timing":
                continue
            key = (record["implementation"], record["batch_size"], record["session"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--batch-sizes", type=_parse_sizes, default=(1, 2, 8, 64))
    parser.add_argument("--implementations", type=_parse_implementations, default=("native", "batch_invariant"))
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
        if len(args.replay_indices) != max(args.batch_sizes):
            raise ValueError("replay indices must have exactly max(batch_sizes) entries")
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
        if tuple(prior_environment.get("implementations", ("native", "batch_invariant"))) != args.implementations:
            raise ValueError("resume implementations differ from existing environment record")
        expected = {
            "adapter": args.adapter,
            "batch_sizes": list(args.batch_sizes),
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
                "batch_sizes": list(args.batch_sizes),
                "implementations": list(args.implementations),
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
        list(adapter.load_example_inputs(max(args.batch_sizes)))
    )

    for implementation in args.implementations:
        invariant = implementation == "batch_invariant"
        mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
        with mode:
            for batch_size in args.batch_sizes:
                batch = _compose_batch(adapter, examples[:batch_size])
                for session in range(args.sessions):
                    key = (implementation, batch_size, session)
                    observed = existing.get(key, 0)
                    if observed == args.repetitions:
                        continue
                    if observed > args.repetitions:
                        raise RuntimeError(
                            f"session {key} has {observed}/{args.repetitions} records"
                        )
                    attempt_id = uuid.uuid4().hex
                    for warmup_index in range(args.warmup):
                        _run_once(adapter, model, batch, args.seed)
                        if args.full_campaign:
                            torch.cuda.synchronize()
                            _append(
                                args.output,
                                {
                                    "record_type": "warmup",
                                    "adapter_execution_configuration": adapter_execution_configuration,
                                    "implementation": implementation,
                                    "batch_size": batch_size,
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
                        output = _run_once(adapter, model, batch, args.seed)
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
                                "implementation": implementation,
                                "batch_size": batch_size,
                                "session": session,
                                "repetition": repetition,
                                "latency_seconds": elapsed_ns / 1_000_000_000,
                                "requests_per_second": batch_size * 1_000_000_000 / elapsed_ns,
                                "peak_incremental_bytes": peak_bytes,
                            },
                        )
                    print(
                        f"completed {implementation} B={batch_size} session={session + 1}/"
                        f"{args.sessions}",
                        flush=True,
                    )
    print(args.output)


if __name__ == "__main__":
    main()

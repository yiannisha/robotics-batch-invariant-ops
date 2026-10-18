"""Run a real wall-clock request queue under four serving alternatives.

The client arrival schedule is precommitted from a seeded exponential process.
Inference, batching, waiting, and completion happen on the measured clock;
request records are only committed after a whole load/mode group completes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import random
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, _compose_fixed_batch, _run_once, compare_outputs


MODES = ("native_dynamic", "batch_invariant_dynamic", "serial_singleton", "fixed_size_padded")
LOAD_RATIOS = {"underload": 0.25, "moderate": 0.55, "near_saturation": 0.9, "saturation": 1.3}


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


def select(value, index):
    if isinstance(value, torch.Tensor):
        return value[index : index + 1]
    if isinstance(value, dict):
        return {key: select(item, index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(select(item, index) for item in value)
    if isinstance(value, list):
        return [select(item, index) for item in value]
    raise TypeError(type(value))


def run(adapter, model, examples, seed, invariant=False, fixed_shape=False):
    mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
    with mode:
        batch = _compose_fixed_batch(adapter, examples) if fixed_shape else _compose_batch(adapter, examples)
        output = _run_once(adapter, model, batch, seed)
        torch.cuda.synchronize()
        return output


def references(adapter, model, examples, logical_batch, fixed_batch, seed):
    native = [run(adapter, model, [example], seed) for example in examples]
    invariant = [run(adapter, model, [example], seed, invariant=True) for example in examples]
    padded = {}
    for index, example in enumerate(examples):
        for position in range(logical_batch):
            batch = [examples[0]] * fixed_batch
            batch[position] = example
            padded[index, position] = select(run(adapter, model, batch, seed, fixed_shape=True), position)
    return native, invariant, padded


def timing_capacity(path: Path, logical_batch: int, input_selection: dict) -> tuple[float, dict]:
    values = []
    environments = []
    fixed_shapes = []
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("record_type") == "environment":
                environments.append(item)
            if item.get("record_type") == "fixed_shape":
                fixed_shapes.append(item)
            if item.get("record_type") == "timing" and item.get("configuration") == "native_dynamic":
                values.append(item["latency_seconds"])
    if len(environments) != 1 or environments[0].get("input_selection") != input_selection:
        raise ValueError("queue input selection differs from full serving-timing inputs")
    if len(fixed_shapes) != 1:
        raise ValueError("queue calibration lacks one fixed-padded shape record")
    if len(values) != 1000:
        raise ValueError("need the full 5x200 native-dynamic timing series to calibrate offered load")
    return logical_batch / statistics.median(values), fixed_shapes[0]["metadata"]


def schedule(count: int, rate: float, seed: int) -> list[float]:
    generator = random.Random(seed)
    arrivals = [0.0]
    for _ in range(count - 1):
        arrivals.append(arrivals[-1] + generator.expovariate(rate))
    return arrivals


def complete_groups(path: Path, count: int) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    records = Counter()
    keys = set()
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            group = (item["configuration"], item["offered_load"])
            request = (*group, item["request_id"])
            if request in keys:
                raise ValueError(f"duplicate queue request {request}")
            keys.add(request)
            records[group] += 1
    if any(value != count for value in records.values()):
        raise ValueError("existing queue output has a partial group; archive it before retrying")
    return set(records)


def run_group(
    adapter, model, examples, reference, *, configuration, load, arrivals,
    logical_batch, fixed_batch, seed, wait_seconds, offered_rate, capacity,
    input_selection,
):
    pending = deque()
    next_arrival = 0
    completed = 0
    batch_number = 0
    records = []
    executed_schedule = []
    started = time.perf_counter()
    request_count = len(arrivals)
    while completed < request_count:
        now = time.perf_counter() - started
        while next_arrival < request_count and arrivals[next_arrival] <= now:
            pending.append(next_arrival)
            next_arrival += 1
        if not pending:
            time.sleep(max(0.0, min(arrivals[next_arrival] - now, 0.01)))
            continue
        max_batch = 1 if configuration == "serial_singleton" else logical_batch
        if max_batch > 1 and len(pending) < max_batch and next_arrival < request_count:
            deadline = arrivals[pending[0]] + wait_seconds
            if now < deadline:
                time.sleep(max(0.0, min(deadline - now, arrivals[next_arrival] - now, 0.01)))
                continue
        request_numbers = [pending.popleft() for _ in range(min(max_batch, len(pending)))]
        indexes = [number % len(examples) for number in request_numbers]
        batch_examples = [examples[index] for index in indexes]
        if configuration == "fixed_size_padded":
            batch_examples += [examples[0]] * (fixed_batch - len(batch_examples))
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        service_started = time.perf_counter() - started
        output = run(
            adapter, model, batch_examples, seed,
            invariant=configuration == "batch_invariant_dynamic",
            fixed_shape=configuration == "fixed_size_padded",
        )
        service_completed = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
        del output
        executed_schedule.append((batch_number, request_numbers, indexes))
        for position, (number, index) in enumerate(zip(request_numbers, indexes, strict=True)):
            records.append({
                "schema_version": 1,
                "configuration": configuration,
                "offered_load": load,
                "offered_rate_requests_per_second": offered_rate,
                "reference_capacity_requests_per_second": capacity,
                "request_id": f"queue-{number:04d}",
                "fixture_index": index,
                "source_replay_index": input_selection["replay_indices"][index],
                "source_request_id": input_selection["request_ids"][index],
                "input_selection": input_selection,
                "arrival_time": arrivals[number],
                "batch_assignment": batch_number,
                "batch_occupancy": len(request_numbers),
                "physical_batch_size": len(batch_examples),
                "target_batch_position": position,
                "service_start": service_started,
                "completion_time": service_completed,
                "queueing_latency_seconds": service_started - arrivals[number],
                "service_latency_seconds": service_completed - service_started,
                "arrival_to_completion_seconds": service_completed - arrivals[number],
                "peak_incremental_bytes": peak_memory,
                "numerical_contract": "unverified",
                "numerical_max_abs_difference": None,
                "reference_type": "fixed_padded_same_position" if configuration == "fixed_size_padded" else "mode_matched_singleton",
            })
        completed += len(request_numbers)
        batch_number += 1
        print(f"{configuration} {load}: {completed}/{request_count}", end="\r", flush=True)
    print(f"{configuration} {load}: {completed} requests in {time.perf_counter() - started:.1f}s", flush=True)
    records_by_request = {record["request_id"]: record for record in records}
    for batch_number, request_numbers, indexes in executed_schedule:
        batch_examples = [examples[index] for index in indexes]
        if configuration == "fixed_size_padded":
            batch_examples += [examples[0]] * (fixed_batch - len(batch_examples))
        replay_output = run(
            adapter, model, batch_examples, seed,
            invariant=configuration == "batch_invariant_dynamic",
            fixed_shape=configuration == "fixed_size_padded",
        )
        for position, (number, index) in enumerate(zip(request_numbers, indexes, strict=True)):
            if configuration == "batch_invariant_dynamic":
                canonical = reference[1][index]
            elif configuration == "fixed_size_padded":
                canonical = reference[2][index, position]
            else:
                canonical = reference[0][index]
            comparisons = compare_outputs(canonical, select(replay_output, position))
            record = records_by_request[f"queue-{number:04d}"]
            record["numerical_contract"] = "exact" if all(item.exact for item in comparisons) else "violated"
            record["numerical_max_abs_difference"] = max(
                (item.max_abs_difference or 0.0) for item in comparisons
            )
            record["verification_batch_assignment"] = batch_number
    if any(record["numerical_contract"] == "unverified" for record in records):
        raise AssertionError("queue verification replay did not cover every request")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--serving-timings", type=Path, required=True)
    parser.add_argument("--logical-batch-size", type=int, default=4)
    parser.add_argument("--fixed-batch-size", type=int, default=8)
    parser.add_argument("--requests-per-group", type=int, default=250)
    parser.add_argument("--batching-window-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--replay-indices", type=_parse_indices, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.requests_per_group < 250:
        raise ValueError("full queue campaign requires at least 250 requests per load and mode")
    if not 1 <= args.logical_batch_size <= args.fixed_batch_size:
        raise ValueError("invalid logical/fixed batch sizes")
    if len(args.replay_indices) != args.fixed_batch_size:
        raise ValueError("replay indices must have exactly fixed_batch_size entries")
    manifest = json.loads(args.replay_manifest.read_text())
    observations = manifest["observations"]
    if max(args.replay_indices) >= len(observations):
        raise ValueError("replay index exceeds frozen manifest")
    selected = [observations[index] for index in args.replay_indices]
    if any(row.get("replay_index") != index or row.get("split") != "diagnostic"
           for index, row in zip(args.replay_indices, selected)):
        raise ValueError("queue input selection must use frozen diagnostic rows")
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
    if args.output.exists():
        with args.output.open() as stream:
            if any(json.loads(line).get("input_selection") != input_selection for line in stream):
                raise ValueError("queue resume selection differs from recorded frozen inputs")
    completed = complete_groups(args.output, args.requests_per_group)
    capacity, timed_fixed_shape = timing_capacity(args.serving_timings, args.logical_batch_size, input_selection)
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    adapter_execution_configuration = (
        adapter.execution_configuration() if hasattr(adapter, "execution_configuration") else None
    )
    examples = [adapter.load_replay_input(index, 0) for index in args.replay_indices]
    padded_examples = examples[:args.logical_batch_size] + [examples[0]] * (args.fixed_batch_size - args.logical_batch_size)
    padded_batch = _compose_fixed_batch(adapter, padded_examples)
    fixed_shape_metadata = {"physical_batch_size": args.fixed_batch_size}
    if hasattr(adapter, "fixed_batch_shape_metadata"):
        fixed_shape_metadata.update(adapter.fixed_batch_shape_metadata(padded_batch))
    if fixed_shape_metadata != timed_fixed_shape:
        raise ValueError("queue fixed-padded input shape differs from full serving timing")
    reference = references(
        adapter, model, examples, args.logical_batch_size,
        args.fixed_batch_size, args.seed,
    )
    for load_index, (load, ratio) in enumerate(LOAD_RATIOS.items()):
        rate = capacity * ratio
        arrivals = schedule(args.requests_per_group, rate, args.seed + load_index)
        for configuration in MODES:
            if (configuration, load) in completed:
                continue
            rows = run_group(
                adapter, model, examples, reference,
                configuration=configuration, load=load, arrivals=arrivals,
                logical_batch=args.logical_batch_size,
                fixed_batch=args.fixed_batch_size, seed=args.seed,
                wait_seconds=args.batching_window_ms / 1000,
                offered_rate=rate, capacity=capacity,
                input_selection=input_selection,
            )
            for row in rows:
                row["adapter_execution_configuration"] = adapter_execution_configuration
                row["fixed_shape_metadata"] = fixed_shape_metadata if configuration == "fixed_size_padded" else None
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a") as stream:
                stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(args.output)


if __name__ == "__main__":
    main()

"""Run frozen-request singleton-vs-batch comparisons with resumable raw records.

Adapters must provide ``load_replay_input(index, noise_id)`` and a manifest
with frozen unrelated-companion indices. Diagnostic subsets are permitted,
but held-out execution requires a numerical freeze record.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, _flatten_tensors, _run_once


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metrics(reference, candidate) -> dict:
    reference_tensors = _flatten_tensors(reference)
    candidate_tensors = _flatten_tensors(candidate)
    if reference_tensors.keys() != candidate_tensors.keys():
        raise ValueError("reference and candidate output structures differ")
    per_tensor = {}
    total_elements = unequal = nonfinite = 0
    sum_abs = sum_squared = sum_reference_squared = 0.0
    maximum = 0.0
    exact = True
    for name, reference_tensor in reference_tensors.items():
        candidate_tensor = candidate_tensors[name]
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"shape mismatch for {name}")
        left = reference_tensor.detach().to(torch.float64)
        right = candidate_tensor.detach().to(torch.float64)
        same = torch.eq(left, right)
        finite = torch.isfinite(left) & torch.isfinite(right)
        delta = torch.where(finite, right - left, torch.zeros_like(right))
        absolute = delta.abs()
        n = left.numel()
        bad = int((~same).sum().item())
        not_finite = int((~finite).sum().item())
        squared = float(delta.square().sum().item())
        reference_squared = float(torch.where(finite, left, 0).square().sum().item())
        max_abs = float(absolute.max().item()) if n else 0.0
        per_tensor[name] = {
            "shape": list(left.shape),
            "exact": bool(torch.equal(left, right)),
            "unequal_element_count": bad,
            "max_absolute_error": max_abs,
            "rmse": math.sqrt(squared / max(n - not_finite, 1)),
            "nonfinite_count": not_finite,
        }
        exact &= per_tensor[name]["exact"]
        total_elements += n
        unequal += bad
        nonfinite += not_finite
        sum_abs += float(absolute.sum().item())
        sum_squared += squared
        sum_reference_squared += reference_squared
        maximum = max(maximum, max_abs)
    return {
        "exact": exact,
        "unequal_element_count": unequal,
        "unequal_element_fraction": unequal / total_elements if total_elements else 0.0,
        "max_absolute_error": maximum,
        "mean_absolute_error": sum_abs / max(total_elements - nonfinite, 1),
        "rmse": math.sqrt(sum_squared / max(total_elements - nonfinite, 1)),
        "relative_l2_error": math.sqrt(sum_squared / max(sum_reference_squared, 1e-300)),
        "nonfinite_count": nonfinite,
        "tensors": per_tensor,
    }


def sample_at(value, index):
    if isinstance(value, torch.Tensor):
        return value[index : index + 1]
    if isinstance(value, dict):
        return {key: sample_at(item, index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(sample_at(item, index) for item in value)
    if isinstance(value, list):
        return [sample_at(item, index) for item in value]
    raise TypeError(type(value))


def positions(size):
    return (("first", 0), ("last", 1)) if size == 2 else (
        ("first", 0), ("middle", size // 2), ("last", size - 1)
    )


def unrelated_companion_rows(observations, target_row, target_location, count):
    """Prefer distinct other tasks, then distinct other episodes.

    Adjacent frames of the target episode are not diverse observations, even
    though their pixel arrays differ.  Choose one episode head per other task
    before filling larger batches with additional non-target episodes.
    """
    episode_heads = {}
    for row in observations:
        episode_heads.setdefault((row["task"], row["episode_id"]), row)
    by_task = {}
    for row in episode_heads.values():
        if row["task"] != target_row["task"]:
            by_task.setdefault(row["task"], []).append(row)
    chosen = []
    used = set()
    for task in sorted(by_task):
        options = by_task[task]
        row = options[target_location % len(options)]
        chosen.append(row)
        used.add((row["task"], row["episode_id"]))
        if len(chosen) == count:
            return chosen
    remaining = []
    for same_task in (False, True):
        pool = [
            row for row in episode_heads.values()
            if (row["task"], row["episode_id"]) not in used
            and (row["task"], row["episode_id"])
            != (target_row["task"], target_row["episode_id"])
            and (row["task"] == target_row["task"]) == same_task
        ]
        if pool:
            rotation = target_location % len(pool)
            remaining.extend(pool[rotation:] + pool[:rotation])
    chosen.extend(remaining[: count - len(chosen)])
    if len(chosen) != count:
        raise ValueError(f"need {count} unrelated companion episodes, found {len(chosen)}")
    return chosen


def arrangements(target, target_id, companions, companion_ids, sizes):
    yield "singleton", "none", "first", [target], 0, [target_id], 1
    for size in sizes:
        if size == 1:
            continue
        diverse = companions[: size - 1]
        diverse_ids = companion_ids[: size - 1]
        if len(diverse) != size - 1:
            raise ValueError(f"need {size - 1} diverse companions")
        for companion_type in ("duplicate", "diverse"):
            selected = [target] * (size - 1) if companion_type == "duplicate" else diverse
            selected_ids = [target_id] * (size - 1) if companion_type == "duplicate" else diverse_ids
            for name, position in positions(size):
                batch = list(selected)
                ids = list(selected_ids)
                batch.insert(position, target)
                ids.insert(position, target_id)
                yield "batch_position", companion_type, name, batch, position, ids, size
        # Reverse the complete request set: unlike reversing only companions,
        # this is an actual permutation even when B=2.
        batch = list(reversed([target, *diverse]))
        ids = list(reversed([target_id, *diverse_ids]))
        yield "permutation", "diverse", "last", batch, size - 1, ids, size
        yield "partition", "diverse", "first", [target, *diverse], 0, [target_id, *diverse_ids], size


def append_many(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode()
    if path.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def completed(path: Path, manifest_hash: str, freeze_hash: str | None) -> set[tuple]:
    keys = set()
    if not path.exists():
        return keys
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("replay_manifest_sha256") != manifest_hash or item.get("numerical_freeze_sha256") != freeze_hash:
                raise ValueError("existing run uses another replay manifest or numerical freeze")
            key = (
                item["request_id"], item["implementation"], item["transformation"],
                item["logical_batch_size"], item["target_batch_position"], item["companion_type"],
            )
            if key in keys:
                raise ValueError(f"duplicate existing comparison: {key}")
            keys.add(key)
    return keys


def run_with_trace(adapter, model, batch, seed, *, capture_trace=True):
    trace = {}
    if hasattr(adapter, "trace_callback"):
        adapter.trace_callback = (lambda label, tensor: (
            trace.__setitem__(label, tensor.detach().clone())
            if label.startswith(("flow.step_", "actions.")) else None
        )) if capture_trace else None
    try:
        output = _run_once(adapter, model, batch, seed)
        torch.cuda.synchronize()
    finally:
        if hasattr(adapter, "trace_callback"):
            adapter.trace_callback = None
    return output, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fidelity-output", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--split", choices=("diagnostic", "heldout"), required=True)
    parser.add_argument("--modes", nargs="+", default=("native", "batch_invariant"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--verbose-arrangements", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if set(args.modes) - {"native", "batch_invariant"} or len(set(args.modes)) != len(args.modes):
        raise ValueError("modes must be a unique subset of native,batch_invariant")
    if (sorted(set(args.batch_sizes)) != sorted(args.batch_sizes)
            or not {1, 2, 4}.issubset(args.batch_sizes)):
        raise ValueError("batch sizes must include 1,2,4 without duplicates")
    if args.split == "heldout" and (not args.freeze or not args.freeze.is_file()):
        raise ValueError("held-out work requires a numerical freeze record")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if args.fidelity_output and args.fidelity_output.exists() and not args.resume:
        raise FileExistsError(args.fidelity_output)
    manifest_hash = digest(args.manifest)
    freeze_hash = digest(args.freeze) if args.freeze else None
    if args.freeze:
        freeze = json.loads(args.freeze.read_text())
        hashes = freeze.get("content_addressed_freeze")
        if not hashes:
            raise ValueError("numerical freeze lacks content-addressed source/checkpoint hashes")
        for filename, expected in hashes.items():
            path = Path(filename)
            if not path.is_file() or path.stat().st_size != expected.get("bytes") or digest(path) != expected.get("sha256"):
                raise ValueError(f"frozen numerical input changed or disappeared: {filename}")
        if str(args.manifest.resolve()) not in hashes:
            raise ValueError("numerical freeze does not bind the selected replay manifest")
    manifest = json.loads(args.manifest.read_text())
    companion_schedule = manifest.get("companion_schedule") or {}
    supported_schedules = {
        "unrelated_episode_heads_v1",
        "distinct_source_observations_v1",
    }
    if (companion_schedule.get("version") not in supported_schedules
            or companion_schedule.get("max_companions", 0) < max(args.batch_sizes) - 1):
        raise ValueError("replay manifest lacks a frozen unrelated-companion schedule")
    observations = manifest["observations"]
    if all(len(item.get("noise", [])) == 3 and len(item.get("request_ids", [])) == 3 for item in observations):
        multiplicity = 3
    else:
        contract = manifest.get("deterministic_decoding_contract") or {}
        if not isinstance(contract, dict):
            raise ValueError("deterministic decoding contract must be a pinned evidence object")
        evidence = Path(contract.get("evidence_path", ""))
        if not all(len(item.get("request_ids", [])) == 1 for item in observations) or not contract.get("reason") or not evidence.is_file() or digest(evidence) != contract.get("evidence_sha256"):
            raise ValueError("replay lacks three explicit noises or a pinned deterministic decoding contract")
        multiplicity = 1
    selected = [row for row in observations if row["split"] == args.split]
    if args.limit_requests is not None:
        selected = selected[: math.ceil(args.limit_requests / multiplicity)]
    done = completed(args.output, manifest_hash, freeze_hash)
    done_counts = Counter((key[0], key[1]) for key in done)
    expected_arrangements = 1 + sum(
        2 * len(positions(size)) + 2 for size in args.batch_sizes if size > 1
    )
    fidelity_done = set()
    if args.fidelity_output and args.fidelity_output.exists():
        opener = gzip.open if args.fidelity_output.suffix == ".gz" else open
        with opener(args.fidelity_output, "rt") as stream:
            for line in stream:
                item = json.loads(line)
                if item.get("replay_manifest_sha256") != manifest_hash or item.get("numerical_freeze_sha256") != freeze_hash:
                    raise ValueError("existing fidelity run uses another replay manifest or numerical freeze")
                if item["request_id"] in fidelity_done:
                    raise ValueError(f"duplicate existing fidelity request: {item['request_id']}")
                fidelity_done.add(item["request_id"])

    adapter = importlib.import_module(args.adapter).adapter
    if not hasattr(adapter, "load_replay_input"):
        raise TypeError("adapter lacks lazy load_replay_input(index, noise_id)")
    provider_manifest_path = getattr(adapter, "replay_manifest_path", None)
    if provider_manifest_path is None and hasattr(adapter, "sample_dir"):
        provider_manifest_path = adapter.sample_dir / "replay_manifest.json"
    if provider_manifest_path is None or not Path(provider_manifest_path).is_file() or digest(Path(provider_manifest_path)) != manifest_hash:
        raise ValueError("adapter's real-input replay manifest differs from the recorded manifest")
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    adapter_execution_configuration = (
        adapter.execution_configuration() if hasattr(adapter, "execution_configuration") else None
    )
    if args.freeze and freeze.get("adapter_execution_configuration") != adapter_execution_configuration:
        raise ValueError("held-out adapter execution configuration differs from the numerical freeze")
    for prior_path in (args.output, args.fidelity_output):
        if prior_path is None or not prior_path.is_file():
            continue
        opener = gzip.open if prior_path.suffix == ".gz" else open
        with opener(prior_path, "rt") as stream:
            for line in stream:
                if json.loads(line).get("adapter_execution_configuration") != adapter_execution_configuration:
                    raise ValueError(f"existing run uses another adapter execution configuration: {prior_path}")
    completed_requests = 0
    for row in selected:
        for noise_id in range(multiplicity):
            if args.limit_requests is not None and completed_requests >= args.limit_requests:
                break
            target_id = row["request_ids"][noise_id]
            if all(done_counts[(target_id, mode)] == expected_arrangements for mode in args.modes) and (
                not args.fidelity_output or target_id in fidelity_done
            ):
                completed_requests += 1
                continue
            target = adapter.load_replay_input(row["replay_index"], noise_id)
            target_location = row["replay_index"]
            if companion_schedule["version"] == "distinct_source_observations_v1":
                companion_indexes = row.get("companion_replay_indices", [
                ])[: max(args.batch_sizes) - 1]
                if (len(companion_indexes) != max(args.batch_sizes) - 1
                        or len(set(companion_indexes)) != len(companion_indexes)
                        or row["replay_index"] in companion_indexes
                        or any(not isinstance(index, int) or not 0 <= index < len(observations)
                               for index in companion_indexes)):
                    raise ValueError(f"invalid source-limited companion schedule for {target_id}")
                companion_rows = [observations[index] for index in companion_indexes]
            else:
                companion_rows = unrelated_companion_rows(
                    observations, row, target_location, max(args.batch_sizes) - 1
                )
                if row.get("companion_replay_indices", [])[: len(companion_rows)] != [
                    item["replay_index"] for item in companion_rows
                ]:
                    raise ValueError(f"frozen companion schedule differs for {target_id}")
            companions = [
                adapter.load_replay_input(item["replay_index"], noise_id)
                for item in companion_rows
            ]
            companion_ids = [item["request_ids"][noise_id] for item in companion_rows]
            singleton_outputs = {}
            pending_records = []
            for implementation in args.modes:
                invariant = implementation == "batch_invariant"
                mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
                with mode:
                    reference, reference_trace = run_with_trace(
                        adapter, model, _compose_batch(adapter, [target]), args.seed
                    )
                    singleton_outputs[implementation] = reference
                    for transformation, companion_type, position_name, requests, target_position, ordering, logical_size in arrangements(
                        target, target_id, companions, companion_ids, args.batch_sizes
                    ):
                        key = (target_id, implementation, transformation, logical_size, position_name, companion_type)
                        if key in done:
                            continue
                        if args.verbose_arrangements:
                            print(f"running {target_id} {implementation} {transformation} B={logical_size} {position_name} {companion_type}", flush=True)
                        started = time.perf_counter()
                        if transformation == "partition":
                            split = max(1, logical_size // 2)
                            candidate, candidate_trace = run_with_trace(
                                adapter, model, _compose_batch(adapter, requests[:split]), args.seed
                            )
                            # Execute the other partition as well: the recorded
                            # schedule must cover the same complete logical set.
                            run_with_trace(
                                adapter, model, _compose_batch(adapter, requests[split:]),
                                args.seed, capture_trace=False,
                            )
                            physical_size = split
                        else:
                            candidate, candidate_trace = run_with_trace(
                                adapter, model, _compose_batch(adapter, requests), args.seed
                            )
                            physical_size = len(requests)
                        elapsed = time.perf_counter() - started
                        output_metrics = metrics(reference, sample_at(candidate, target_position))
                        flow_metrics = {
                            label: metrics(reference_trace[label], sample_at(value, target_position))
                            for label, value in candidate_trace.items()
                            if label in reference_trace
                        }
                        pending_records.append(
                            {
                                "schema_version": 1,
                                "experiment_id": str(uuid.uuid4()),
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "replay_manifest_sha256": manifest_hash,
                                "numerical_freeze_sha256": freeze_hash,
                                "adapter": args.adapter,
                                "adapter_execution_configuration": adapter_execution_configuration,
                                "gpu": torch.cuda.get_device_name(),
                                "torch": torch.__version__,
                                "dtype": str(next(iter(_flatten_tensors(candidate).values())).dtype),
                                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "execution_mode": "eager",
                                "request_id": target_id,
                                "task": row["task"],
                                "episode_id": row["episode_id"],
                                "observation_index": row["observation_index"],
                                "noise_id": noise_id,
                                "source_image_sha256": row.get(
                                    "image_sha256", row.get("source_image_rgb_sha256")
                                ),
                                "request_noise_sha256": row.get("noise", [])[noise_id]["sha256"] if multiplicity == 3 else None,
                                "implementation": implementation,
                                "transformation": transformation,
                                "batch_size": physical_size,
                                "logical_batch_size": logical_size,
                                "target_batch_position": position_name,
                                "target_batch_index": target_position,
                                "companion_type": companion_type,
                                "companion_ids": [identifier for index, identifier in enumerate(ordering) if index != target_position],
                                "request_ordering": ordering,
                                "partition_id": f"first-{physical_size}-of-{logical_size}" if transformation == "partition" else "single_batch",
                                "latency_seconds": elapsed,
                                "output": output_metrics,
                                "flow_steps": flow_metrics,
                            }
                        )
                        done.add(key)
                        done_counts[(target_id, implementation)] += 1
            append_many(args.output, pending_records)
            if args.fidelity_output and {"native", "batch_invariant"} <= set(singleton_outputs) and target_id not in fidelity_done:
                append_many(
                    args.fidelity_output,
                    [{
                        "schema_version": 1,
                        "experiment_id": str(uuid.uuid4()),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "request_id": target_id,
                        "task": row["task"],
                        "episode_id": row["episode_id"],
                        "observation_index": row["observation_index"],
                        "noise_id": noise_id,
                        "source_image_sha256": row.get(
                            "image_sha256", row.get("source_image_rgb_sha256")
                        ),
                        "request_noise_sha256": row.get("noise", [])[noise_id]["sha256"] if multiplicity == 3 else None,
                        "replay_manifest_sha256": manifest_hash,
                        "adapter_execution_configuration": adapter_execution_configuration,
                        "numerical_freeze_sha256": freeze_hash,
                        "comparison": "batch_invariant_singleton_vs_native_singleton",
                        "output": metrics(singleton_outputs["native"], singleton_outputs["batch_invariant"]),
                    }],
                )
                fidelity_done.add(target_id)
            completed_requests += 1
            print(f"completed {args.split} {completed_requests} request-noise pairs", flush=True)
    print(args.output)


if __name__ == "__main__":
    main()

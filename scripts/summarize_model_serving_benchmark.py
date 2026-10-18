"""Summarize raw model-serving timing records across independent sessions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    environment = None
    fixed_shape = None
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with args.input.open() as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if record.get("record_type") == "environment":
                if environment is not None:
                    raise ValueError("multiple environment records")
                environment = record
            elif record.get("record_type") == "fixed_shape":
                if fixed_shape is not None:
                    raise ValueError("multiple fixed-padded shape records")
                fixed_shape = record.get("metadata")
            elif record.get("record_type") == "timing":
                grouped[(record["configuration"], record["session"])].append(record)
            elif record.get("record_type") == "warmup":
                continue
            else:
                raise ValueError(f"unexpected record at line {line_number}")
    if environment is None:
        raise ValueError("missing environment record")
    if fixed_shape is None:
        raise ValueError("missing fixed-padded shape record")

    expected_configurations = {
        "native_dynamic",
        "batch_invariant_dynamic",
        "serial_singleton",
        "fixed_size_padded",
    }
    expected_keys = {
        (configuration, session)
        for configuration in expected_configurations
        for session in range(environment["sessions"])
    }
    if set(grouped) != expected_keys:
        raise ValueError(
            f"configuration mismatch: missing={sorted(expected_keys - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_keys)}"
        )

    sessions = []
    repetitions = int(environment["repetitions"])
    for (configuration, session), records in sorted(grouped.items()):
        if len(records) != repetitions:
            raise ValueError(
                f"incomplete {configuration} session={session}: "
                f"{len(records)}/{repetitions}"
            )
        if sorted(record["repetition"] for record in records) != list(range(repetitions)):
            raise ValueError(f"non-contiguous repetitions for {configuration} session={session}")
        latencies = [record["latency_seconds"] for record in records]
        median_latency = statistics.median(latencies)
        sessions.append(
            {
                "configuration": configuration,
                "session": session,
                "samples": len(records),
                "median_latency_seconds": median_latency,
                "p95_latency_seconds": _percentile(latencies, 0.95),
                "requests_per_second": (
                    environment["logical_batch_size"] / median_latency
                ),
                "peak_incremental_bytes": max(
                    record["peak_incremental_bytes"] for record in records
                ),
            }
        )

    configurations = []
    by_configuration: dict[str, list[dict]] = defaultdict(list)
    for record in sessions:
        by_configuration[record["configuration"]].append(record)
    for configuration, records in sorted(by_configuration.items()):
        median_latency = statistics.median(
            record["median_latency_seconds"] for record in records
        )
        configurations.append(
            {
                "configuration": configuration,
                "median_of_session_medians_latency_seconds": median_latency,
                "p95_of_session_p95_latency_seconds": _percentile(
                    [record["p95_latency_seconds"] for record in records], 0.95
                ),
                "requests_per_second": (
                    environment["logical_batch_size"] / median_latency
                ),
                "peak_incremental_bytes": max(
                    record["peak_incremental_bytes"] for record in records
                ),
            }
        )

    indexed = {record["configuration"]: record for record in configurations}
    native = indexed["native_dynamic"]
    comparisons = [
        {
            "configuration": record["configuration"],
            "latency_over_native_dynamic": (
                record["median_of_session_medians_latency_seconds"]
                / native["median_of_session_medians_latency_seconds"]
            ),
            "throughput_over_native_dynamic": (
                record["requests_per_second"] / native["requests_per_second"]
            ),
        }
        for record in configurations
    ]
    report = {
        "schema_version": 1,
        "source": str(args.input),
        "environment": environment,
        "fixed_shape_metadata": fixed_shape,
        "session_summaries": sessions,
        "configurations": configurations,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

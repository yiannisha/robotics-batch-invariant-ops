"""Summarize raw complete-policy timing records without discarding sessions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
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

    environments = []
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    with args.input.open() as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if record.get("record_type") == "environment":
                environments.append(record)
            elif record.get("record_type") == "timing":
                grouped[
                    (record["implementation"], record["batch_size"], record["session"])
                ].append(record)
            elif record.get("record_type") == "warmup":
                continue
            else:
                raise ValueError(f"unexpected record at line {line_number}")
    if len(environments) != 1:
        raise ValueError(f"expected one environment record, found {len(environments)}")
    environment = environments[0]

    sessions = []
    expected_repetitions = environment["repetitions"]
    for (implementation, batch_size, session), records in sorted(grouped.items()):
        if len(records) != expected_repetitions:
            raise ValueError(
                f"incomplete {implementation} B={batch_size} session={session}: "
                f"{len(records)}/{expected_repetitions}"
            )
        repetitions = sorted(record["repetition"] for record in records)
        if repetitions != list(range(expected_repetitions)):
            raise ValueError(
                f"non-contiguous repetitions for {implementation} B={batch_size} "
                f"session={session}"
            )
        latencies = [record["latency_seconds"] for record in records]
        sessions.append(
            {
                "implementation": implementation,
                "batch_size": batch_size,
                "session": session,
                "samples": len(records),
                "median_latency_seconds": statistics.median(latencies),
                "p95_latency_seconds": _percentile(latencies, 0.95),
                "requests_per_second": batch_size / statistics.median(latencies),
                "peak_incremental_bytes": max(
                    record["peak_incremental_bytes"] for record in records
                ),
            }
        )

    expected_keys = {
        (implementation, batch_size, session)
        for implementation in ("native", "batch_invariant")
        for batch_size in environment["batch_sizes"]
        for session in range(environment["sessions"])
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise ValueError(f"configuration mismatch: missing={missing}, extra={extra}")

    configurations = []
    by_configuration: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for session in sessions:
        by_configuration[(session["implementation"], session["batch_size"])].append(session)
    for (implementation, batch_size), records in sorted(by_configuration.items()):
        median_latency = statistics.median(
            record["median_latency_seconds"] for record in records
        )
        configurations.append(
            {
                "implementation": implementation,
                "batch_size": batch_size,
                "median_of_session_medians_latency_seconds": median_latency,
                "p95_of_session_p95_latency_seconds": _percentile(
                    [record["p95_latency_seconds"] for record in records], 0.95
                ),
                "requests_per_second": batch_size / median_latency,
                "peak_incremental_bytes": max(
                    record["peak_incremental_bytes"] for record in records
                ),
            }
        )

    indexed = {
        (record["implementation"], record["batch_size"]): record
        for record in configurations
    }
    comparisons = []
    for batch_size in environment["batch_sizes"]:
        native = indexed[("native", batch_size)]
        invariant = indexed[("batch_invariant", batch_size)]
        comparisons.append(
            {
                "batch_size": batch_size,
                "invariant_over_native_latency": (
                    invariant["median_of_session_medians_latency_seconds"]
                    / native["median_of_session_medians_latency_seconds"]
                ),
                "invariant_over_native_throughput": (
                    invariant["requests_per_second"] / native["requests_per_second"]
                ),
                "invariant_minus_native_peak_incremental_bytes": (
                    invariant["peak_incremental_bytes"]
                    - native["peak_incremental_bytes"]
                ),
            }
        )

    report = {
        "schema_version": 1,
        "source": str(args.input),
        "environment": environment,
        "session_summaries": sessions,
        "configurations": configurations,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

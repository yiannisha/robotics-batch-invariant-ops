"""Summarize retained per-request wall-clock queue measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty metric series")
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(items: list[dict]) -> dict:
    arrivals = [item["arrival_time"] for item in items]
    completions = [item["completion_time"] for item in items]
    duration = max(completions) - min(arrivals)
    if duration <= 0:
        raise ValueError("nonpositive campaign duration")
    occupancy = Counter()
    for item in items:
        occupancy[item["batch_assignment"]] = item["batch_occupancy"]
    if any(value != sum(item["batch_assignment"] == batch for item in items) for batch, value in occupancy.items()):
        raise ValueError("batch occupancy records disagree with assignments")
    arrival_to_completion = [item["arrival_to_completion_seconds"] for item in items]
    queueing = [item["queueing_latency_seconds"] for item in items]
    service = [item["service_latency_seconds"] for item in items]
    reference_capacity = items[0]["reference_capacity_requests_per_second"]
    throughput = len(items) / duration
    return {
        "configuration": items[0]["configuration"],
        "offered_load": items[0]["offered_load"],
        "offered_rate_requests_per_second": items[0]["offered_rate_requests_per_second"],
        "requests": len(items),
        "batches": len(occupancy),
        "duration_seconds": duration,
        "throughput_requests_per_second": throughput,
        "throughput_over_reference_capacity": throughput / reference_capacity,
        "median_arrival_to_completion_seconds": statistics.median(arrival_to_completion),
        "p95_arrival_to_completion_seconds": percentile(arrival_to_completion, 0.95),
        "median_queueing_seconds": statistics.median(queueing),
        "p95_queueing_seconds": percentile(queueing, 0.95),
        "median_service_seconds": statistics.median(service),
        "p95_service_seconds": percentile(service, 0.95),
        "occupancy_distribution": dict(sorted(Counter(occupancy.values()).items())),
        "mean_batch_occupancy": statistics.fmean(occupancy.values()),
        "numerical_violations": sum(item["numerical_contract"] == "violated" for item in items),
        "mean_peak_incremental_bytes": statistics.fmean(item["peak_incremental_bytes"] for item in items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.input.open() as stream:
        for line in stream:
            item = json.loads(line)
            groups[(item["configuration"], item["offered_load"])].append(item)
    report = {
        "schema_version": 1,
        "source": str(args.input),
        "groups": [summarize(items) for _, items in sorted(groups.items())],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"groups": len(report["groups"]), "requests": sum(x["requests"] for x in report["groups"]) }))


if __name__ == "__main__":
    main()

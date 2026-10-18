"""Reduce append-only success progress snapshots into observed throughput."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


MINIMUM_PROJECTION_SECONDS = 15 * 60


def parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("history snapshot lacks a timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("history timestamp must include a timezone")
    return parsed


def load_history(path: Path) -> list[dict]:
    snapshots = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed history line {line_number}") from error
        if not isinstance(row, dict) or row.get("schema_version") not in (1, 2):
            raise ValueError(f"unsupported history line {line_number}")
        snapshots.append(row)
    if len(snapshots) < 2:
        raise ValueError("at least two progress snapshots are required")
    return snapshots


def latest_consistent_epoch(snapshots: list[dict]) -> list[dict]:
    """Select the newest contiguous history segment with one fixed campaign scope."""

    if not snapshots:
        return []
    start = len(snapshots) - 1
    reference = snapshots[-1]
    planned = reference.get("planned_paired_episodes")
    order = [item.get("model_id") for item in reference.get("paths", [])]
    next_total = reference.get("raw_paired_episodes")
    next_paths = {
        item.get("model_id"): item.get("raw_paired_episodes")
        for item in reference.get("paths", [])
    }
    for index in range(len(snapshots) - 2, -1, -1):
        row = snapshots[index]
        row_order = [item.get("model_id") for item in row.get("paths", [])]
        row_total = row.get("raw_paired_episodes")
        row_paths = {
            item.get("model_id"): item.get("raw_paired_episodes")
            for item in row.get("paths", [])
        }
        compatible = (
            row.get("planned_paired_episodes") == planned
            and row_order == order
            and isinstance(row_total, int)
            and isinstance(next_total, int)
            and row_total <= next_total
            and all(
                isinstance(row_paths.get(model_id), int)
                and isinstance(next_paths.get(model_id), int)
                and row_paths[model_id] <= next_paths[model_id]
                for model_id in order
            )
        )
        if not compatible:
            break
        start = index
        next_total = row_total
        next_paths = row_paths
    return snapshots[start:]


def summarize(snapshots: list[dict]) -> dict:
    timestamps = [parse_timestamp(row.get("generated_at_utc")) for row in snapshots]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("history timestamps are not strictly increasing")
    planned = snapshots[0].get("planned_paired_episodes")
    if (not isinstance(planned, int) or planned <= 0
            or any(row.get("planned_paired_episodes") != planned for row in snapshots)):
        raise ValueError("planned episode cardinality changed across history")

    path_orders = [[item.get("model_id") for item in row.get("paths", [])] for row in snapshots]
    if not path_orders[0] or any(order != path_orders[0] for order in path_orders[1:]):
        raise ValueError("path inventory/order changed across history")
    total_counts = [row.get("raw_paired_episodes") for row in snapshots]
    if (any(not isinstance(value, int) or value < 0 for value in total_counts)
            or any(right < left for left, right in zip(total_counts, total_counts[1:]))):
        raise ValueError("raw episode totals are invalid or decreased")

    elapsed_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    total_delta = total_counts[-1] - total_counts[0]
    rate_per_hour = total_delta * 3600 / elapsed_seconds
    remaining = planned - total_counts[-1]
    projection_ready = (
        elapsed_seconds >= MINIMUM_PROJECTION_SECONDS and rate_per_hour > 0
    )
    per_path = []
    first_paths = {row["model_id"]: row for row in snapshots[0]["paths"]}
    last_paths = {row["model_id"]: row for row in snapshots[-1]["paths"]}
    for model_id in path_orders[0]:
        first = first_paths[model_id]
        last = last_paths[model_id]
        first_count = first.get("raw_paired_episodes")
        last_count = last.get("raw_paired_episodes")
        if (not isinstance(first_count, int) or not isinstance(last_count, int)
                or last_count < first_count):
            raise ValueError(f"path episode count decreased or is invalid: {model_id}")
        delta = last_count - first_count
        path_rate = delta * 3600 / elapsed_seconds
        path_planned = last.get("planned_paired_episodes")
        per_path.append({
            "model_id": model_id,
            "live_status": last.get("live_status"),
            "episode_delta": delta,
            "episodes_per_hour": path_rate,
            "current_episodes": last_count,
            "planned_episodes": path_planned,
            "remaining_episodes": (
                path_planned - last_count if isinstance(path_planned, int) else None
            ),
        })
    return {
        "schema_version": 1,
        "evidence_tier": "observed_linear_projection_not_completion_or_commitment",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_snapshot_count": len(snapshots),
        "observation_start_utc": snapshots[0]["generated_at_utc"],
        "observation_end_utc": snapshots[-1]["generated_at_utc"],
        "elapsed_seconds": elapsed_seconds,
        "planned_paired_episodes": planned,
        "current_paired_episodes": total_counts[-1],
        "remaining_paired_episodes": remaining,
        "episode_delta": total_delta,
        "observed_episodes_per_hour": rate_per_hour,
        "minimum_projection_seconds": MINIMUM_PROJECTION_SECONDS,
        "projection_ready": projection_ready,
        "linear_projection_hours_remaining": (
            remaining / rate_per_hour if projection_ready else None
        ),
        "paths": per_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", type=Path,
        default=Path("results/cross_model/closed_loop/live_success_status_history.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/cross_model/closed_loop/live_success_throughput.json"),
    )
    args = parser.parse_args()
    report = summarize(latest_consistent_epoch(load_history(args.history)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "history_snapshot_count", "observed_episodes_per_hour",
            "linear_projection_hours_remaining",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()

from copy import deepcopy
import json

import pytest

from scripts.summarize_live_success_throughput import (
    latest_consistent_epoch,
    load_history,
    summarize,
)


def snapshot(timestamp: str, total: int, model_count: int) -> dict:
    return {
        "schema_version": 1,
        "generated_at_utc": timestamp,
        "planned_paired_episodes": 100,
        "raw_paired_episodes": total,
        "raw_paired_successes": total,
        "paths": [{
            "model_id": "model", "live_status": "paired_campaign_partial",
            "planned_paired_episodes": 100,
            "raw_paired_episodes": model_count,
            "raw_paired_successes": model_count,
            "condition_episodes": {},
        }],
    }


def test_throughput_uses_monotonic_observed_history() -> None:
    report = summarize([
        snapshot("2026-10-09T00:00:00+00:00", 20, 20),
        snapshot("2026-10-09T01:00:00+00:00", 30, 30),
    ])

    assert report["episode_delta"] == 10
    assert report["observed_episodes_per_hour"] == 10
    assert report["projection_ready"] is True
    assert report["linear_projection_hours_remaining"] == 7
    assert report["paths"][0]["remaining_episodes"] == 70


def test_throughput_rejects_decreasing_path_count() -> None:
    first = snapshot("2026-10-09T00:00:00+00:00", 20, 20)
    second = snapshot("2026-10-09T01:00:00+00:00", 21, 19)

    with pytest.raises(ValueError, match="path episode count decreased"):
        summarize([first, second])


def test_throughput_rejects_inventory_change() -> None:
    first = snapshot("2026-10-09T00:00:00+00:00", 20, 20)
    second = deepcopy(snapshot("2026-10-09T01:00:00+00:00", 21, 21))
    second["paths"][0]["model_id"] = "different"

    with pytest.raises(ValueError, match="inventory"):
        summarize([first, second])


def test_short_observation_window_withholds_eta() -> None:
    report = summarize([
        snapshot("2026-10-09T00:00:00+00:00", 20, 20),
        snapshot("2026-10-09T00:01:00+00:00", 30, 30),
    ])

    assert report["observed_episodes_per_hour"] == 600
    assert report["projection_ready"] is False
    assert report["linear_projection_hours_remaining"] is None


def test_history_loader_accepts_mixed_raw_and_validated_ledger_schemas(tmp_path) -> None:
    first = snapshot("2026-10-09T00:00:00+00:00", 20, 20)
    second = snapshot("2026-10-09T01:00:00+00:00", 30, 30)
    second["schema_version"] = 2
    second["validated_paired_episodes"] = 29
    path = tmp_path / "history.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (first, second)) + "\n")

    assert load_history(path) == [first, second]


def test_latest_epoch_excludes_an_older_campaign_denominator() -> None:
    older = snapshot("2026-10-08T23:00:00+00:00", 10, 10)
    older["planned_paired_episodes"] = 80
    current_one = snapshot("2026-10-09T00:00:00+00:00", 20, 20)
    current_two = snapshot("2026-10-09T01:00:00+00:00", 30, 30)

    assert latest_consistent_epoch([older, current_one, current_two]) == [
        current_one, current_two,
    ]

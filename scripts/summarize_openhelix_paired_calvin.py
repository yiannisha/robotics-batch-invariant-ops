"""Validate and summarize the four-condition OpenHelix CALVIN campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from run_openhelix_paired_calvin import (
    CONDITIONS,
    NUM_SEQUENCES,
    array_bundle_sha256,
    sha256,
)


CONDITION_ORDER = tuple(CONDITIONS)
THRESHOLDS = tuple(range(1, 6))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261015


def load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_gzip_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_bootstrap(differences: list[int], seed: int) -> dict[str, Any]:
    if not differences:
        raise ValueError("paired bootstrap requires at least one difference")
    generator = random.Random(seed)
    count = len(differences)
    means = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "method": "paired_percentile_bootstrap",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "difference": statistics.fmean(differences),
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
    }


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    expected = {
        "actions", "robot_states", "scene_states", "rewards", "dones",
        "subtask_indices", "policy_call_indices", "action_chunk_indices",
        "batch_sizes", "target_positions",
    }
    if set(arrays) != expected:
        raise ValueError(f"OpenHelix trajectory array keys differ: {path}")
    lengths = {value.shape[0] for value in arrays.values()}
    if (len(lengths) != 1 or arrays["actions"].shape[1:] != (7,)
            or arrays["robot_states"].shape[1:] != (15,)
            or arrays["scene_states"].shape[1:] != (24,)):
        raise ValueError(f"OpenHelix trajectory array shapes differ: {path}")
    if any(not np.isfinite(arrays[name]).all() for name in (
        "actions", "robot_states", "scene_states", "rewards"
    )):
        raise ValueError(f"OpenHelix trajectory contains nonfinite values: {path}")
    return arrays


def validate_condition(root: Path, condition: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = root / condition
    manifest_path = directory / "condition_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    repaired, dynamic = CONDITIONS[condition]
    if (manifest.get("model") != "openhelix"
            or manifest.get("condition") != condition
            or manifest.get("invariant_ops") is not repaired
            or manifest.get("execution_mode")
                != ("dynamic_batch" if dynamic else "serial_singleton")
            or manifest.get("sequence_start") != 0
            or manifest.get("sequence_count") != NUM_SEQUENCES
            or manifest.get("official_sequence_count") != NUM_SEQUENCES
            or manifest.get("policy_calls_per_subtask") != 60
            or manifest.get("execute_len") != 20
            or manifest.get("request_local_noise") is not True):
        raise ValueError(f"OpenHelix condition manifest differs: {condition}")
    rows = []
    for sequence_index in range(NUM_SEQUENCES):
        metadata_path = directory / "rollouts" / f"sequence_{sequence_index:04d}.json.gz"
        arrays_path = directory / "rollouts" / f"sequence_{sequence_index:04d}.npz"
        if not metadata_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(f"OpenHelix condition is incomplete: {condition} {sequence_index}")
        row = load_gzip_json(metadata_path)
        if (row.get("sequence_index") != sequence_index
                or row.get("initialization_id") != sequence_index
                or row.get("explicit_init_state_id") != sequence_index
                or row.get("condition") != condition
                or row.get("trajectory_npz_path") != str(arrays_path.resolve())
                or row.get("trajectory_npz_sha256") != sha256(arrays_path)
                or not isinstance(row.get("success"), bool)
                or row.get("success") != (row.get("tasks_completed") == 5)
                or row.get("sequence_length") != 5
                or not isinstance(row.get("initial_observation_sha256"), str)
                or len(row["initial_observation_sha256"]) != 64
                or not isinstance(row.get("first_noise_sha256"), str)
                or len(row["first_noise_sha256"]) != 64):
            raise ValueError(f"OpenHelix rollout metadata differs: {metadata_path}")
        arrays = load_arrays(arrays_path)
        if (row.get("environment_steps") != arrays["actions"].shape[0]
                or row.get("trajectory_arrays_sha256") != array_bundle_sha256(arrays)):
            raise ValueError(f"OpenHelix rollout trajectory differs: {metadata_path}")
        row = dict(row)
        row["source_metadata_path"] = str(metadata_path.resolve())
        row["source_metadata_sha256"] = sha256(metadata_path)
        rows.append(row)
    extras = set((directory / "rollouts").glob("sequence_*.json.gz")) - {
        directory / "rollouts" / f"sequence_{index:04d}.json.gz"
        for index in range(NUM_SEQUENCES)
    }
    if extras:
        raise ValueError(f"OpenHelix condition contains unexpected sequence files: {condition}")
    return manifest, rows


def first_divergence(left: np.ndarray, right: np.ndarray) -> int | None:
    common = min(len(left), len(right))
    if common:
        unequal = np.flatnonzero(np.any(left[:common] != right[:common], axis=1))
        if unequal.size:
            return int(unequal[0])
    return common if len(left) != len(right) else None


def summarize(root: Path, episodes_path: Path, summary_path: Path) -> dict[str, Any]:
    manifests = {}
    by_condition = {}
    combined = []
    for condition in CONDITION_ORDER:
        manifest, rows = validate_condition(root, condition)
        manifests[condition] = manifest
        by_condition[condition] = {row["sequence_index"]: row for row in rows}
        combined.extend(rows)
    expected = set(range(NUM_SEQUENCES))
    if any(set(rows) != expected for rows in by_condition.values()):
        raise ValueError("OpenHelix paired sequence coverage differs")

    reset_mismatches = []
    noise_mismatches = []
    for sequence_index in range(NUM_SEQUENCES):
        group = [by_condition[condition][sequence_index] for condition in CONDITION_ORDER]
        if len({row["initial_condition_sha256"] for row in group}) != 1:
            raise ValueError(f"OpenHelix initial-condition identity differs: {sequence_index}")
        if len({row["initial_observation_sha256"] for row in group}) != 1:
            reset_mismatches.append(sequence_index)
        if len({row["first_noise_sha256"] for row in group}) != 1:
            noise_mismatches.append(sequence_index)
    if noise_mismatches:
        raise ValueError(f"OpenHelix first request noise differs: {noise_mismatches[:10]}")

    condition_metrics = {}
    for condition in CONDITION_ORDER:
        values = [by_condition[condition][index]["tasks_completed"] for index in range(NUM_SEQUENCES)]
        condition_metrics[condition] = {
            "sequences": NUM_SEQUENCES,
            "task_threshold_success_counts": {
                str(threshold): sum(value >= threshold for value in values)
                for threshold in THRESHOLDS
            },
            "task_threshold_success_rates": {
                str(threshold): sum(value >= threshold for value in values) / NUM_SEQUENCES
                for threshold in THRESHOLDS
            },
            "average_sequence_length": statistics.fmean(values),
        }

    paired_intervals = {}
    disagreement_counts = {}
    action_divergence_counts = {}
    trajectory_divergence_counts = {}
    divergence_details = []
    for pair_index, (left, right) in enumerate(itertools.combinations(CONDITION_ORDER, 2)):
        pair = f"{left}|{right}"
        paired_intervals[pair] = {}
        disagreement_counts[pair] = {}
        action_divergence_counts[pair] = 0
        trajectory_divergence_counts[pair] = 0
        for threshold in THRESHOLDS:
            differences = [
                int(by_condition[right][index]["tasks_completed"] >= threshold)
                - int(by_condition[left][index]["tasks_completed"] >= threshold)
                for index in range(NUM_SEQUENCES)
            ]
            disagreement_counts[pair][str(threshold)] = sum(value != 0 for value in differences)
            paired_intervals[pair][str(threshold)] = paired_bootstrap(
                differences, BOOTSTRAP_SEED + pair_index * 10 + threshold
            )
        for sequence_index in range(NUM_SEQUENCES):
            left_path = Path(by_condition[left][sequence_index]["trajectory_npz_path"])
            right_path = Path(by_condition[right][sequence_index]["trajectory_npz_path"])
            left_arrays = load_arrays(left_path)
            right_arrays = load_arrays(right_path)
            action_first = first_divergence(left_arrays["actions"], right_arrays["actions"])
            robot_first = first_divergence(left_arrays["robot_states"], right_arrays["robot_states"])
            scene_first = first_divergence(left_arrays["scene_states"], right_arrays["scene_states"])
            if action_first is not None:
                action_divergence_counts[pair] += 1
            if robot_first is not None or scene_first is not None:
                trajectory_divergence_counts[pair] += 1
            if action_first is not None or robot_first is not None or scene_first is not None:
                divergence_details.append({
                    "pair": pair, "sequence_index": sequence_index,
                    "first_action_divergence_step": action_first,
                    "first_robot_state_divergence_step": robot_first,
                    "first_scene_state_divergence_step": scene_first,
                })

    combined.sort(key=lambda row: (row["sequence_index"], CONDITION_ORDER.index(row["condition"])))
    write_gzip_jsonl_atomic(episodes_path, combined)
    summary = {
        "schema_version": 1,
        "evidence_tier": "full_paired_closed_loop",
        "model_id": "openhelix",
        "benchmark": "CALVIN ABC-D official 1000 five-task sequences",
        "sequences_per_condition": NUM_SEQUENCES,
        "paired_condition_sequence_records": len(combined),
        "episodes_path": str(episodes_path.resolve()),
        "episodes_sha256": sha256(episodes_path),
        "condition_manifests": {
            condition: {
                "path": str((root / condition / "condition_manifest.json").resolve()),
                "sha256": sha256(root / condition / "condition_manifest.json"),
            }
            for condition in CONDITION_ORDER
        },
        "condition_metrics": condition_metrics,
        "paired_confidence_intervals": paired_intervals,
        "disagreement_counts": disagreement_counts,
        "action_divergence_counts": action_divergence_counts,
        "trajectory_divergence_counts": trajectory_divergence_counts,
        "divergence_details": divergence_details,
        "cross_condition_initial_observation_mismatch_count": len(reset_mismatches),
        "cross_condition_initial_observation_mismatch_sequences": reset_mismatches,
        "first_request_noise_mismatch_count": 0,
    }
    write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.root, args.episodes, args.output)
    print(json.dumps({
        "paired_records": summary["paired_condition_sequence_records"],
        "condition_metrics": summary["condition_metrics"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

"""Combine four complete, request-local LeRobot LIBERO conditions.

This reducer never turns a pilot into full evidence: by default it requires
50 matched reset seeds on each of the 10 checkpoint-matched tasks, four
conditions, matching physical-observation hashes within each batching
comparison, matching request noise, and agreement with each condition's
official LeRobot success metrics.  Cross-implementation comparisons exclude
and report any reset whose physical observation differs instead of treating it
as causally paired.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path


CONDITIONS = (
    "native_singleton", "native_dynamic",
    "patched_singleton", "patched_dynamic",
)
PHYSICAL_BATCH = {"native_singleton": 1, "native_dynamic": 4, "patched_singleton": 1, "patched_dynamic": 4}
INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256 = (
    "329cf78b3367555a0440616959ee8515e9490116530792b55db0ea3d1ef50bdf"
)
INTERNVLA_CURRENT_SERVER_SHA256 = (
    "7f125a360bd3a249369147bae7865cc110a75d61ad1abc50c95b6e0246152980"
)
INTERNVLA_LABEL_PADDING_PATCH = b'''        # The official eval transform emits one label per prompt token and
        # fills the entire tensor with the training ignore index.  Prompt
        # lengths can differ across genuinely batched requests, so labels
        # must follow the same right-padding rule as the prompt tensors.
        "VQA.labels": -100,
'''


def rollout_paths(directory: Path) -> list[Path]:
    return sorted((*directory.glob("*.json"), *directory.glob("*.json.gz")))


def read_rollout(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_bootstrap(differences: list[int], seed: int, resamples: int = 10_000) -> dict:
    generator = random.Random(seed)
    count = len(differences)
    means = [
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    return {
        "difference": statistics.fmean(differences),
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
        "method": "paired_percentile_bootstrap", "resamples": resamples, "seed": seed,
    }


def first_divergent_step(left: list, right: list) -> int | None:
    for index, (one, other) in enumerate(zip(left, right)):
        if one != other:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def validate_official_successes(
    task_metric: dict, raw_successes: list[bool], *, condition: str, task: int,
) -> str:
    """Cross-check raw outcomes against either supported official schema.

    LeRobot evaluators preserve a per-episode ``metrics.successes`` vector.
    UVA's pinned official evaluator instead preserves only ``n_episodes`` and
    aggregate ``n_success``.  The aggregate-only form is acceptable only when
    its exact episode cardinality and count agree with all preserved raw
    outcomes; it never substitutes for missing raw episodes.
    """
    if any(type(value) is not bool for value in raw_successes):
        raise ValueError(f"raw successes are not boolean for {condition} task {task}")
    n_success = task_metric.get("n_success")
    if type(n_success) is not int:
        raise ValueError(f"official n_success is invalid for {condition} task {task}")
    metrics = task_metric.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"official metrics are invalid for {condition} task {task}")
    if "successes" in metrics:
        successes = metrics["successes"]
        if (not isinstance(successes, list)
                or len(successes) < len(raw_successes)
                or any(type(value) is not bool for value in successes)
                or n_success != sum(successes)
                or raw_successes != successes[:len(raw_successes)]):
            raise ValueError(
                f"raw and official successes disagree for {condition} task {task}"
            )
        return "per_episode"
    if (task_metric.get("n_episodes") != len(raw_successes)
            or n_success != sum(raw_successes)):
        raise ValueError(
            f"raw and aggregate official successes disagree for {condition} task {task}"
        )
    return "aggregate_only"


def write_if_absent_or_identical(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"existing paired output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_bytes_if_absent_or_identical(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"existing paired output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def validate_stable_manifests(manifests: dict[str, dict]) -> dict:
    """Fail closed on provenance drift, including one proven B=1-only exception."""
    stable_fields = (
        "model", "checkpoint", "checkpoint_weight", "checkpoint_sha256",
        "official_evaluator_sha256", "official_policy_sha256", "paired_wrapper_sha256",
        "tokenizer_provenance_sha256", "deterministic_decoding_sha256", "seed",
    )
    first_manifest = manifests[CONDITIONS[0]]
    if any(
        any(manifest.get(field) != first_manifest.get(field) for field in stable_fields)
        for manifest in manifests.values()
    ):
        raise ValueError("paired conditions used different checkpoint, source, wrapper, or seed")

    wrapper_hashes = {
        condition: manifest.get("shared_rollout_wrapper_sha256")
        for condition, manifest in manifests.items()
    }
    if len(set(wrapper_hashes.values())) == 1:
        return {
            "status": "identical_across_conditions",
            "shared_rollout_wrapper_sha256": next(iter(wrapper_hashes.values())),
        }

    # InternVLA standard's native B=1 run preceded the VQA-label collation
    # fix.  Reconstructing that exact legacy source from the current source by
    # removing the five-line patch must reproduce its pinned hash.  The patch
    # can only affect stacking examples of unequal prompt lengths; with one
    # physical example torch.stack receives one unchanged labels tensor.
    expected = {
        "native_singleton": INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256,
        "native_dynamic": INTERNVLA_CURRENT_SERVER_SHA256,
        "patched_singleton": INTERNVLA_CURRENT_SERVER_SHA256,
        "patched_dynamic": INTERNVLA_CURRENT_SERVER_SHA256,
    }
    if first_manifest.get("model") != "internvla_a15_standard" or wrapper_hashes != expected:
        raise ValueError("paired conditions used different checkpoint, source, wrapper, or seed")
    current_source = Path(__file__).with_name("run_internvla_paired_server.py").read_bytes()
    if hashlib.sha256(current_source).hexdigest() != INTERNVLA_CURRENT_SERVER_SHA256:
        raise ValueError("current InternVLA paired server differs from the pinned compatibility proof")
    if current_source.count(INTERNVLA_LABEL_PADDING_PATCH) != 1:
        raise ValueError("cannot reconstruct the pinned legacy InternVLA paired server")
    legacy_source = current_source.replace(INTERNVLA_LABEL_PADDING_PATCH, b"", 1)
    if hashlib.sha256(legacy_source).hexdigest() != INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256:
        raise ValueError("reconstructed legacy InternVLA paired server hash differs")
    return {
        "status": "proven_b1_noop_source_patch",
        "legacy_condition": "native_singleton",
        "legacy_shared_rollout_wrapper_sha256": INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256,
        "current_shared_rollout_wrapper_sha256": INTERNVLA_CURRENT_SERVER_SHA256,
        "exact_source_delta": "add VQA.labels to right-padding keys",
        "physical_batch_size": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initializations-per-task", type=int, default=50)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--gzip-episodes", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.initializations_per_task <= 50:
        raise ValueError("paired LIBERO aggregation requires 1..50 initializations")
    if args.task_count < 1:
        raise ValueError("task_count must be positive")
    if args.initializations_per_task < 50 and not any("pilot" in part for part in args.root.parts):
        raise ValueError("reduced paired aggregation must live in a pilot directory")

    manifests = {}
    metrics = {}
    episodes = {}
    official_success_granularity = {}
    metric_sources = {}
    manifest_sources = {}
    for condition in CONDITIONS:
        directory = args.root / condition
        manifest_path = directory / "condition_manifest.json"
        info_path = directory / "official" / "eval_info.json"
        if not manifest_path.is_file() or not info_path.is_file():
            raise FileNotFoundError(f"missing official paired condition {condition}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("condition") != condition
                or manifest.get("batch_size") != PHYSICAL_BATCH[condition]
                or manifest.get("invariant_ops") != condition.startswith("patched")
                or manifest.get("explicit_init_state_by_initialization_id") is not True
                or manifest.get("episodes_per_task", 0) < args.initializations_per_task):
            raise ValueError(f"condition provenance/shape differs for {condition}")
        manifests[condition] = manifest
        manifest_sources[condition] = {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}
        info = json.loads(info_path.read_text())
        task_metrics = {row["task_id"]: row for row in info.get("per_task", [])}
        if set(task_metrics) != set(range(args.task_count)):
            raise ValueError(
                f"official evaluator did not finish all {args.task_count} tasks for {condition}"
            )
        metrics[condition] = task_metrics
        metric_sources[condition] = {"path": str(info_path.resolve()), "sha256": sha256(info_path)}
        rows = {}
        for path in rollout_paths(directory / "rollouts"):
            payload = read_rollout(path)
            expected_init_ids = [int(value) - manifest["seed"] for value in payload.get("seeds", [])]
            if payload.get("explicit_init_state_ids") != expected_init_ids:
                raise ValueError(f"rollout lacks explicit requested init-state IDs in {path}")
            init_by_seed = dict(zip(payload.get("seeds", []), expected_init_ids))
            for episode in payload.get("episodes", []):
                task = episode.get("task")
                initialization = episode.get("initialization_id")
                if task not in range(args.task_count) or not isinstance(initialization, int):
                    raise ValueError(f"invalid paired task/reset record in {path}")
                if initialization >= args.initializations_per_task:
                    continue  # LeRobot can run extra physical slots in a final B4 batch.
                key = (task, initialization)
                if key in rows:
                    raise ValueError(f"duplicate paired episode {key} for {condition}")
                episode["condition"] = condition
                episode["source_rollout_path"] = str(path.resolve())
                episode["explicit_init_state_id"] = init_by_seed.get(episode.get("reset_seed"))
                if episode["explicit_init_state_id"] != initialization:
                    raise ValueError(f"episode init-state assignment differs from request in {path}")
                rows[key] = episode
        wanted = {
            (task, initialization)
            for task in range(args.task_count)
            for initialization in range(args.initializations_per_task)
        }
        if set(rows) != wanted:
            raise ValueError(f"{condition} has {len(rows)}/{len(wanted)} paired LIBERO episodes")
        official_success_granularity[condition] = {}
        for task in range(args.task_count):
            raw_successes = [
                rows[task, index]["success"]
                for index in range(args.initializations_per_task)
            ]
            official_success_granularity[condition][str(task)] = (
                validate_official_successes(
                    task_metrics[task], raw_successes, condition=condition, task=task,
                )
            )
        episodes[condition] = rows

    first_manifest = manifests[CONDITIONS[0]]
    wrapper_compatibility = validate_stable_manifests(manifests)
    pairs = {
        (task, initialization)
        for task in range(args.task_count)
        for initialization in range(args.initializations_per_task)
    }
    cross_implementation_reset_mismatches = []
    for key in pairs:
        rows = [episodes[condition][key] for condition in CONDITIONS]
        sensors = {
            condition: episodes[condition][key]["initial_observation_sha256"]
            for condition in CONDITIONS
        }
        if (sensors["native_singleton"] != sensors["native_dynamic"]
                or sensors["patched_singleton"] != sensors["patched_dynamic"]):
            raise ValueError(f"within-implementation paired physical reset differs at {key}")
        if len(set(sensors.values())) != 1:
            cross_implementation_reset_mismatches.append({
                "task": key[0], "initialization_id": key[1], "hashes": sensors,
            })
        if len({row["first_noise_sha256"] for row in rows}) != 1:
            raise ValueError(f"paired request-local first noise differs at {key}")
        if any(row["reset_seed"] != first_manifest["seed"] + key[1] for row in rows):
            raise ValueError(f"paired reset seed differs at {key}")

    comparisons = list(itertools.combinations(CONDITIONS, 2))
    success_counts = {
        condition: sum(episodes[condition][key]["success"] for key in pairs)
        for condition in CONDITIONS
    }
    disagreement_counts = {}
    action_divergence_counts = {}
    paired_confidence_intervals = {}
    paired_sample_counts = {}
    excluded_initial_observation_mismatch_counts = {}
    for pair_index, (left, right) in enumerate(comparisons):
        name = f"{left}|{right}"
        ordered = sorted(
            key for key in pairs
            if (episodes[left][key]["initial_observation_sha256"]
                == episodes[right][key]["initial_observation_sha256"])
        )
        if not ordered:
            raise ValueError(f"no physically paired resets remain for comparison {name}")
        paired_sample_counts[name] = len(ordered)
        excluded_initial_observation_mismatch_counts[name] = len(pairs) - len(ordered)
        differences = [int(episodes[right][key]["success"]) - int(episodes[left][key]["success"]) for key in ordered]
        disagreement_counts[name] = sum(value != 0 for value in differences)
        action_divergence_counts[name] = sum(
            first_divergent_step(episodes[left][key]["action_sequence"], episodes[right][key]["action_sequence"]) is not None
            for key in ordered
        )
        paired_confidence_intervals[name] = paired_bootstrap(differences, 20261005 + pair_index)
    per_task = []
    for task in range(args.task_count):
        task_keys = [(task, index) for index in range(args.initializations_per_task)]
        per_task.append({
            "task": task,
            "initializations": args.initializations_per_task,
            "success_counts": {
                condition: sum(episodes[condition][key]["success"] for key in task_keys)
                for condition in CONDITIONS
            },
            "success_ci95": {
                condition: metrics[condition][task].get("pc_success_ci95")
                for condition in CONDITIONS
            },
            "disagreement_counts": {
                f"{left}|{right}": sum(
                    episodes[left][key]["success"] != episodes[right][key]["success"]
                    for key in task_keys
                    if (episodes[left][key]["initial_observation_sha256"]
                        == episodes[right][key]["initial_observation_sha256"])
                )
                for left, right in comparisons
            },
            "paired_sample_counts": {
                f"{left}|{right}": sum(
                    episodes[left][key]["initial_observation_sha256"]
                    == episodes[right][key]["initial_observation_sha256"]
                    for key in task_keys
                )
                for left, right in comparisons
            },
            "action_divergence_counts": {
                f"{left}|{right}": sum(
                    first_divergent_step(episodes[left][key]["action_sequence"], episodes[right][key]["action_sequence"]) is not None
                    for key in task_keys
                    if (episodes[left][key]["initial_observation_sha256"]
                        == episodes[right][key]["initial_observation_sha256"])
                )
                for left, right in comparisons
            },
        })
    combined = [
        episodes[condition][key]
        for key in sorted(pairs) for condition in CONDITIONS
    ]
    raw_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined)
    raw_path = args.root / ("episodes.jsonl.gz" if args.gzip_episodes else "episodes.jsonl")
    if args.gzip_episodes:
        write_bytes_if_absent_or_identical(
            raw_path, gzip.compress(raw_text.encode(), compresslevel=6, mtime=0)
        )
    else:
        write_if_absent_or_identical(raw_path, raw_text)
    summary = {
        "schema_version": 1,
        "evidence_tier": "full" if args.initializations_per_task == 50 else "pilot",
        "model_id": first_manifest["model"],
        "tasks": args.task_count, "initializations_per_task": args.initializations_per_task,
        "paired_start_seed": first_manifest["seed"],
        "paired_episodes": len(combined),
        "episodes_path": str(raw_path.resolve()), "episodes_sha256": sha256(raw_path),
        "condition_eval_info": metric_sources,
        "official_success_granularity": official_success_granularity,
        "condition_manifests": manifest_sources,
        "shared_rollout_wrapper_compatibility": wrapper_compatibility,
        "success_counts": success_counts,
        "disagreement_counts": disagreement_counts,
        "action_divergence_counts": action_divergence_counts,
        "paired_sample_counts": paired_sample_counts,
        "excluded_initial_observation_mismatch_counts": (
            excluded_initial_observation_mismatch_counts
        ),
        "cross_implementation_initial_observation_mismatches": (
            cross_implementation_reset_mismatches
        ),
        "paired_confidence_intervals": paired_confidence_intervals,
        "per_task": per_task,
    }
    summary_path = args.root / "paired_summary.json"
    write_if_absent_or_identical(summary_path, json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "evidence_tier": summary["evidence_tier"], "episodes": len(combined),
        "tasks": args.task_count,
    }))


if __name__ == "__main__":
    main()

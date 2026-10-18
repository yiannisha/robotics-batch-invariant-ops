"""Independent fail-closed audit for CogACT-Small SimplerEnv evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
ROOT = Path("/workspace/robotics-batch-invariant-ops")
TASK_MANIFEST = (
    ROOT
    / "results/cross_model/cogact_small/full/closed_loop/paired/"
    "evaluator_task_manifest.json"
)
TASK_MANIFEST_SHA256 = "37c70a256a17c0e11a2343140d550549e9b74ff4896059ae14f863b96f439eba"
CHECKPOINT_SHA256 = "013cf0d4954e8a560064905ee03785ce3d0fa126697d4410cfde9a6754665e44"
CHECKPOINT_PATH = "/dev/shm/cogact/checkpoints/CogACT-Small.pt"
BASE_SEED = 20261005
MAX_STEPS = (120, 120, 120, 120, 113, 113, 113, 113, 113, 113, 80, 80)
# Pose placement is an environment contract, not a policy-family shortcut:
# the Google drawer environments expose agent.base_pose, while the Google
# MoveNear and coke-can environments expose extra.tcp_pose like the WidowX
# environments do.  Keep the mapping explicit so the independent audit does
# not infer it from producer output.
EXPECTED_POSE_KEYS = (
    "tcp_pose", "tcp_pose", "tcp_pose", "tcp_pose",
    "base_pose", "base_pose", "base_pose", "base_pose", "base_pose", "base_pose",
    "tcp_pose", "tcp_pose",
)

# The one-episode probe predates the copy-owning msgpack noise fix.  Its exact
# source hashes remain pinned as probe provenance.  Every 496-episode campaign
# condition must use the current reviewed wrappers.
SOURCE_HASHES = {
    "probe": {
        "client_source_sha256": "055189e588f154c75a08e48a6309f4dcd6e351f62987dfefa9585e7c2411061f",
        "server_source_sha256": "610a9366fbba81c4c485a83e409d36e05762315b52a908c5b14ffb7751ceb1a9",
    },
    "campaign": {
        "client_source_sha256": "afa350668d8d77b6edc9ad048706369e56a5fa4a5163abf1efa31142f38f3ba6",
        "server_source_sha256": "ce2a0a8c7ac0795f91a5544b5fe52e180931ddbcd6bb394f9c1894da09cfc531",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def finite_vector(value: object, length: int | None = None) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and (length is None or len(value) == length)
        and all(
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(number)
            for number in value
        )
    )


def finite_matrix(value: object, rows: int, columns: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(finite_vector(row, columns) for row in value)
    )


def valid_state(value: object, expected_pose_key: str) -> bool:
    return (
        isinstance(value, dict)
        and expected_pose_key in {"tcp_pose", "base_pose"}
        and set(value) == {"qpos", "qvel", expected_pose_key}
        and finite_vector(value["qpos"])
        and finite_vector(value["qvel"])
        and len(value["qpos"]) == len(value["qvel"])
        and finite_vector(value[expected_pose_key])
    )


def expected_noise_hash(task: int, initialization: int, step: int) -> str:
    sequence = np.random.SeedSequence(
        [BASE_SEED, task, initialization, step, 0xC06AC7]
    )
    noise = np.random.Generator(np.random.PCG64(sequence)).standard_normal(
        (16, 7), dtype=np.float32
    )
    return hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()


def successes_in_initialization_order(values: dict[int, bool]) -> list[bool]:
    """Return outcomes in numeric initialization order, independent of filenames."""
    return [values[initialization] for initialization in sorted(values)]


def validate_task_manifest() -> list[dict]:
    if not TASK_MANIFEST.is_file() or sha256(TASK_MANIFEST) != TASK_MANIFEST_SHA256:
        raise ValueError("CogACT pinned task manifest is missing or changed")
    manifest = json.loads(TASK_MANIFEST.read_text())
    tasks = manifest.get("tasks")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model_id") != "cogact_small"
        or manifest.get("task_suite") != "simplerenv_cogact_semantic_12_variant"
        or manifest.get("cogact_repository_revision")
        != "b174a1b86deedfab4d198d935207e7bb0527994e"
        or manifest.get("simplerenv_repository_revision")
        != "ccfe3809766839a2fcfb7a3d3c9abff585189188"
        or manifest.get("base_seed") != BASE_SEED
        or manifest.get("conditions") != list(CONDITIONS)
        or manifest.get("task_count") != 12
        or manifest.get("initializations_per_condition") != 496
        or manifest.get("logical_episodes_all_conditions") != 1984
        or not isinstance(tasks, list)
        or len(tasks) != 12
        or [task.get("task_id") for task in tasks] != list(range(12))
        or [task.get("initializations") for task in tasks]
        != [24, 24, 24, 24, 50, 50, 50, 50, 50, 50, 50, 50]
        or [task.get("policy_setup") for task in tasks]
        != ["widowx_bridge"] * 4 + ["google_robot"] * 8
    ):
        raise ValueError("CogACT evaluator task manifest differs from the pinned suite")
    return tasks


def validate_server_metadata(root: Path, condition: str, batch_size: int, repaired: bool) -> None:
    metadata_path = root / "server_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "protocol_version": "2.1",
        "policy_type": "cogact_small",
        "checkpoint": "/workspace/assets/cogact_small",
        "checkpoint_weight_sha256": CHECKPOINT_SHA256,
        "task_suite": "simplerenv_cogact_semantic_12_variant",
        "paired_condition": condition,
        "genuine_batched_policy_call": True,
        "maximum_request_batch": batch_size,
        "batch_invariant_ops": repaired,
        "request_local_noise": True,
        "noise_shape": [16, 7],
        "cfg_scale": 1.5,
        "ddim_steps": 10,
        "unnorm_keys": {
            "google_robot": "fractal20220817_data",
            "widowx_bridge": "bridge_orig",
        },
        "action_chunk_size": 16,
        "action_dim": 7,
        "adaptive_ensemble_alpha": 0.1,
        "llama_construction_dtype": "torch.bfloat16",
        "llama_random_weight_initialization": "skipped_before_strict_checkpoint_load",
        "final_vlm_dtype": "torch.bfloat16",
        "final_action_head_dtype": "torch.float32",
    }
    if metadata != expected:
        raise ValueError("CogACT live server metadata differs from the requested condition")


def audit(root: Path) -> dict:
    tasks = validate_task_manifest()
    manifest_path = root / "condition_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("unknown CogACT condition")
    batch_size, repaired = CONDITIONS[condition]
    task_count = manifest.get("task_count")
    planned = manifest.get("planned_initializations")
    if (task_count, planned) == (1, 1) and condition == "native_singleton":
        evidence_kind = "probe"
    elif (task_count, planned) == (12, 496):
        evidence_kind = "campaign"
    else:
        raise ValueError("CogACT cardinality is neither the pinned probe nor full campaign")
    required = {
        "schema_version": 1,
        "model": "cogact_small",
        "condition": condition,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "seed": BASE_SEED,
        "task_suite": "simplerenv_cogact_semantic_12_variant",
        "task_manifest_path": str(TASK_MANIFEST),
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "task_count": task_count,
        "planned_initializations": planned,
        "request_local_noise": True,
        "noise_shape": [16, 7],
        "noise_generator": (
            "numpy.PCG64.SeedSequence([seed,task,initialization,step,0xC06AC7])"
        ),
        "explicit_physical_configuration": True,
        "websocket_ping_timeout_s": 600,
        "checkpoint_weight_path": CHECKPOINT_PATH,
        "checkpoint_weight_sha256": CHECKPOINT_SHA256,
        **SOURCE_HASHES[evidence_kind],
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("CogACT manifest differs from the pinned evaluator contract")
    metadata_path = root / "server_metadata.json"
    if manifest.get("server_metadata_path") != str(metadata_path.resolve()):
        raise ValueError("CogACT condition manifest points to another server record")
    if manifest.get("server_metadata_sha256") != sha256(metadata_path):
        raise ValueError("CogACT live server record changed after condition freeze")
    validate_server_metadata(root, condition, batch_size, repaired)

    expected_counts = {task["task_id"]: task["initializations"] for task in tasks[:task_count]}
    counts: Counter[int] = Counter()
    successes: Counter[int] = Counter()
    observed_successes: dict[int, dict[int, bool]] = {
        task: {} for task in range(task_count)
    }
    seen: set[tuple[int, int]] = set()
    files = sorted((root / "rollouts").glob("*.json"))
    files += sorted((root / "rollouts").glob("*.json.gz"))
    for path in files:
        payload = read_json(path)
        task = payload.get("task_id")
        ids = payload.get("explicit_initialization_ids")
        rows = payload.get("episodes")
        if (
            task not in range(task_count)
            or payload.get("task_name") != tasks[task]["name"]
            or not isinstance(ids, list)
            or not ids
            or ids != list(range(ids[0], ids[0] + len(ids)))
            or ids[0] % batch_size != 0
            or len(ids) != min(batch_size, expected_counts[task] - ids[0])
            or any(index not in range(expected_counts[task]) for index in ids)
            or payload.get("seeds") != [BASE_SEED + index for index in ids]
            or not isinstance(rows, list)
            or len(rows) != len(ids)
            or (path.name[:-3] if path.name.endswith(".gz") else path.name)
            != f"task_{task}_initializations_{ids[0]}_{ids[-1]}.json"
        ):
            raise ValueError(f"invalid CogACT physical rollout group: {path}")
        for row, initialization in zip(rows, ids, strict=True):
            key = (task, initialization)
            if key in seen:
                raise ValueError(f"duplicate CogACT task/configuration: {key}")
            seen.add(key)
            states = row.get("trajectory")
            actions = row.get("action_sequence")
            selected = row.get("raw_action_sequence")
            chunks = row.get("raw_action_chunks")
            noise_hashes = row.get("noise_sha256")
            rewards = row.get("rewards")
            done = row.get("done")
            truncated = row.get("truncated")
            physical_batches = row.get("policy_batch_sizes")
            success = row.get("success")
            steps = len(actions) if isinstance(actions, list) else -1
            expected_noise = [
                expected_noise_hash(task, initialization, step) for step in range(steps)
            ]
            if (
                row.get("task") != task
                or row.get("task_name") != tasks[task]["name"]
                or row.get("task_group") != "simplerenv_cogact_semantic_12_variant"
                or not isinstance(row.get("task_description"), str)
                or not row["task_description"]
                or row.get("policy_setup") != tasks[task]["policy_setup"]
                or row.get("initialization_id") != initialization
                or row.get("reset_seed") != BASE_SEED + initialization
                or not valid_hash(row.get("initial_observation_sha256"))
                or not isinstance(success, bool)
                or not isinstance(states, list)
                or len(states) != steps + 1
                or steps <= 0
                or steps > MAX_STEPS[task]
                or any(
                    not valid_state(state, EXPECTED_POSE_KEYS[task])
                    for state in states
                )
                or any(not finite_vector(action, 7) for action in actions)
                or not isinstance(selected, list)
                or len(selected) != steps
                or any(not finite_vector(action, 7) for action in selected)
                or not isinstance(chunks, list)
                or len(chunks) != steps
                or any(not finite_matrix(chunk, 16, 7) for chunk in chunks)
                or noise_hashes != expected_noise
                or len(set(noise_hashes)) != steps
                or not isinstance(rewards, list)
                or len(rewards) != steps
                or any(
                    not isinstance(reward, (int, float))
                    or isinstance(reward, bool)
                    or not math.isfinite(reward)
                    for reward in rewards
                )
                or not isinstance(done, list)
                or len(done) != steps
                or any(not isinstance(value, bool) for value in done)
                or not isinstance(truncated, list)
                or len(truncated) != steps
                or any(not isinstance(value, bool) for value in truncated)
                or not truncated[-1]
                or success != any(done)
                or not isinstance(physical_batches, list)
                or len(physical_batches) != steps
                or physical_batches[0] != len(ids)
                or any(
                    not isinstance(size, int) or not 1 <= size <= len(ids)
                    for size in physical_batches
                )
            ):
                raise ValueError(f"incomplete or inconsistent CogACT episode: {path}")
            counts[task] += 1
            successes[task] += int(success)
            observed_successes[task][initialization] = success

    for task in range(task_count):
        observed = {
            initialization
            for observed_task, initialization in seen
            if observed_task == task
        }
        if observed != set(range(len(observed))):
            raise ValueError(f"noncontiguous CogACT configuration prefix for task {task}")
        if counts[task] and any(
            counts[previous] != expected_counts[previous] for previous in range(task)
        ):
            raise ValueError(f"CogACT task {task} started before its predecessor completed")

    info_path = root / "official/eval_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        per_task = info.get("per_task")
        overall = info.get("overall", {})
        if (
            info.get("schema_version") != 1
            or info.get("condition") != condition
            or info.get("task_suite") != "simplerenv_cogact_semantic_12_variant"
            or not isinstance(per_task, list)
            or len(per_task) != task_count
            or [item.get("task_id") for item in per_task] != list(range(task_count))
            or any(
                item.get("task_name") != tasks[index]["name"]
                or item.get("metrics", {}).get("successes")
                != successes_in_initialization_order(observed_successes[index])
                or item.get("n_episodes") != counts[index]
                or item.get("n_success") != successes[index]
                for index, item in enumerate(per_task)
            )
            or overall.get("n_episodes") != sum(counts.values())
            or overall.get("n_success") != sum(successes.values())
        ):
            raise ValueError("CogACT raw episodes and aggregate metrics differ")
    elif sum(counts.values()) == planned:
        raise ValueError("complete CogACT evidence lacks official aggregate metrics")

    return {
        "schema_version": 1,
        "evidence_tier": (
            "checkpoint_matched_probe"
            if evidence_kind == "probe"
            else "running_prefix_only_not_paired_completion"
        ),
        "model_id": "cogact_small",
        "condition": condition,
        "task_count": task_count,
        "episodes_cached": sum(counts.values()),
        "successes_cached": sum(successes.values()),
        "planned_episodes": planned,
        "complete": sum(counts.values()) == planned,
        "integrity_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.condition_root), sort_keys=True))


if __name__ == "__main__":
    main()

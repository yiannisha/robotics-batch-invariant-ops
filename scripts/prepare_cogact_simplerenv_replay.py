"""Reconstruct 1,000 exact CogACT observations from paired SimplerEnv rollouts.

The retained native-singleton campaign contains the reset configuration,
executed action sequence, and physical trajectory for 496 distinct episodes.
This script deterministically selects 100 task-balanced episodes, replays each
one in the pinned evaluator, and captures ten genuine policy-request images.
Three request-local DDIM noises are stored per observation; noise zero exactly
matches the success-rate evaluator contract.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_cogact_paired_client import (
    make_environment,
    physical_hash,
    request_noise as evaluator_noise,
    state_record,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


BASE_SEED = 20261005
NOISE_EXTENSION_DOMAIN = 0xC06AC7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def rollout_path(root: Path, task: int, initialization: int) -> Path:
    prefix = f"task_{task}_initializations_{initialization}_{initialization}.json"
    found = [path for path in (root / prefix, root / f"{prefix}.gz") if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"exactly one CogACT rollout required for {task}/{initialization}")
    return found[0]


def selected_episodes(tasks: list[dict]) -> list[tuple[dict, int]]:
    rows = []
    for task in tasks:
        # 9 from each of the four 24-initialization Bridge tasks and 8 from
        # each of the eight 50-initialization Google tasks: exactly 100.
        count = 9 if task["task_id"] < 4 else 8
        if task["initializations"] < count:
            raise ValueError(f"task {task['task_id']} lacks {count} source episodes")
        rows.extend((task, initialization) for initialization in range(count))
    if len(rows) != 100:
        raise RuntimeError("CogACT task-balanced episode selection is not exactly 100")
    return rows


def selected_steps(length: int) -> list[int]:
    if length < 10:
        raise ValueError("CogACT source episode has fewer than ten policy requests")
    values = np.linspace(0, length - 1, 10, dtype=np.int64).tolist()
    if len(set(values)) != 10:
        raise ValueError("CogACT selected request boundaries are not distinct")
    return values


def equal_state(observed: dict, expected: dict) -> bool:
    return set(observed) == set(expected) and all(
        np.array_equal(np.asarray(observed[key]), np.asarray(expected[key]))
        for key in observed
    )


def save_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(temporary, format="PNG")
    os.replace(temporary, path)
    return sha256(path)


def validate_bundle(bundle: dict, root: Path, task: int, initialization: int) -> bool:
    rows = bundle.get("observations")
    if (bundle.get("schema_version") != 1 or bundle.get("task") != task
            or bundle.get("initialization") != initialization
            or not isinstance(rows, list) or len(rows) != 10):
        return False
    return all(
        (root / row["images"]["observation"]).is_file()
        and sha256(root / row["images"]["observation"])
        == row["image_sha256"]["observation"]
        for row in rows
    )


def reconstruct_episode(
    *, task: dict, initialization: int, rollout_path_value: Path, output_root: Path,
) -> dict:
    payload = read_json(rollout_path_value)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError(f"CogACT singleton rollout is malformed: {rollout_path_value}")
    rollout = episodes[0]
    actions, trajectory = rollout.get("action_sequence"), rollout.get("trajectory")
    if (rollout.get("task") != task["task_id"]
            or rollout.get("task_name") != task["name"]
            or rollout.get("initialization_id") != initialization
            or rollout.get("reset_seed") != BASE_SEED + initialization
            or not isinstance(actions, list) or not isinstance(trajectory, list)
            or len(trajectory) != len(actions) + 1):
        raise ValueError(f"CogACT rollout provenance differs for {task['task_id']}/{initialization}")
    wanted = selected_steps(len(actions))
    wanted_set = set(wanted)
    env, obs, config = make_environment(task, initialization, BASE_SEED + initialization)
    captures = []
    try:
        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=config["obs_camera_name"]
        )
        if physical_hash(obs, image) != rollout.get("initial_observation_sha256"):
            raise ValueError(f"CogACT reset hash differs for {task['task_id']}/{initialization}")
        for step, action in enumerate(actions):
            observed_state = state_record(obs)
            if not equal_state(observed_state, trajectory[step]):
                raise ValueError(
                    f"CogACT replay state differs at {task['task_id']}/{initialization}/{step}"
                )
            if step in wanted_set:
                relative = (
                    f"frames/task_{task['task_id']:02d}_init_{initialization:02d}_"
                    f"step_{step:03d}.png"
                )
                image_hash = save_png(output_root / relative, image)
                captures.append({
                    "task_index": task["task_id"], "task": task["name"],
                    "initialization": initialization,
                    "reset_seed": BASE_SEED + initialization,
                    "step": step,
                    "instruction": env.get_language_instruction(),
                    "policy_setup": task["policy_setup"],
                    "state": observed_state,
                    "images": {"observation": relative},
                    "image_sha256": {"observation": image_hash},
                    "physical_observation_sha256": physical_hash(obs, image),
                })
            obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
            image = get_image_from_maniskill2_obs_dict(
                env, obs, camera_name=config["obs_camera_name"]
            )
        if not equal_state(state_record(obs), trajectory[-1]):
            raise ValueError(f"CogACT final replay state differs for {task['task_id']}/{initialization}")
    finally:
        env.close()
    if [row["step"] for row in captures] != wanted:
        raise RuntimeError("CogACT replay omitted a selected request boundary")
    return {
        "schema_version": 1,
        "task": task["task_id"], "task_name": task["name"],
        "initialization": initialization,
        "rollout_source": str(rollout_path_value.resolve()),
        "rollout_source_sha256": sha256(rollout_path_value),
        "observations": captures,
    }


def request_noise(task: int, initialization: int, step: int, noise_id: int) -> np.ndarray:
    if noise_id == 0:
        return evaluator_noise(BASE_SEED, task, initialization, step)
    sequence = np.random.SeedSequence([
        BASE_SEED, task, initialization, step, NOISE_EXTENSION_DOMAIN, noise_id,
    ])
    return np.random.Generator(np.random.PCG64(sequence)).standard_normal(
        (16, 7), dtype=np.float32,
    )


def companion_indices(rows: list[dict], target: dict, count: int) -> list[int]:
    heads = {}
    for row in rows:
        heads.setdefault((row["task_index"], row["episode_id"]), row)
    other = [
        row for key, row in heads.items()
        if key != (target["task_index"], target["episode_id"])
        and row["task_index"] != target["task_index"]
    ]
    same = [
        row for key, row in heads.items()
        if key != (target["task_index"], target["episode_id"])
        and row["task_index"] == target["task_index"]
    ]
    other.sort(key=lambda row: (row["task_index"], row["episode_id"]))
    same.sort(key=lambda row: row["episode_id"])
    distinct_tasks, selected = set(), []
    for row in other:
        if row["task_index"] not in distinct_tasks:
            selected.append(row); distinct_tasks.add(row["task_index"])
    # Rotate within the other-task pool itself.  Rotating a concatenation of
    # other- and same-task episodes can wrap into the same-task suffix even
    # when enough genuinely unrelated task episodes are available.
    remaining_other = [row for row in other if row not in selected]
    if remaining_other and len(selected) < count:
        rotation = target["replay_index"] % len(remaining_other)
        rotated = remaining_other[rotation:] + remaining_other[:rotation]
        selected.extend(rotated[: count - len(selected)])
    if same and len(selected) < count:
        rotation = target["replay_index"] % len(same)
        rotated = same[rotation:] + same[:rotation]
        selected.extend(rotated[: count - len(selected)])
    if (len(selected) != count
            or len({row["episode_id"] for row in selected}) != count):
        raise ValueError("CogACT companion schedule lacks distinct unrelated episodes")
    return [row["replay_index"] for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output_root / "replay_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    task_manifest = json.loads(args.task_manifest.read_text())
    if (task_manifest.get("task_suite") != "simplerenv_cogact_semantic_12_variant"
            or len(task_manifest.get("tasks", [])) != 12):
        raise ValueError("CogACT task manifest differs from the pinned evaluator suite")
    if not args.adapter_evidence.is_file():
        raise FileNotFoundError(args.adapter_evidence)

    chosen = selected_episodes(task_manifest["tasks"])
    diagnostic_episodes = {
        (task["task_id"], initialization)
        for task, initialization in (
            [(task, 0) for task in task_manifest["tasks"]]
            + [(task, 1) for task in task_manifest["tasks"][:8]]
        )
    }
    bundle_paths = []
    for task, initialization in chosen:
        bundle_path = (
            args.output_root / "episodes"
            / f"task_{task['task_id']:02d}_init_{initialization:02d}.json"
        )
        source = rollout_path(args.rollouts, task["task_id"], initialization)
        if bundle_path.is_file():
            bundle = json.loads(bundle_path.read_text())
            if not validate_bundle(bundle, args.output_root, task["task_id"], initialization):
                raise ValueError(f"existing CogACT replay bundle changed: {bundle_path}")
        else:
            bundle = reconstruct_episode(
                task=task, initialization=initialization,
                rollout_path_value=source, output_root=args.output_root,
            )
            write_json_atomic(bundle_path, bundle)
        bundle_paths.append(bundle_path)
        print(
            f"reconstructed CogACT task={task['task_id']} initialization={initialization}",
            flush=True,
        )

    rows = []
    for bundle_path in bundle_paths:
        bundle = json.loads(bundle_path.read_text())
        for observation in bundle["observations"]:
            replay_index = len(rows)
            task = observation["task_index"]
            initialization = observation["initialization"]
            step = observation["step"]
            rows.append({
                "id": f"cogact-task-{task:02d}-init-{initialization:02d}-step-{step:03d}",
                "replay_index": replay_index,
                "task_index": task, "task": observation["task"],
                "episode_id": f"task-{task:02d}-init-{initialization:02d}",
                "observation_index": step,
                "split": (
                    "diagnostic" if (task, initialization) in diagnostic_episodes else "heldout"
                ),
                "initialization": initialization, "reset_seed": observation["reset_seed"],
                "instruction": observation["instruction"],
                "policy_setup": observation["policy_setup"],
                "state": observation["state"], "images": observation["images"],
                "image_sha256": observation["image_sha256"],
                "physical_observation_sha256": observation["physical_observation_sha256"],
                "source_bundle": str(bundle_path.resolve()),
                "source_bundle_sha256": sha256(bundle_path),
            })
    if (len(rows) != 1000
            or sum(row["split"] == "diagnostic" for row in rows) != 200):
        raise RuntimeError("CogACT replay split is not exactly 200 diagnostic / 800 held out")
    for row in rows:
        row["companion_replay_indices"] = companion_indices(rows, row, 31)

    noise_path = args.output_root / "request_noise.npy"
    if noise_path.exists():
        raise FileExistsError(noise_path)
    temporary = noise_path.with_name(noise_path.name + f".tmp.{os.getpid()}")
    noise = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(1000, 3, 16, 7)
    )
    for row in rows:
        entries = []
        for noise_id in range(3):
            value = request_noise(
                row["task_index"], row["initialization"], row["observation_index"], noise_id
            )
            noise[row["replay_index"], noise_id] = value
            entries.append({
                "noise_id": noise_id,
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            })
        row["noise"] = entries
        row["request_ids"] = [f"{row['id']}-noise-{value}" for value in range(3)]
    noise.flush(); del noise
    os.replace(temporary, noise_path)

    checkpoint_files = [
        args.checkpoint / "checkpoints/CogACT-Small.pt",
        args.checkpoint / "config.json",
        args.checkpoint / "dataset_statistics.json",
    ]
    sources = [Path(json.loads(path.read_text())["rollout_source"]) for path in bundle_paths]
    report = {
        "schema_version": 1, "model_id": "cogact_small",
        "dataset": "paired CogACT SimplerEnv exact rollout replay",
        "fixture_root": str(args.output_root.resolve()),
        "source_root": str(args.rollouts.resolve()),
        "source_sha256": {
            str(path.resolve().relative_to(args.rollouts.resolve())): sha256(path)
            for path in sources
        },
        "source_task_manifest": str(args.task_manifest.resolve()),
        "source_task_manifest_sha256": sha256(args.task_manifest),
        "source_episode_bundles": [str(path.resolve()) for path in bundle_paths],
        "source_episode_bundle_sha256": [sha256(path) for path in bundle_paths],
        "episode_split_method": "20 pinned task-balanced episodes; no episode crosses splits",
        "split_counts": {"diagnostic": 200, "heldout": 800},
        "selected_episodes": 100, "observations_per_episode": 10,
        "companion_schedule": {
            "version": "unrelated_episode_heads_v1", "max_companions": 31,
            "distinct_episode_required": True,
        },
        "noise_file": str(noise_path.resolve()),
        "noise_file_sha256": sha256(noise_path),
        "noise_shape": [1000, 3, 16, 7], "noise_dtype": "float32",
        "noise_contract": "noise 0 reproduces paired evaluator; noises 1/2 append noise ID",
        "checkpoint_files": [str(path.resolve()) for path in checkpoint_files],
        "checkpoint_sha256": [sha256(path) for path in checkpoint_files],
        "adapter_evidence_path": str(args.adapter_evidence.resolve()),
        "adapter_evidence_sha256": sha256(args.adapter_evidence),
        "observations": rows,
    }
    write_json_atomic(manifest_path, report)
    print(json.dumps({
        "observations": len(rows), "diagnostic": 200, "heldout": 800,
        "episodes": len(bundle_paths), "manifest": str(manifest_path),
    }))


if __name__ == "__main__":
    main()

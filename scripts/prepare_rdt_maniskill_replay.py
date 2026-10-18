"""Reconstruct 1,000 exact RDT ManiSkill policy observations from paired rollouts.

The completed native-singleton success campaign retained reset seeds and every
executed action.  This script replays those exact 250 episodes in the pinned
official environments and captures four genuine policy-request boundaries per
episode, including both history images and real 8-D proprioception.  Episode
bundles make rendering resumable.  Three request-local [64,128] diffusion
noises are then stored for each observation; noise 0 exactly follows the
paired evaluator's request-noise contract.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401  # registers the pinned task IDs
import numpy as np
from PIL import Image


NOISE_DOMAIN = 0x524454
BASE_SEED = 20241201
TASKS = (
    "PegInsertionSide-v1", "PickCube-v1", "StackCube-v1",
    "PlugCharger-v1", "PushCube-v1",
)


def numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def render_image(env: Any) -> np.ndarray:
    return np.ascontiguousarray(numpy_value(env.render()).squeeze(0), dtype=np.uint8)


def proprio(obs: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(
        numpy_value(obs["agent"]["qpos"])[:, :-1].reshape(-1), dtype=np.float32
    )


def physical_observation_hash(obs: dict[str, Any], image: np.ndarray) -> str:
    digest = hashlib.sha256()
    for key, value in (("qpos", proprio(obs)), ("render", image)):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def make_env(task_name: str) -> Any:
    return gym.make(
        task_name,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        sim_backend="cpu",
    )


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


def read_rollout(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError(f"expected one native-singleton episode in {path}")
    return episodes[0]


def rollout_path(root: Path, task: int, initialization: int) -> Path:
    candidates = [
        root / f"maniskill_{task}_initializations_{initialization}_{initialization}.json",
        root / f"maniskill_{task}_initializations_{initialization}_{initialization}.json.gz",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"exactly one rollout required for task/init {task}/{initialization}")
    return found[0]


def selected_boundaries(length: int) -> list[int]:
    chunks = (length + 15) // 16
    if chunks < 4:
        raise ValueError(f"episode has only {chunks} policy request boundaries")
    selected = np.linspace(0, chunks - 1, 4, dtype=np.int64).tolist()
    if len(set(selected)) != 4:
        raise ValueError("four selected policy boundaries are not distinct")
    return [16 * value for value in selected]


def save_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(temporary, format="PNG")
    os.replace(temporary, path)
    return sha256(path)


def validate_bundle(bundle: dict, output_root: Path, task: int, initialization: int) -> bool:
    rows = bundle.get("observations")
    if (
        bundle.get("schema_version") != 1
        or bundle.get("task") != task
        or bundle.get("initialization") != initialization
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        return False
    for row in rows:
        for key, relative in row.get("images", {}).items():
            if relative is None:
                continue
            path = output_root / relative
            if not path.is_file() or sha256(path) != row.get("image_sha256", {}).get(key):
                return False
    return True


def reconstruct_episode(
    *, task: int, initialization: int, rollout: dict, output_root: Path,
    rollout_source: Path,
) -> dict:
    actions = rollout.get("action_sequence")
    states = rollout.get("trajectory")
    if (
        rollout.get("task") != task
        or rollout.get("task_name") != TASKS[task]
        or rollout.get("initialization_id") != initialization
        or rollout.get("reset_seed") != BASE_SEED + initialization
        or not isinstance(actions, list)
        or not isinstance(states, list)
        or len(actions) != len(states)
    ):
        raise ValueError(f"rollout provenance differs for task/init {task}/{initialization}")
    selected = selected_boundaries(len(actions))
    selected_set = set(selected)
    env = make_env(TASKS[task])
    captures = []
    try:
        obs, _ = env.reset(seed=BASE_SEED + initialization)
        current = render_image(env)
        if physical_observation_hash(obs, current) != rollout.get("initial_observation_sha256"):
            raise ValueError(f"reset observation changed for task/init {task}/{initialization}")
        history: deque[np.ndarray | None] = deque((None, current), maxlen=2)
        for step, action in enumerate(actions):
            observed_state = proprio(obs)
            if not np.array_equal(observed_state, np.asarray(states[step], dtype=np.float32)):
                raise ValueError(
                    f"replayed proprio differs at task/init/step {task}/{initialization}/{step}"
                )
            if step in selected_set:
                chunk_index = step // 16
                images: dict[str, str | None] = {}
                image_hashes: dict[str, str | None] = {}
                for history_index, image in enumerate(history):
                    key = f"history_{history_index}"
                    if image is None:
                        images[key] = None
                        image_hashes[key] = None
                    else:
                        relative = (
                            f"frames/task_{task:02d}_init_{initialization:02d}_"
                            f"chunk_{chunk_index:03d}_{key}.png"
                        )
                        path = output_root / relative
                        images[key] = relative
                        image_hashes[key] = save_png(path, image)
                captures.append({
                    "task": task,
                    "task_name": TASKS[task],
                    "initialization": initialization,
                    "reset_seed": BASE_SEED + initialization,
                    "step": step,
                    "chunk_index": chunk_index,
                    "proprio": observed_state.tolist(),
                    "images": images,
                    "image_sha256": image_hashes,
                    "history_valid": [value is not None for value in history],
                    "physical_observation_sha256": physical_observation_hash(obs, current),
                })
            obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
            current = render_image(env)
            history.append(current)
    finally:
        env.close()
    if [row["step"] for row in captures] != selected:
        raise RuntimeError(f"missing selected replay boundaries for {task}/{initialization}")
    return {
        "schema_version": 1,
        "task": task,
        "task_name": TASKS[task],
        "initialization": initialization,
        "reset_seed": BASE_SEED + initialization,
        "rollout_source": str(rollout_source.resolve()),
        "rollout_source_sha256": sha256(rollout_source),
        "observations": captures,
    }


def request_noise(task: int, initialization: int, chunk: int, noise_id: int) -> np.ndarray:
    entropy = [BASE_SEED, task, initialization, chunk, NOISE_DOMAIN]
    if noise_id:
        entropy.append(noise_id)
    return np.random.default_rng(np.random.SeedSequence(entropy)).standard_normal(
        (64, 128), dtype=np.float32
    )


def companion_indices(rows: list[dict], target: dict, count: int) -> list[int]:
    episode_heads = {}
    for row in rows:
        episode_heads.setdefault((row["task_index"], row["episode_id"]), row)
    by_task = {}
    for row in episode_heads.values():
        if row["task_index"] != target["task_index"]:
            by_task.setdefault(row["task_index"], []).append(row)
    chosen, used = [], set()
    for task in sorted(by_task):
        options = sorted(by_task[task], key=lambda row: row["episode_id"])
        row = options[target["replay_index"] % len(options)]
        chosen.append(row)
        used.add((row["task_index"], row["episode_id"]))
    for same_task in (False, True):
        pool = [
            row for row in episode_heads.values()
            if (row["task_index"], row["episode_id"]) not in used
            and (row["task_index"], row["episode_id"])
            != (target["task_index"], target["episode_id"])
            and (row["task_index"] == target["task_index"]) == same_task
        ]
        pool.sort(key=lambda row: (row["task_index"], row["episode_id"]))
        if pool:
            rotation = target["replay_index"] % len(pool)
            chosen.extend((pool[rotation:] + pool[:rotation])[: count - len(chosen)])
        if len(chosen) == count:
            break
    if len(chosen) != count or len({row["episode_id"] for row in chosen}) != count:
        raise ValueError("RDT companion schedule lacks distinct unrelated episodes")
    return [row["replay_index"] for row in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--siglip", type=Path, required=True)
    parser.add_argument("--adapter-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output_root / "replay_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    task_manifest = json.loads(args.task_manifest.read_text())
    if task_manifest.get("task_names") != list(TASKS):
        raise ValueError("RDT task manifest differs from the five pinned tasks")
    if not args.adapter_evidence.is_file():
        raise FileNotFoundError(args.adapter_evidence)

    bundle_paths = []
    for task in range(5):
        for initialization in range(50):
            bundle_path = args.output_root / "episodes" / f"task_{task:02d}_init_{initialization:02d}.json"
            source = rollout_path(args.rollouts, task, initialization)
            if bundle_path.is_file():
                bundle = json.loads(bundle_path.read_text())
                if not validate_bundle(bundle, args.output_root, task, initialization):
                    raise ValueError(f"existing replay bundle is incomplete or changed: {bundle_path}")
            else:
                rollout = read_rollout(source)
                bundle = reconstruct_episode(
                    task=task,
                    initialization=initialization,
                    rollout=rollout,
                    output_root=args.output_root,
                    rollout_source=source,
                )
                write_json_atomic(bundle_path, bundle)
            bundle_paths.append(bundle_path)
            print(f"reconstructed RDT task={task} initialization={initialization}", flush=True)

    rows = []
    for bundle_path in bundle_paths:
        bundle = json.loads(bundle_path.read_text())
        for observation in bundle["observations"]:
            replay_index = len(rows)
            task = observation["task"]
            initialization = observation["initialization"]
            chunk = observation["chunk_index"]
            split = "diagnostic" if initialization % 5 == 0 else "heldout"
            rows.append({
                "id": f"rdt-task-{task:02d}-init-{initialization:02d}-chunk-{chunk:03d}",
                "replay_index": replay_index,
                "task_index": task,
                "task": observation["task_name"],
                "episode_id": f"task-{task:02d}-init-{initialization:02d}",
                "observation_index": observation["step"],
                "split": split,
                "initialization": initialization,
                "reset_seed": observation["reset_seed"],
                "chunk_index": chunk,
                "proprio": observation["proprio"],
                "images": observation["images"],
                "image_sha256": observation["image_sha256"],
                "history_valid": observation["history_valid"],
                "physical_observation_sha256": observation["physical_observation_sha256"],
                "source_bundle": str(bundle_path.resolve()),
                "source_bundle_sha256": sha256(bundle_path),
            })
    if len(rows) != 1000 or sum(row["split"] == "diagnostic" for row in rows) != 200:
        raise RuntimeError("RDT replay split is not exactly 200 diagnostic / 800 held out")
    for row in rows:
        row["companion_replay_indices"] = companion_indices(rows, row, 31)

    noise_path = args.output_root / "request_noise.npy"
    if noise_path.exists():
        raise FileExistsError(noise_path)
    noise_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_noise = noise_path.with_name(noise_path.name + f".tmp.{os.getpid()}")
    noise = np.lib.format.open_memmap(
        temporary_noise, mode="w+", dtype=np.float32, shape=(1000, 3, 64, 128)
    )
    for row in rows:
        entries = []
        for noise_id in range(3):
            value = request_noise(
                row["task_index"], row["initialization"], row["chunk_index"], noise_id
            )
            noise[row["replay_index"], noise_id] = value
            entries.append({
                "noise_id": noise_id,
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            })
        row["noise"] = entries
        row["request_ids"] = [f"{row['id']}-noise-{noise_id}" for noise_id in range(3)]
    noise.flush()
    del noise
    os.replace(temporary_noise, noise_path)

    artifacts = [
        args.checkpoint / "rdt/mp_rank_00_model_states.pt",
        *sorted((args.checkpoint / "lang_embeds").glob("*.pt")),
        args.siglip / "model.safetensors",
        args.siglip / "config.json",
        args.siglip / "preprocessor_config.json",
    ]
    rollout_sources = [Path(json.loads(path.read_text())["rollout_source"]) for path in bundle_paths]
    source_sha256 = {
        str(path.resolve().relative_to(args.rollouts.resolve())): sha256(path)
        for path in rollout_sources
    }
    report = {
        "schema_version": 1,
        "model_id": "rdt_1b",
        "dataset": "paired RDT ManiSkill five-task exact rollout replay",
        "fixture_root": str(args.output_root.resolve()),
        "source_root": str(args.rollouts.resolve()),
        "source_sha256": source_sha256,
        "source_task_manifest": str(args.task_manifest.resolve()),
        "source_task_manifest_sha256": sha256(args.task_manifest),
        "source_episode_bundles": [str(path.resolve()) for path in bundle_paths],
        "source_episode_bundle_sha256": [sha256(path) for path in bundle_paths],
        "episode_split_method": "initialization_id_mod_5_equals_zero",
        "split_counts": {"diagnostic": 200, "heldout": 800},
        "selected_episodes": 250,
        "observations_per_episode": 4,
        "companion_schedule": {
            "version": "unrelated_episode_heads_v1",
            "max_companions": 31,
            "distinct_episode_required": True,
        },
        "noise_file": str(noise_path.resolve()),
        "noise_file_sha256": sha256(noise_path),
        "noise_shape": [1000, 3, 64, 128],
        "noise_dtype": "float32",
        "noise_contract": (
            "noise 0 reproduces paired evaluator SeedSequence; noises 1/2 append a request-local ID"
        ),
        "checkpoint_files": [str(path.resolve()) for path in artifacts],
        "checkpoint_sha256": [sha256(path) for path in artifacts],
        "adapter_evidence_path": str(args.adapter_evidence.resolve()),
        "adapter_evidence_sha256": sha256(args.adapter_evidence),
        "observations": rows,
    }
    write_json_atomic(manifest_path, report)
    print(json.dumps({
        "observations": len(rows),
        "diagnostic": sum(row["split"] == "diagnostic" for row in rows),
        "heldout": sum(row["split"] == "heldout" for row in rows),
        "request_noise_pairs": sum(len(row["request_ids"]) for row in rows),
        "output": str(manifest_path),
    }))


if __name__ == "__main__":
    main()

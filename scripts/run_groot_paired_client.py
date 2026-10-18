"""Run one resumable paired GR00T N1.7 condition on official LIBERO-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
CHECKPOINT = Path("/workspace/assets/groot_n17/repo/libero_10")
MODEL_ROOT = Path("/workspace/models/isaac-groot")
NOISE_SHAPE = (40, 132)
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
DUMMY_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


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


def physical_observation_hash(obs: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in (
        "video.image", "video.wrist_image", "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw", "state.gripper",
        "annotation.human.action.task_description",
    ):
        value = obs[key]
        if isinstance(value, str):
            digest.update(key.encode())
            digest.update(value.encode())
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def state_vector(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate([
        np.asarray(obs[f"state.{key}"], dtype=np.float32).reshape(-1)
        for key in ACTION_KEYS
    ]).astype(np.float32)


def request_noise(seed: int, task: int, initialization: int, chunk: int) -> np.ndarray:
    sequence = np.random.SeedSequence([seed, task, initialization, chunk, 0x47524F4F54])
    return np.random.default_rng(sequence).standard_normal(NOISE_SHAPE, dtype=np.float32)


def noise_hash(noise: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(noise, dtype=np.float32).tobytes()).hexdigest()


def success_summary(successes: list[bool]) -> dict[str, Any]:
    count, passed = len(successes), sum(successes)
    proportion = passed / count
    z = 1.959963984540054
    z2 = z * z
    denominator = 1 + z2 / count
    centre = (proportion + z2 / (2 * count)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / count + z2 / (4 * count * count)
    ) / denominator
    return {
        "n_episodes": count,
        "n_success": passed,
        "pc_success": 100 * proportion,
        "pc_success_ci95": [100 * max(0.0, centre - half), 100 * min(1.0, centre + half)],
    }


class PairedClient:
    def __init__(self, host: str, port: int, condition: str) -> None:
        from gr00t.policy.server_client import PolicyClient

        self.client = PolicyClient(host=host, port=port, timeout_ms=300_000, strict=False)
        if not self.client.ping():
            raise RuntimeError("GR00T paired server did not answer ping")
        self.metadata = self.client.get_modality_config()
        required_batch, repaired = CONDITIONS[condition]
        expected = {
            "protocol_version": "1.0",
            "policy_type": "groot_n17",
            "task_suite": "libero_10",
            "paired_condition": condition,
            "maximum_request_batch": required_batch,
            "batch_invariant_ops": repaired,
            "genuine_batched_policy_call": True,
            "request_local_noise": True,
            "noise_shape": [40, 132],
            "flow_steps": 4,
            "action_horizon": 16,
            "executed_action_steps": 8,
            "action_dim": 7,
        }
        if any(self.metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"GR00T server metadata differs: {self.metadata}")

    def request(self, rows: list[dict[str, Any]], seed: int, task: int) -> dict[int, np.ndarray]:
        examples = []
        for row in rows:
            obs = row["obs"]
            chunk = row["chunk_index"]
            noise = request_noise(seed, task, row["initialization_id"], chunk)
            examples.append({
                "image": np.asarray(obs["video.image"], dtype=np.uint8),
                "wrist_image": np.asarray(obs["video.wrist_image"], dtype=np.uint8),
                "state": state_vector(obs),
                "task": row["task_description"],
                "noise": noise,
                "paired_request_id": (
                    f"task{task}-init{row['initialization_id']}-chunk{chunk}"
                ),
            })
        response, _ = self.client.get_action({"examples": examples})
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        ids = [example["paired_request_id"] for example in examples]
        if (
            actions.shape != (len(rows), 16, 7)
            or response.get("request_ids") != ids
            or response.get("physical_batch_size") != len(rows)
        ):
            raise RuntimeError("GR00T response differs from requested physical batch")
        return {row["initialization_id"]: actions[index, :8] for index, row in enumerate(rows)}

    def close(self) -> None:
        self.client.close()


def cache_path(output: Path, task: int, ids: list[int]) -> Path:
    return output / "rollouts" / f"libero_10_{task}_initializations_{ids[0]}_{ids[-1]}.json"


def load_cached(path: Path, task: int, ids: list[int], seed: int) -> list[dict] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if (
        value.get("task_id") != task
        or value.get("explicit_init_state_ids") != ids
        or value.get("seeds") != [seed + index for index in ids]
    ):
        raise RuntimeError(f"cached GR00T rollout has another reset schedule: {path}")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or [row.get("initialization_id") for row in episodes] != ids:
        raise RuntimeError(f"cached GR00T rollout is incomplete: {path}")
    return episodes


def run_group(
    client: PairedClient,
    task: Any,
    task_id: int,
    initial_states: Any,
    ids: list[int],
    *,
    seed: int,
    num_steps_wait: int,
) -> list[dict]:
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
    from libero.libero.utils import get_libero_path

    environments = []
    rows: dict[int, dict[str, Any]] = {}
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    try:
        for initialization in ids:
            env = LiberoEnv(task_bddl_file=bddl, task_description=task.language)
            env.reset(seed=seed + initialization)
            raw = env._env.set_init_state(initial_states[initialization])
            for _ in range(num_steps_wait):
                raw, _, _, _ = env._env.step(DUMMY_ACTION)
            obs = env._process_observation(raw)
            first_noise = request_noise(seed, task_id, initialization, 0)
            environments.append((initialization, env))
            rows[initialization] = {
                "task": task_id,
                "task_group": "libero_10",
                "task_description": task.language,
                "initialization_id": initialization,
                "reset_seed": seed + initialization,
                "initial_observation_sha256": physical_observation_hash(obs),
                "first_noise_sha256": noise_hash(first_noise),
                "success": False,
                "trajectory": [],
                "action_sequence": [],
                "rewards": [],
                "done": [],
                "policy_batch_sizes": [],
                "obs": obs,
                "active": True,
                "action_plan": deque(),
                "chunk_index": 0,
            }
        env_by_id = dict(environments)
        for _step in range(720):
            active = [rows[index] for index in ids if rows[index]["active"]]
            if not active:
                break
            requesting = [row for row in active if not row["action_plan"]]
            if requesting:
                chunks = client.request(requesting, seed, task_id)
                for row in requesting:
                    row["action_plan"].extend(chunks[row["initialization_id"]])
                    row["policy_batch_sizes"].append(len(requesting))
                    row["chunk_index"] += 1
            for row in active:
                flat_action = np.asarray(row["action_plan"].popleft(), dtype=np.float32).copy()
                action = {
                    f"action.{key}": flat_action[index:index + 1]
                    for index, key in enumerate(ACTION_KEYS)
                }
                row["trajectory"].append(state_vector(row["obs"]).tolist())
                obs, reward, terminated, truncated, info = env_by_id[
                    row["initialization_id"]
                ].step(action)
                successful = bool(info.get("success", False))
                row["action_sequence"].append(flat_action.tolist())
                row["rewards"].append(float(reward))
                row["done"].append(successful)
                row["obs"] = obs
                if successful or terminated or truncated:
                    row["success"] = successful
                    row["active"] = False
        result = []
        for initialization in ids:
            row = rows[initialization]
            for key in ("obs", "active", "action_plan", "chunk_index"):
                row.pop(key)
            result.append(row)
        return result
    finally:
        for _, env in environments:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    args = parser.parse_args()
    if args.task_limit is not None and not 1 <= args.task_limit <= 10:
        raise ValueError("task-limit must be in [1, 10]")
    if not 1 <= args.episodes_per_task <= 50:
        raise ValueError("episodes-per-task must be in [1, 50]")
    if os.environ.get("MUJOCO_GL") != "egl" or not os.environ.get("LIBERO_CONFIG_PATH"):
        raise EnvironmentError("GR00T paired client requires EGL and pinned LIBERO_CONFIG_PATH")

    task_manifest = json.loads(args.task_manifest.read_text())
    batch_size, repaired = CONDITIONS[args.condition]
    own_source = Path(__file__).resolve()
    server_source = own_source.with_name("run_groot_paired_server.py")
    manifest = {
        "schema_version": 1,
        "model": "groot_n17",
        "condition": args.condition,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_index_sha256": sha256(CHECKPOINT / "model.safetensors.index.json"),
        "artifact_files": task_manifest["artifact_files"],
        "official_evaluator_sha256": sha256(MODEL_ROOT / "gr00t/eval/rollout_policy.py"),
        "official_environment_sha256": sha256(
            MODEL_ROOT / "gr00t/eval/sim/LIBERO/libero_env.py"
        ),
        "official_policy_sha256": sha256(MODEL_ROOT / "gr00t/policy/gr00t_policy.py"),
        "runtime_helper_sha256": sha256(own_source.with_name("groot_runtime.py")),
        "paired_wrapper_sha256": sha256(own_source),
        "shared_rollout_wrapper_sha256": sha256(server_source),
        "task_manifest_path": str(args.task_manifest.resolve()),
        "task_manifest_sha256": sha256(args.task_manifest),
        "episodes_per_task": args.episodes_per_task,
        "task_count": args.task_limit or 10,
        "seed": args.seed,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "episode_length": 720,
        "num_steps_wait": args.num_steps_wait,
        "num_open_loop_steps": 8,
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "noise_shape": [40, 132],
        "flow_steps": 4,
        "genuine_batched_policy_call": True,
        "official_processor_and_decode": True,
        "official_environment_action_transform": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("GR00T paired resume provenance differs")
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    descriptions = [suite.get_task(index).language for index in range(10)]
    if descriptions != task_manifest.get("task_descriptions"):
        raise RuntimeError("live LIBERO-10 tasks differ from pinned GR00T task manifest")
    client = PairedClient(args.host, args.port, args.condition)
    write_json_atomic(args.output / "server_metadata.json", client.metadata)
    started = time.time()
    try:
        for task_id in range(args.task_limit or 10):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            for start in range(0, args.episodes_per_task, batch_size):
                ids = list(range(start, min(start + batch_size, args.episodes_per_task)))
                path = cache_path(args.output, task_id, ids)
                if load_cached(path, task_id, ids, args.seed) is not None:
                    continue
                episodes = run_group(
                    client,
                    task,
                    task_id,
                    initial_states,
                    ids,
                    seed=args.seed,
                    num_steps_wait=args.num_steps_wait,
                )
                write_json_atomic(path, {
                    "task_id": task_id,
                    "task_group": "libero_10",
                    "seeds": [args.seed + index for index in ids],
                    "explicit_init_state_ids": ids,
                    "episodes": episodes,
                })
                print(
                    f"GR00T {args.condition}: task={task_id} ids={ids} "
                    f"success={sum(row['success'] for row in episodes)}",
                    flush=True,
                )
    finally:
        client.close()

    per_task, overall_successes = [], []
    for task_id in range(args.task_limit or 10):
        rows = {}
        description = None
        for path in sorted((args.output / "rollouts").glob(f"libero_10_{task_id}_*.json")):
            for episode in json.loads(path.read_text())["episodes"]:
                initialization = episode["initialization_id"]
                if initialization in rows:
                    raise RuntimeError(f"duplicate GR00T episode {task_id}/{initialization}")
                rows[initialization] = episode
                description = episode["task_description"]
        if set(rows) != set(range(args.episodes_per_task)):
            raise RuntimeError(
                f"GR00T task {task_id} has {len(rows)}/{args.episodes_per_task} episodes"
            )
        successes = [rows[index]["success"] for index in range(args.episodes_per_task)]
        overall_successes.extend(successes)
        per_task.append({
            "task_group": "libero_10",
            "task_id": task_id,
            "task_description": description,
            "metrics": {
                "successes": successes,
                "sum_rewards": [sum(rows[index]["rewards"]) for index in range(args.episodes_per_task)],
            },
            **success_summary(successes),
        })
    info = {
        "per_task": per_task,
        "per_group": {"libero_10": success_summary(overall_successes)},
        "overall": {**success_summary(overall_successes), "eval_s": time.time() - started},
    }
    write_json_atomic(args.output / "official/eval_info.json", info)
    print(
        f"Verified GR00T {args.condition}: "
        f"{info['overall']['n_success']}/{info['overall']['n_episodes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

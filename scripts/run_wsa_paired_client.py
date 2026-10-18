"""Run one resumable paired WSA-Base condition on official LIBERO-10."""

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

from evaluation.LIBERO.eval_libero_server_client import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
)
from evaluation.LIBERO.model2libero_interface import LiberoModelClient
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
CHECKPOINT = Path("/workspace/assets/wsa/checkpoint")
MODEL_ROOT = Path("/workspace/models/wsa")
NOISE_SHAPE = (10, 32)


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
        "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos",
    ):
        value = np.ascontiguousarray(np.asarray(obs[key]))
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def request_noise(seed: int, task: int, initialization: int, chunk: int) -> np.ndarray:
    sequence = np.random.SeedSequence([seed, task, initialization, chunk, 0x575341])
    return np.random.default_rng(sequence).standard_normal(NOISE_SHAPE, dtype=np.float32)


def noise_hash(noise: np.ndarray) -> str:
    value = np.ascontiguousarray(noise, dtype=np.float32)
    return hashlib.sha256(value.tobytes()).hexdigest()


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


def build_state(obs: dict[str, Any]) -> np.ndarray:
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    axisangle = (
        np.zeros(3, dtype=np.float32)
        if math.isclose(float(denominator), 0.0)
        else quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator
    )
    return np.concatenate((
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32), axisangle,
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    )).astype(np.float32)


def update_history(row: dict[str, Any]) -> None:
    head = np.ascontiguousarray(np.asarray(row["obs"]["agentview_image"])[::-1, ::-1])
    wrist = np.ascontiguousarray(np.asarray(row["obs"]["robot0_eye_in_hand_image"])[::-1, ::-1])
    if len(row["action_plan"]) <= 15:
        row["head_history"].append(head)
        row["wrist_history"].append(wrist)
        del row["head_history"][:-16]
        del row["wrist_history"][:-16]
    if not row["head_history"]:
        row["head_history"].append(head)
        row["wrist_history"].append(wrist)


def history_pair(history: list[np.ndarray]) -> np.ndarray:
    past = max(len(history) - 16, 0)
    return np.stack([history[past], history[-1]], axis=0)


class PairedClient:
    def __init__(self, host: str, port: int, condition: str) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        required_batch = 1 if condition.endswith("singleton") else 4
        if (self.metadata.get("protocol_version") != "2.1"
                or self.metadata.get("policy_type") != "wsa_base"
                or self.metadata.get("task_suite") != "libero_10"
                or self.metadata.get("paired_condition") != condition
                or self.metadata.get("genuine_batched_policy_call") is not True
                or self.metadata.get("maximum_request_batch") != required_batch
                or self.metadata.get("request_local_noise") is not True
                or self.metadata.get("noise_shape") != [10, 32]):
            raise RuntimeError(f"WSA server metadata differs: {self.metadata}")

    def request(self, rows: list[dict[str, Any]], seed: int, task: int) -> dict[int, np.ndarray]:
        examples = []
        for row in rows:
            chunk = row["chunk_index"]
            noise = request_noise(seed, task, row["initialization_id"], chunk)
            examples.append({
                "head_history": history_pair(row["head_history"]),
                "wrist_history": history_pair(row["wrist_history"]),
                "state": build_state(row["obs"]),
                "task": row["task_description"],
                "noise": noise,
                "paired_request_id": f"task{task}-init{row['initialization_id']}-chunk{chunk}",
            })
        response = self.client.predict_action({
            "type": "infer",
            "request_id": "|".join(example["paired_request_id"] for example in examples),
            "payload": {"examples": examples},
        })
        if not response.get("ok"):
            raise RuntimeError(f"WSA inference failed: {response}")
        data = response.get("data", {})
        actions = np.asarray(data.get("actions"), dtype=np.float32)
        ids = [example["paired_request_id"] for example in examples]
        if (actions.shape != (len(rows), 10, 7) or data.get("request_ids") != ids
                or data.get("physical_batch_size") != len(rows)):
            raise RuntimeError("WSA response differs from requested physical batch")
        return {row["initialization_id"]: actions[index] for index, row in enumerate(rows)}

    def close(self) -> None:
        self.client.close()


def cache_path(output: Path, task: int, ids: list[int]) -> Path:
    return output / "rollouts" / f"libero_10_{task}_initializations_{ids[0]}_{ids[-1]}.json"


def load_cached(path: Path, task: int, ids: list[int], seed: int) -> list[dict] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if (value.get("task_id") != task or value.get("explicit_init_state_ids") != ids
            or value.get("seeds") != [seed + value for value in ids]):
        raise RuntimeError(f"cached WSA rollout has another reset schedule: {path}")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or [row.get("initialization_id") for row in episodes] != ids:
        raise RuntimeError(f"cached WSA rollout is incomplete: {path}")
    return episodes


def run_group(
    client: PairedClient, task: Any, task_id: int, initial_states: Any, ids: list[int],
    *, seed: int, num_steps_wait: int,
) -> list[dict]:
    environments = []
    rows = {}
    try:
        for initialization in ids:
            env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, 0)
            env.reset()
            obs = env.set_init_state(initial_states[initialization])
            for _ in range(num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            first_noise = request_noise(seed, task_id, initialization, 0)
            environments.append((initialization, env))
            rows[initialization] = {
                "task": task_id, "task_group": "libero_10",
                "task_description": description, "initialization_id": initialization,
                "reset_seed": seed + initialization,
                "initial_observation_sha256": physical_observation_hash(obs),
                "first_noise_sha256": noise_hash(first_noise),
                "success": False, "trajectory": [], "action_sequence": [],
                "rewards": [], "done": [], "policy_batch_sizes": [],
                "obs": obs, "active": True, "action_plan": deque(),
                "head_history": [], "wrist_history": [], "chunk_index": 0,
            }
        env_by_id = dict(environments)
        for _step in range(520):
            active = [rows[index] for index in ids if rows[index]["active"]]
            if not active:
                break
            for row in active:
                update_history(row)
            requesting = [row for row in active if not row["action_plan"]]
            if requesting:
                chunks = client.request(requesting, seed, task_id)
                for row in requesting:
                    row["action_plan"].extend(chunks[row["initialization_id"]])
                    row["policy_batch_sizes"].append(len(requesting))
                    row["chunk_index"] += 1
            for row in active:
                action = np.asarray(row["action_plan"].popleft(), dtype=np.float32).copy()
                action = action[:7]
                action[6] = 1.0 - 2.0 * (action[6] > 0.5)
                row["trajectory"].append(build_state(row["obs"]).tolist())
                obs, reward, done, _ = env_by_id[row["initialization_id"]].step(action.tolist())
                row["action_sequence"].append(action.tolist())
                row["rewards"].append(float(reward))
                row["done"].append(bool(done))
                row["obs"] = obs
                if done:
                    row["success"], row["active"] = True, False
        result = []
        for initialization in ids:
            row = rows[initialization]
            for key in (
                "obs", "active", "action_plan", "head_history", "wrist_history", "chunk_index",
            ):
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
        raise EnvironmentError("WSA paired client requires EGL and pinned LIBERO_CONFIG_PATH")
    task_manifest = json.loads(args.task_manifest.read_text())
    batch_size, repaired = CONDITIONS[args.condition]
    own_source = Path(__file__).resolve()
    server_source = own_source.with_name("run_wsa_paired_server.py")
    manifest = {
        "schema_version": 1, "model": "wsa_base", "condition": args.condition,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT / "model.safetensors"),
        "artifact_files": task_manifest["artifact_files"],
        "official_evaluator_sha256": sha256(MODEL_ROOT / "evaluation/Libero/inference.py"),
        "official_server_sha256": sha256(MODEL_ROOT / "evaluation/Libero/model_server.py"),
        "official_policy_sha256": sha256(MODEL_ROOT / "src/lerobot/policies/WSA_Base/modeling_wsa_base.py"),
        "paired_wrapper_sha256": sha256(own_source),
        "shared_rollout_wrapper_sha256": sha256(server_source),
        "task_manifest_path": str(args.task_manifest.resolve()),
        "task_manifest_sha256": sha256(args.task_manifest),
        "episodes_per_task": args.episodes_per_task,
        "task_count": args.task_limit or 10,
        "seed": args.seed, "batch_size": batch_size,
        "invariant_ops": repaired, "task_suite": "libero_10", "episode_length": 520,
        "num_steps_wait": args.num_steps_wait, "num_open_loop_steps": 10,
        "image_history_interval": 15, "control_mode": "absolute",
        "gripper_mode": "libero_open_prob", "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True, "noise_shape": [10, 32], "flow_steps": 10,
        "genuine_batched_policy_call": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("WSA paired resume provenance differs")
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    descriptions = [suite.get_task(index).language for index in range(10)]
    if descriptions != task_manifest.get("task_descriptions"):
        raise RuntimeError("live LIBERO-10 tasks differ from pinned WSA task manifest")
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
                    client, task, task_id, initial_states, ids,
                    seed=args.seed, num_steps_wait=args.num_steps_wait,
                )
                write_json_atomic(path, {
                    "task_id": task_id, "task_group": "libero_10",
                    "seeds": [args.seed + value for value in ids],
                    "explicit_init_state_ids": ids, "episodes": episodes,
                })
                print(
                    f"WSA {args.condition}: task={task_id} ids={ids} "
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
                    raise RuntimeError(f"duplicate WSA episode {task_id}/{initialization}")
                rows[initialization] = episode
                description = episode["task_description"]
        if set(rows) != set(range(args.episodes_per_task)):
            raise RuntimeError(
                f"WSA task {task_id} has {len(rows)}/{args.episodes_per_task} episodes"
            )
        successes = [rows[index]["success"] for index in range(args.episodes_per_task)]
        overall_successes.extend(successes)
        per_task.append({
            "task_group": "libero_10", "task_id": task_id,
            "task_description": description,
            "metrics": {
                "successes": successes,
                "sum_rewards": [
                    sum(rows[index]["rewards"])
                    for index in range(args.episodes_per_task)
                ],
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
        f"Verified WSA {args.condition}: "
        f"{info['overall']['n_success']}/{info['overall']['n_episodes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

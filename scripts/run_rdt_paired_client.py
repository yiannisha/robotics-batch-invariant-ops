"""Run one resumable paired RDT-1B condition on the official ManiSkill tasks."""

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

import gymnasium as gym
import mani_skill.envs  # noqa: F401  # registers the official task IDs
import numpy as np

from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
TASKS = (
    "PegInsertionSide-v1", "PickCube-v1", "StackCube-v1", "PlugCharger-v1", "PushCube-v1",
)
CHECKPOINT = Path("/workspace/assets/rdt/checkpoint")
MODEL_ROOT = Path("/workspace/models/rdt")
BASE_SEED = 20241201


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


def numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def render_image(env: Any) -> np.ndarray:
    image = numpy_value(env.render()).squeeze(0)
    return np.ascontiguousarray(image, dtype=np.uint8)


def proprio(obs: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(numpy_value(obs["agent"]["qpos"])[:, :-1].reshape(-1), dtype=np.float32)


def physical_observation_hash(obs: dict[str, Any], image: np.ndarray) -> str:
    digest = hashlib.sha256()
    for key, value in (("qpos", proprio(obs)), ("render", image)):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def request_noise(task: int, initialization: int, chunk: int) -> np.ndarray:
    sequence = np.random.SeedSequence([BASE_SEED, task, initialization, chunk, 0x524454])
    return np.random.default_rng(sequence).standard_normal((64, 128), dtype=np.float32)


def noise_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def success_value(info: dict[str, Any]) -> bool:
    value = numpy_value(info.get("success", False)).reshape(-1)
    return bool(value[0]) if value.size else False


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
        "n_episodes": count, "n_success": passed, "pc_success": 100 * proportion,
        "pc_success_ci95": [100 * max(0.0, centre - half), 100 * min(1.0, centre + half)],
    }


class PairedClient:
    def __init__(self, host: str, port: int, condition: str) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        batch = 1 if condition.endswith("singleton") else 4
        if (self.metadata.get("protocol_version") != "2.1"
                or self.metadata.get("policy_type") != "rdt_1b"
                or self.metadata.get("task_suite") != "maniskill_five"
                or self.metadata.get("paired_condition") != condition
                or self.metadata.get("genuine_batched_policy_call") is not True
                or self.metadata.get("maximum_request_batch") != batch
                or self.metadata.get("request_local_noise") is not True
                or self.metadata.get("noise_shape") != [64, 128]):
            raise RuntimeError(f"RDT server metadata differs: {self.metadata}")

    def request(self, rows: list[dict[str, Any]], task_id: int) -> dict[int, np.ndarray]:
        examples = []
        for row in rows:
            history = list(row["image_history"])
            current = history[-1]
            values = [current if value is None else value for value in history]
            noise = request_noise(task_id, row["initialization_id"], row["chunk_index"])
            examples.append({
                "task_name": row["task_name"], "image_history": np.stack(values),
                "history_valid": [value is not None for value in history],
                "proprio": proprio(row["obs"]), "noise": noise,
                "paired_request_id": (
                    f"task{task_id}-init{row['initialization_id']}-chunk{row['chunk_index']}"
                ),
            })
        response = self.client.predict_action({
            "type": "infer",
            "request_id": "|".join(value["paired_request_id"] for value in examples),
            "payload": {"examples": examples},
        })
        if not response.get("ok"):
            raise RuntimeError(f"RDT inference failed: {response}")
        data = response.get("data", {})
        actions = np.asarray(data.get("actions"), dtype=np.float32)
        ids = [value["paired_request_id"] for value in examples]
        if (actions.shape != (len(rows), 64, 8) or data.get("request_ids") != ids
                or data.get("physical_batch_size") != len(rows)):
            raise RuntimeError("RDT response differs from requested physical batch")
        return {row["initialization_id"]: actions[index, ::4] for index, row in enumerate(rows)}

    def close(self) -> None:
        self.client.close()


def cache_path(output: Path, task: int, ids: list[int]) -> Path:
    return output / "rollouts" / f"maniskill_{task}_initializations_{ids[0]}_{ids[-1]}.json"


def load_cached(path: Path, task: int, ids: list[int]) -> list[dict] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    seeds = [BASE_SEED + initialization for initialization in ids]
    if (value.get("task_id") != task or value.get("explicit_init_state_ids") != ids
            or value.get("seeds") != seeds):
        raise RuntimeError(f"cached RDT rollout has another reset schedule: {path}")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or [row.get("initialization_id") for row in episodes] != ids:
        raise RuntimeError(f"cached RDT rollout is incomplete: {path}")
    return episodes


def make_env(task_name: str) -> Any:
    return gym.make(
        task_name, obs_mode="rgb", control_mode="pd_joint_pos",
        # The pinned evaluator exposes --reward-mode and forwards it here.
        # PlugCharger-v1 in the installed official ManiSkill release does not
        # implement dense reward, while sparse is supported by all five tasks.
        # Reward is never an input to the policy; success comes from info.
        render_mode="rgb_array", reward_mode="sparse",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"}, sim_backend="cpu",
    )


def run_group(client: PairedClient, task_id: int, task_name: str, ids: list[int]) -> list[dict]:
    environments, rows = [], {}
    try:
        for initialization in ids:
            env = make_env(task_name)
            obs, _ = env.reset(seed=BASE_SEED + initialization)
            image = render_image(env)
            environments.append((initialization, env))
            first_noise = request_noise(task_id, initialization, 0)
            rows[initialization] = {
                "task": task_id, "task_group": "maniskill_five", "task_name": task_name,
                "initialization_id": initialization, "reset_seed": BASE_SEED + initialization,
                "initial_observation_sha256": physical_observation_hash(obs, image),
                "first_noise_sha256": noise_hash(first_noise), "success": False,
                "trajectory": [], "action_sequence": [], "rewards": [], "done": [],
                "policy_batch_sizes": [], "obs": obs, "active": True,
                "action_plan": deque(), "image_history": deque((None, image), maxlen=2),
                "chunk_index": 0,
            }
        env_by_id = dict(environments)
        for _step in range(400):
            active = [rows[index] for index in ids if rows[index]["active"]]
            if not active:
                break
            requesting = [row for row in active if not row["action_plan"]]
            if requesting:
                chunks = client.request(requesting, task_id)
                for row in requesting:
                    row["action_plan"].extend(chunks[row["initialization_id"]])
                    row["policy_batch_sizes"].append(len(requesting))
                    row["chunk_index"] += 1
            for row in active:
                action = np.asarray(row["action_plan"].popleft(), dtype=np.float32)
                row["trajectory"].append(proprio(row["obs"]).tolist())
                obs, reward, terminated, truncated, info = env_by_id[row["initialization_id"]].step(action)
                image = render_image(env_by_id[row["initialization_id"]])
                row["action_sequence"].append(action.tolist())
                row["rewards"].append(float(numpy_value(reward).reshape(-1)[0]))
                row["done"].append(bool(numpy_value(terminated).reshape(-1)[0]
                                        or numpy_value(truncated).reshape(-1)[0]))
                row["obs"] = obs
                row["image_history"].append(image)
                if success_value(info):
                    row["success"], row["active"] = True, False
        result = []
        for initialization in ids:
            row = rows[initialization]
            for key in ("obs", "active", "action_plan", "image_history", "chunk_index"):
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
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    args = parser.parse_args()
    if args.task_limit is not None and not 1 <= args.task_limit <= len(TASKS):
        raise ValueError(f"task-limit must be in [1, {len(TASKS)}]")
    if not 1 <= args.episodes_per_task <= 50:
        raise ValueError("episodes-per-task must be in [1, 50]")
    task_manifest = json.loads(args.task_manifest.read_text())
    batch_size, repaired = CONDITIONS[args.condition]
    own_source = Path(__file__).resolve()
    server_source = own_source.with_name("run_rdt_paired_server.py")
    manifest = {
        "schema_version": 1, "model": "rdt_1b", "condition": args.condition,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT / "rdt/mp_rank_00_model_states.pt"),
        "artifact_files": task_manifest["artifact_files"],
        "official_evaluator_sha256": sha256(MODEL_ROOT / "eval_sim/eval_rdt_maniskill.py"),
        "official_wrapper_sha256": sha256(MODEL_ROOT / "scripts/maniskill_model.py"),
        "official_policy_sha256": sha256(MODEL_ROOT / "models/rdt_runner.py"),
        "paired_wrapper_sha256": sha256(own_source),
        "shared_rollout_wrapper_sha256": sha256(server_source),
        "task_manifest_path": str(args.task_manifest.resolve()),
        "task_manifest_sha256": sha256(args.task_manifest),
        "episodes_per_task": args.episodes_per_task,
        "task_count": args.task_limit or len(TASKS),
        "seed": BASE_SEED, "batch_size": batch_size,
        "invariant_ops": repaired, "task_suite": "maniskill_five", "episode_length": 400,
        "reward_mode": "sparse",
        "control_mode": "pd_joint_pos", "control_frequency": 25,
        "action_chunk_size": 64, "executed_action_stride": 4,
        "request_local_noise": True, "noise_shape": [64, 128], "diffusion_steps": 5,
        "genuine_batched_policy_call": True,
        "language_padding": "fixed_right_30" if repaired else "longest_in_batch_right",
        "explicit_init_state_by_initialization_id": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("RDT paired resume provenance differs")
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    if task_manifest.get("task_names") != list(TASKS):
        raise RuntimeError("RDT task manifest differs from official five-task suite")
    client = PairedClient(args.host, args.port, args.condition)
    write_json_atomic(args.output / "server_metadata.json", client.metadata)
    started = time.time()
    try:
        for task_id, task_name in enumerate(TASKS[:args.task_limit]):
            for begin in range(0, args.episodes_per_task, batch_size):
                ids = list(range(begin, min(begin + batch_size, args.episodes_per_task)))
                path = cache_path(args.output, task_id, ids)
                if load_cached(path, task_id, ids) is not None:
                    continue
                episodes = run_group(client, task_id, task_name, ids)
                write_json_atomic(path, {
                    "task_id": task_id, "task_group": "maniskill_five", "task_name": task_name,
                    "seeds": [BASE_SEED + value for value in ids],
                    "explicit_init_state_ids": ids, "episodes": episodes,
                })
                print(
                    f"RDT {args.condition}: task={task_name} ids={ids} "
                    f"success={sum(row['success'] for row in episodes)}", flush=True,
                )
    finally:
        client.close()
    per_task, all_successes = [], []
    for task_id, task_name in enumerate(TASKS[:args.task_limit]):
        rows = {}
        for path in sorted((args.output / "rollouts").glob(f"maniskill_{task_id}_*.json")):
            for episode in json.loads(path.read_text())["episodes"]:
                initialization = episode["initialization_id"]
                if initialization in rows:
                    raise RuntimeError(f"duplicate RDT episode {task_id}/{initialization}")
                rows[initialization] = episode
        if set(rows) != set(range(args.episodes_per_task)):
            raise RuntimeError(
                f"RDT task {task_id} has {len(rows)}/{args.episodes_per_task} episodes"
            )
        successes = [rows[index]["success"] for index in range(args.episodes_per_task)]
        all_successes.extend(successes)
        per_task.append({
            "task_group": "maniskill_five", "task_id": task_id, "task_description": task_name,
            "metrics": {"successes": successes,
                        "sum_rewards": [sum(rows[index]["rewards"])
                                        for index in range(args.episodes_per_task)]},
            **success_summary(successes),
        })
    info = {
        "per_task": per_task, "per_group": {"maniskill_five": success_summary(all_successes)},
        "overall": {**success_summary(all_successes), "eval_s": time.time() - started},
    }
    write_json_atomic(args.output / "official/eval_info.json", info)
    print(
        f"Verified RDT {args.condition}: "
        f"{info['overall']['n_success']}/{info['overall']['n_episodes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

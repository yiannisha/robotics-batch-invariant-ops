"""Run one resumable paired OpenVLA-OFT condition on official LIBERO-Spatial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
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
DETERMINISTIC_CONTRACT = {
    "inference_head": "l1_regression",
    "stochastic_request_inputs": "none",
    "do_sample": False,
}
DETERMINISTIC_SHA256 = hashlib.sha256(
    json.dumps(DETERMINISTIC_CONTRACT, sort_keys=True).encode()
).hexdigest()
CHECKPOINT = Path("/workspace/assets/openvla_oft/checkpoint")
MODEL_ROOT = Path("/workspace/models/openvla-oft")


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
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        if (self.metadata.get("protocol_version") != "2.1"
                or self.metadata.get("policy_type") != "openvla_oft"
                or self.metadata.get("task_suite") != "libero_spatial"
                or self.metadata.get("paired_condition") != condition
                or self.metadata.get("genuine_batched_policy_call") is not True
                or self.metadata.get("deterministic_contract_sha256") != DETERMINISTIC_SHA256):
            raise RuntimeError(f"OpenVLA-OFT server metadata differs: {self.metadata}")

    def request(self, rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
        examples = [{
            "image": [row["obs"]["agentview_image"], row["obs"]["robot0_eye_in_hand_image"]],
            "state": LiberoModelClient._extract_state(row["obs"]),
            "task": row["task"],
            "paired_request_id": row["request_id"],
        } for row in rows]
        response = self.client.predict_action({
            "type": "infer", "request_id": "|".join(row["request_id"] for row in rows),
            "payload": {"examples": examples, "do_sample": False},
        })
        if not response.get("ok"):
            raise RuntimeError(f"OpenVLA-OFT inference failed: {response}")
        data = response.get("data", {})
        actions = np.asarray(data.get("actions"), dtype=np.float32)
        expected_ids = [row["request_id"] for row in rows]
        if (actions.shape != (len(rows), 8, 7)
                or data.get("request_ids") != expected_ids
                or data.get("physical_batch_size") != len(rows)
                or data.get("deterministic_contract_sha256") != DETERMINISTIC_SHA256):
            raise RuntimeError("OpenVLA-OFT response differs from requested physical batch")
        return {row["initialization_id"]: actions[index] for index, row in enumerate(rows)}

    def close(self) -> None:
        self.client.close()


def cache_path(output: Path, task: int, ids: list[int]) -> Path:
    return output / "rollouts" / f"libero_spatial_{task}_initializations_{ids[0]}_{ids[-1]}.json"


def load_cached(path: Path, task: int, ids: list[int], seed: int) -> list[dict] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if (value.get("task_id") != task or value.get("explicit_init_state_ids") != ids
            or value.get("seeds") != [seed + value for value in ids]):
        raise RuntimeError(f"cached OpenVLA-OFT rollout has another reset schedule: {path}")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or [row.get("initialization_id") for row in episodes] != ids:
        raise RuntimeError(f"cached OpenVLA-OFT rollout is incomplete: {path}")
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
            environments.append((initialization, env))
            rows[initialization] = {
                "task": task_id, "task_group": "libero_spatial",
                "task_description": description, "initialization_id": initialization,
                "reset_seed": seed + initialization,
                "initial_observation_sha256": physical_observation_hash(obs),
                "first_noise_sha256": DETERMINISTIC_SHA256,
                "success": False, "trajectory": [], "action_sequence": [],
                "rewards": [], "done": [], "policy_batch_sizes": [],
                "obs": obs, "active": True, "chunk": None,
            }
        env_by_id = dict(environments)
        for step in range(220):
            active = [rows[index] for index in ids if rows[index]["active"]]
            if not active:
                break
            if step % 8 == 0:
                requests = [{
                    "obs": row["obs"], "task": row["task_description"],
                    "initialization_id": row["initialization_id"],
                    "request_id": f"task{task_id}-init{row['initialization_id']}-chunk{step // 8}",
                } for row in active]
                chunks = client.request(requests)
                for row in active:
                    row["chunk"] = chunks[row["initialization_id"]]
                    row["policy_batch_sizes"].append(len(active))
            for row in active:
                action = np.asarray(row["chunk"][step % 8], dtype=np.float32).copy()
                # Official OpenVLA-OFT LIBERO postprocessing: [0,1] ->
                # [-1,+1], binarize, then invert RLDS gripper convention.
                action[-1] = -np.sign(2 * action[-1] - 1)
                row["trajectory"].append(LiberoModelClient._extract_state(row["obs"]).tolist())
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
            for key in ("obs", "active", "chunk"):
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
        raise EnvironmentError("OpenVLA-OFT paired client requires EGL and pinned LIBERO_CONFIG_PATH")
    task_manifest = json.loads(args.task_manifest.read_text())
    batch_size, repaired = CONDITIONS[args.condition]
    own_source = Path(__file__).resolve()
    server_source = own_source.with_name("run_openvla_oft_paired_server.py")
    manifest = {
        "schema_version": 1, "model": "openvla_oft", "condition": args.condition,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_index_sha256": sha256(CHECKPOINT / "model.safetensors.index.json"),
        "checkpoint_files": task_manifest["checkpoint_files"],
        "official_evaluator_sha256": sha256(MODEL_ROOT / "experiments/robot/libero/run_libero_eval.py"),
        "official_libero_utils_sha256": sha256(MODEL_ROOT / "experiments/robot/libero/libero_utils.py"),
        "official_policy_sha256": sha256(MODEL_ROOT / "prismatic/extern/hf/modeling_prismatic.py"),
        "paired_wrapper_sha256": sha256(own_source),
        "shared_rollout_wrapper_sha256": sha256(server_source),
        "task_manifest_path": str(args.task_manifest.resolve()),
        "task_manifest_sha256": sha256(args.task_manifest),
        "episodes_per_task": args.episodes_per_task,
        "task_count": args.task_limit or 10,
        "seed": args.seed, "batch_size": batch_size,
        "invariant_ops": repaired, "task_suite": "libero_spatial", "episode_length": 220,
        "num_steps_wait": args.num_steps_wait, "num_open_loop_steps": 8,
        "explicit_init_state_by_initialization_id": True,
        "deterministic_decoding_contract": DETERMINISTIC_CONTRACT,
        "deterministic_decoding_sha256": DETERMINISTIC_SHA256,
        "genuine_batched_policy_call": True,
        "center_crop_area": 0.9, "num_images_in_input": 2, "use_proprio": True,
        "prompt_padding": "canonical_right" if repaired else "longest_in_batch_right",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("OpenVLA-OFT paired resume provenance differs")
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    descriptions = [suite.get_task(index).language for index in range(10)]
    if descriptions != task_manifest.get("task_descriptions"):
        raise RuntimeError("live LIBERO-Spatial tasks differ from pinned task manifest")
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
                    "task_id": task_id, "task_group": "libero_spatial",
                    "seeds": [args.seed + value for value in ids],
                    "explicit_init_state_ids": ids, "episodes": episodes,
                })
                print(
                    f"OpenVLA-OFT {args.condition}: task={task_id} ids={ids} "
                    f"success={sum(row['success'] for row in episodes)}",
                    flush=True,
                )
    finally:
        client.close()

    per_task, overall_successes = [], []
    for task_id in range(args.task_limit or 10):
        rows = {}
        description = None
        for path in sorted((args.output / "rollouts").glob(f"libero_spatial_{task_id}_*.json")):
            for episode in json.loads(path.read_text())["episodes"]:
                initialization = episode["initialization_id"]
                if initialization in rows:
                    raise RuntimeError(f"duplicate OpenVLA-OFT episode {task_id}/{initialization}")
                rows[initialization] = episode
                description = episode["task_description"]
        if set(rows) != set(range(args.episodes_per_task)):
            raise RuntimeError(
                f"OpenVLA-OFT task {task_id} has "
                f"{len(rows)}/{args.episodes_per_task} episodes"
            )
        successes = [rows[index]["success"] for index in range(args.episodes_per_task)]
        overall_successes.extend(successes)
        per_task.append({
            "task_group": "libero_spatial", "task_id": task_id,
            "task_description": description,
            "metrics": {
                "successes": successes,
                "sum_rewards": [
                    sum(rows[i]["rewards"]) for i in range(args.episodes_per_task)
                ],
            },
            **success_summary(successes),
        })
    info = {
        "per_task": per_task,
        "per_group": {"libero_spatial": success_summary(overall_successes)},
        "overall": {**success_summary(overall_successes), "eval_s": time.time() - started},
    }
    write_json_atomic(args.output / "official/eval_info.json", info)
    print(
        f"Verified OpenVLA-OFT {args.condition}: "
        f"{info['overall']['n_success']}/{info['overall']['n_episodes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Run one resumable paired InternVLA-A1.5 condition on official LIBERO-10.

The simulator/task construction and robot interface are imported from the
official evaluator.  Episodes advance in synchronized groups so dynamic
conditions issue genuine B<=4 policy calls; singleton conditions issue B1.
Every group is cached atomically for safe resume.
"""

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
    TASK_SUITE_MAX_STEPS,
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
CHECKPOINT_SHA256 = "222b6ed5dbd3224f441ad2f9165838a890aad5934282209fb635ca4daf073de0"
VLM_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
OFFICIAL_ROOT = Path("/workspace/models/internvla-a-series")
OFFICIAL_CLIENT = OFFICIAL_ROOT / "evaluation/LIBERO/eval_libero_server_client.py"
OFFICIAL_BACKEND = OFFICIAL_ROOT / "evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py"
STANDARD_MODEL = OFFICIAL_ROOT / "src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py"
OPTIMIZED_MODEL = OFFICIAL_ROOT / "src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5_optimized.py"


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
    keys = (
        "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos",
    )
    digest = hashlib.sha256()
    for key in keys:
        if key not in obs:
            raise KeyError(f"LIBERO observation lacks physical sensor {key}")
        value = np.ascontiguousarray(np.asarray(obs[key]))
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def success_summary(successes: list[bool]) -> dict[str, Any]:
    count = len(successes)
    passed = sum(successes)
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


class PairedBatchClient:
    def __init__(self, host: str, port: int, *, condition: str, replan_steps: int) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        LiberoModelClient._validate_server_metadata(self.metadata)
        if self.metadata.get("paired_condition") != condition:
            raise RuntimeError(
                f"server condition {self.metadata.get('paired_condition')} != client {condition}"
            )
        if self.metadata.get("genuine_batched_policy_call") is not True:
            raise RuntimeError("InternVLA server does not attest genuine batched policy execution")
        self.chunk_size = int(self.metadata["chunk_size"])
        self.action_dim = int(self.metadata["action_dim"])
        self.replan_steps = int(replan_steps)
        if not 1 <= self.replan_steps <= self.chunk_size:
            raise ValueError("replan_steps must be within the server action chunk")

    @staticmethod
    def _example(obs: dict[str, Any], task: str, request_id: str, noise_seed: int) -> dict:
        primary = np.ascontiguousarray(np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1])
        wrist = np.ascontiguousarray(
            np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1]
        )
        return {
            "image": [primary, wrist],
            "lang": task,
            "task": task,
            "state": LiberoModelClient._extract_state(obs),
            "paired_request_id": request_id,
            "noise_seed": noise_seed,
        }

    def request_chunks(self, rows: list[dict[str, Any]]) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        examples = [
            self._example(row["obs"], row["task"], row["request_id"], row["noise_seed"])
            for row in rows
        ]
        response = self.client.predict_action({
            "type": "infer",
            "request_id": "|".join(row["request_id"] for row in rows),
            "payload": {"examples": examples, "do_sample": False},
        })
        if not response.get("ok"):
            raise RuntimeError(f"InternVLA inference error: {response}")
        data = response.get("data", {})
        actions = np.asarray(data.get("actions"), dtype=np.float32)
        expected_ids = [row["request_id"] for row in rows]
        if (actions.ndim != 3 or actions.shape[0] != len(rows)
                or data.get("request_ids") != expected_ids
                or data.get("physical_batch_size") != len(rows)):
            raise RuntimeError("InternVLA server response differs from the requested physical batch")
        hashes = data.get("noise_sha256")
        if not isinstance(hashes, list) or len(hashes) != len(rows):
            raise RuntimeError("InternVLA server omitted request-local noise hashes")
        space = data.get("action_space", {})
        low = np.asarray(space.get("low"), dtype=np.float32).reshape(-1)
        high = np.asarray(space.get("high"), dtype=np.float32).reshape(-1)
        chunks, noise_hashes = {}, {}
        for index, row in enumerate(rows):
            chunk = np.array(actions[index], dtype=np.float32, copy=True)
            dim = min(chunk.shape[-1], low.size, high.size)
            chunk[:, :dim] = np.clip(chunk[:, :dim], low[:dim], high[:dim])
            chunks[row["initialization_id"]] = chunk
            noise_hashes[row["initialization_id"]] = hashes[index]
        return chunks, noise_hashes

    def close(self) -> None:
        self.client.close()


def cache_path(output: Path, task_id: int, initializations: list[int]) -> Path:
    return output / "rollouts" / (
        f"libero_10_{task_id}_initializations_{initializations[0]}_{initializations[-1]}.json"
    )


def load_cached_group(path: Path, task_id: int, initializations: list[int], seed: int) -> list[dict] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    expected_seeds = [seed + index for index in initializations]
    if (value.get("task_id") != task_id or value.get("seeds") != expected_seeds
            or value.get("explicit_init_state_ids") != initializations):
        raise RuntimeError(f"cached InternVLA rollout has another physical reset schedule: {path}")
    episodes = value.get("episodes")
    if (not isinstance(episodes, list) or len(episodes) != len(initializations)
            or [row.get("initialization_id") for row in episodes] != initializations):
        raise RuntimeError(f"cached InternVLA rollout is incomplete: {path}")
    return episodes


def rollout_group(
    *, client: PairedBatchClient, task: Any, task_id: int, initial_states: Any,
    initializations: list[int], seed: int, num_steps_wait: int, replan_steps: int,
) -> list[dict]:
    environments = []
    records = {}
    try:
        for initialization in initializations:
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, 7)
            env.reset()
            obs = env.set_init_state(initial_states[initialization])
            for _ in range(num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            environments.append((initialization, env))
            records[initialization] = {
                "task": task_id,
                "task_group": "libero_10",
                "task_description": task_description,
                "initialization_id": initialization,
                "reset_seed": seed + initialization,
                "initial_observation_sha256": physical_observation_hash(obs),
                "first_noise_sha256": None,
                "success": False,
                "trajectory": [],
                "action_sequence": [],
                "rewards": [],
                "done": [],
                "policy_batch_sizes": [],
                "obs": obs,
                "active": True,
                "chunk": None,
            }

        max_steps = TASK_SUITE_MAX_STEPS["libero_10"]
        for step in range(max_steps):
            active = [records[index] for index in initializations if records[index]["active"]]
            if not active:
                break
            if step % replan_steps == 0:
                chunk_index = step // replan_steps
                requests = []
                for row in active:
                    request_id = f"task{task_id}-init{row['initialization_id']}-chunk{chunk_index}"
                    requests.append({
                        "obs": row["obs"], "task": row["task_description"],
                        "initialization_id": row["initialization_id"],
                        "request_id": request_id,
                        # Independent fixed seeds make the noise for a request
                        # identical regardless of its batch companions/position.
                        "noise_seed": row["reset_seed"] + chunk_index * 100_000_000,
                    })
                chunks, noise_hashes = client.request_chunks(requests)
                for row in active:
                    initialization = row["initialization_id"]
                    row["chunk"] = chunks[initialization]
                    row["policy_batch_sizes"].append(len(active))
                    if row["first_noise_sha256"] is None:
                        row["first_noise_sha256"] = noise_hashes[initialization]

            env_by_id = dict(environments)
            for row in active:
                action = np.asarray(row["chunk"][step % replan_steps], dtype=np.float32).copy()
                if action.shape[-1] >= client.action_dim:
                    action = action[:client.action_dim]
                else:
                    action = np.pad(action, (0, client.action_dim - action.shape[-1]))
                if action.shape[-1] >= 7:
                    action[6] = 1.0 if action[6] < 0.5 else -1.0
                row["trajectory"].append(LiberoModelClient._extract_state(row["obs"]).tolist())
                obs, reward, done, _ = env_by_id[row["initialization_id"]].step(action.tolist())
                row["action_sequence"].append(action.tolist())
                row["rewards"].append(float(reward))
                row["done"].append(bool(done))
                row["obs"] = obs
                if done:
                    row["success"] = True
                    row["active"] = False
        episodes = []
        for initialization in initializations:
            row = records[initialization]
            if row["first_noise_sha256"] is None:
                raise RuntimeError("InternVLA rollout never made a policy request")
            row.pop("obs")
            row.pop("active")
            row.pop("chunk")
            episodes.append(row)
        return episodes
    finally:
        for _, env in environments:
            env.close()


def validate_manifest(path: Path, expected: dict) -> None:
    if path.is_file():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError(f"InternVLA paired resume provenance differs: {path}")
    else:
        write_json_atomic(path, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("standard", "optimized"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=8)
    args = parser.parse_args()
    if args.episodes_per_task != 50:
        raise ValueError("full InternVLA paired campaign requires exactly 50 episodes per task")
    if os.environ.get("MUJOCO_GL") != "egl" or not os.environ.get("LIBERO_CONFIG_PATH"):
        raise EnvironmentError("paired InternVLA client requires EGL and pinned LIBERO_CONFIG_PATH")
    weight = args.checkpoint / "model.safetensors"
    vlm_weight = args.vlm / "model.safetensors-00001-of-00001.safetensors"
    if sha256(weight) != CHECKPOINT_SHA256 or sha256(vlm_weight) != VLM_SHA256:
        raise ValueError("InternVLA paired client inputs differ from pinned exact weights")

    batch_size, repaired = CONDITIONS[args.condition]
    own_source = Path(__file__).resolve()
    server_wrapper = own_source.with_name("run_internvla_paired_server.py")
    policy_source = STANDARD_MODEL if args.backend == "standard" else OPTIMIZED_MODEL
    model_id = f"internvla_a15_{args.backend}"
    manifest = {
        "schema_version": 1,
        "model": model_id,
        "backend": args.backend,
        "condition": args.condition,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight": str(weight.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vlm_weight": str(vlm_weight.resolve()),
        "vlm_sha256": VLM_SHA256,
        "official_evaluator_sha256": sha256(OFFICIAL_CLIENT),
        "official_backend_sha256": sha256(OFFICIAL_BACKEND),
        "official_policy_sha256": sha256(policy_source),
        "paired_wrapper_sha256": sha256(own_source),
        "shared_rollout_wrapper_sha256": sha256(server_wrapper),
        "episodes_per_task": 50,
        "seed": args.seed,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "episode_length": 520,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "genuine_batched_policy_call": True,
        "prompt_padding": "fixed_right_650" if repaired else "longest_right",
        "optimized_graph_cache_policy": (
            "recapture_per_policy_call" if args.backend == "optimized" else None
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    validate_manifest(args.output / "condition_manifest.json", manifest)

    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    client = PairedBatchClient(
        args.host, args.port, condition=args.condition, replan_steps=args.replan_steps,
    )
    write_json_atomic(args.output / "server_metadata.json", {
        key: client.metadata.get(key) for key in (
            "policy_type", "ckpt_path", "stats_key", "robot_type", "action_mode",
            "chunk_size", "action_dim", "protocol_version", "preprocessing_owner",
            "deterministic_inference_preprocess", "paired_condition",
            "genuine_batched_policy_call", "maximum_request_batch",
            "batch_invariant_ops", "request_local_noise", "prompt_padding",
            "optimized_graph_cache_policy",
        )
    })
    started = time.time()
    try:
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            for start in range(0, 50, batch_size):
                initializations = list(range(start, min(start + batch_size, 50)))
                path = cache_path(args.output, task_id, initializations)
                if load_cached_group(path, task_id, initializations, args.seed) is not None:
                    continue
                episodes = rollout_group(
                    client=client, task=task, task_id=task_id, initial_states=initial_states,
                    initializations=initializations, seed=args.seed,
                    num_steps_wait=args.num_steps_wait, replan_steps=args.replan_steps,
                )
                write_json_atomic(path, {
                    "task_id": task_id,
                    "task_group": "libero_10",
                    "seeds": [args.seed + index for index in initializations],
                    "explicit_init_state_ids": initializations,
                    "episodes": episodes,
                })
                print(
                    f"InternVLA {args.backend} {args.condition}: task={task_id} "
                    f"initializations={initializations} success={sum(row['success'] for row in episodes)}",
                    flush=True,
                )
    finally:
        client.close()

    per_task = []
    overall_successes = []
    for task_id in range(10):
        rows = {}
        task_description = None
        for path in sorted((args.output / "rollouts").glob(f"libero_10_{task_id}_*.json")):
            for episode in json.loads(path.read_text()).get("episodes", []):
                initialization = episode["initialization_id"]
                if initialization in rows:
                    raise RuntimeError(f"duplicate InternVLA episode task={task_id} init={initialization}")
                rows[initialization] = episode
                task_description = episode["task_description"]
        if set(rows) != set(range(50)):
            raise RuntimeError(f"InternVLA condition has {len(rows)}/50 episodes for task {task_id}")
        successes = [bool(rows[index]["success"]) for index in range(50)]
        overall_successes.extend(successes)
        per_task.append({
            "task_group": "libero_10", "task_id": task_id,
            "task_description": task_description,
            "metrics": {"successes": successes, "sum_rewards": [sum(rows[i]["rewards"]) for i in range(50)]},
            **success_summary(successes),
        })
    info = {
        "per_task": per_task,
        "per_group": {"libero_10": success_summary(overall_successes)},
        "overall": {
            **success_summary(overall_successes),
            "eval_s": time.time() - started,
            "eval_ep_s": (time.time() - started) / 500,
        },
    }
    write_json_atomic(args.output / "official/eval_info.json", info)
    print(
        f"Verified InternVLA {args.backend} {args.condition}: "
        f"{info['overall']['n_success']}/500 official LIBERO-10 episodes",
        flush=True,
    )


if __name__ == "__main__":
    main()

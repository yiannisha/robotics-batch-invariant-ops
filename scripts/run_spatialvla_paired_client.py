"""Run one resumable paired SpatialVLA condition in official SimplerEnv."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat

from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
ROOT = Path("/workspace/models/simplerenv-openvla")
TASK_MANIFEST = Path(
    "/workspace/robotics-batch-invariant-ops/results/cross_model/"
    "spatialvla/full/closed_loop/paired/evaluator_task_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def composed_quaternion(yaw: float = 0.0) -> list[float]:
    return list((Pose(q=euler2quat(0.0, 0.0, yaw)) * Pose(q=[0, 0, 0, 1])).q)


def drawer_config(initialization: int) -> dict[str, Any]:
    blocks = [
        ("frl_apartment_stage_simple", {}),
        ("modern_bedroom_no_roof", {}),
        ("modern_office_no_roof", {}),
        ("frl_apartment_stage_simple", {"light_mode": "brighter"}),
        ("frl_apartment_stage_simple", {"light_mode": "darker"}),
        ("frl_apartment_stage_simple", {"station_name": "mk_station2"}),
        ("frl_apartment_stage_simple", {"station_name": "mk_station3"}),
    ]
    values = []
    for scene, extra in blocks:
        for x in np.linspace(0.65, 0.85, 3):
            for y in np.linspace(-0.2, 0.2, 3):
                values.append({"scene_name": scene, "robot_x": float(x), "robot_y": float(y),
                               "additional": {"shader_dir": "rt", **extra}})
    if not 0 <= initialization < 50 or len(values) != 63:
        raise ValueError("drawer initialization is outside the pinned first-50-of-63 schedule")
    return values[initialization]


def coke_config(initialization: int) -> dict[str, Any]:
    values = []
    for option in ("lr_switch", "upright", "laid_vertically"):
        for x in np.linspace(-0.35, -0.12, 5):
            for y in np.linspace(-0.02, 0.42, 5):
                values.append({"obj_x": float(x), "obj_y": float(y), "additional": {option: True}})
    if not 0 <= initialization < 50 or len(values) != 75:
        raise ValueError("coke initialization is outside the pinned first-50-of-75 schedule")
    return values[initialization]


def task_configuration(task: dict[str, Any], initialization: int) -> dict[str, Any]:
    task_id = task["task_id"]
    overlay_root = ROOT / "ManiSkill2_real2sim/data/real_inpainting"
    if task_id <= 2:
        return {
            "robot": "widowx", "scene_name": "bridge_table_1_v1",
            "robot_x": 0.147, "robot_y": 0.028, "robot_quat": composed_quaternion(),
            "control_freq": 5, "sim_freq": 500, "max_steps": 60,
            "rgb_overlay_path": str(overlay_root / "bridge_real_eval_1.png"),
            "additional": {}, "obj_options": {"episode_id": initialization},
            "obs_camera_name": "3rd_view_camera",
        }
    if task_id == 3:
        return {
            "robot": "widowx_sink_camera_setup", "scene_name": "bridge_table_1_v2",
            "robot_x": 0.127, "robot_y": 0.06, "robot_quat": composed_quaternion(),
            "control_freq": 5, "sim_freq": 500, "max_steps": 120,
            "rgb_overlay_path": str(overlay_root / "bridge_sink.png"),
            "additional": {}, "obj_options": {"episode_id": initialization},
            "obs_camera_name": "3rd_view_camera",
        }
    if 4 <= task_id <= 9:
        value = drawer_config(initialization)
        return {
            "robot": "google_robot_static", "scene_name": value["scene_name"],
            "robot_x": value["robot_x"], "robot_y": value["robot_y"],
            "robot_quat": composed_quaternion(), "control_freq": 3,
            "sim_freq": 513, "max_steps": 113, "rgb_overlay_path": None,
            "additional": value["additional"], "obj_options": {"init_xy": np.array([0.0, 0.0])},
            "obs_camera_name": "overhead_camera",
        }
    if task_id == 10:
        return {
            "robot": "google_robot_static", "scene_name": "google_pick_coke_can_1_v4",
            "robot_x": 0.35, "robot_y": 0.21, "robot_quat": composed_quaternion(-0.09),
            "control_freq": 3, "sim_freq": 513, "max_steps": 80,
            "rgb_overlay_path": None, "additional": {},
            "obj_options": {"episode_id": initialization},
            "obs_camera_name": "overhead_camera",
        }
    if task_id == 11:
        value = coke_config(initialization)
        return {
            "robot": "google_robot_static", "scene_name": "google_pick_coke_can_1_v4",
            "robot_x": 0.35, "robot_y": 0.20, "robot_quat": composed_quaternion(),
            "control_freq": 3, "sim_freq": 513, "max_steps": 80,
            "rgb_overlay_path": None, "additional": value["additional"],
            "obj_options": {"init_xy": np.array([value["obj_x"], value["obj_y"]])},
            "obs_camera_name": "overhead_camera",
        }
    raise ValueError(f"unsupported SpatialVLA task ID {task_id}")


def make_environment(task: dict[str, Any], initialization: int, seed: int):
    config = task_configuration(task, initialization)
    control_mode = get_robot_control_mode(config["robot"], "spatialvla")
    kwargs = {
        "obs_mode": "rgbd", "robot": config["robot"], "sim_freq": config["sim_freq"],
        "control_mode": control_mode, "control_freq": config["control_freq"],
        "max_episode_steps": config["max_steps"], "scene_name": config["scene_name"],
        "camera_cfgs": {"add_segmentation": True},
        "rgb_overlay_path": config["rgb_overlay_path"], **config["additional"],
    }
    env = build_maniskill2_env(task["env_name"], **kwargs)
    options = {
        "robot_init_options": {
            "init_xy": np.array([config["robot_x"], config["robot_y"]]),
            "init_rot_quat": np.asarray(config["robot_quat"]),
        },
        "obj_init_options": config["obj_options"],
    }
    obs, _ = env.reset(seed=seed, options=options)
    return env, obs, config


def pose_state(obs: dict[str, Any]) -> tuple[str, np.ndarray]:
    """Return the robot pose field exposed by the active SimplerEnv robot.

    WidowX tasks expose the end-effector pose under ``extra.tcp_pose`` while
    the Google robot tasks expose the mobile base pose under
    ``agent.base_pose``.  The official 12-task suite contains both robot
    families, so treating ``tcp_pose`` as universal makes the campaign stop
    exactly at the family boundary.
    """
    extra = obs.get("extra")
    if isinstance(extra, dict) and "tcp_pose" in extra:
        return "tcp_pose", np.asarray(extra["tcp_pose"])
    agent = obs.get("agent")
    if isinstance(agent, dict) and "base_pose" in agent:
        return "base_pose", np.asarray(agent["base_pose"])
    raise KeyError("observation has neither extra.tcp_pose nor agent.base_pose")


def physical_hash(obs: dict[str, Any], image: np.ndarray) -> str:
    digest = hashlib.sha256()
    pose_key, pose = pose_state(obs)
    for key, value in (
        ("image", image), ("qpos", obs["agent"]["qpos"]),
        ("qvel", obs["agent"]["qvel"]), (pose_key, pose),
    ):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def state_record(obs: dict[str, Any]) -> dict[str, list[float]]:
    pose_key, pose = pose_state(obs)
    state = {
        "qpos": np.asarray(obs["agent"]["qpos"]).tolist(),
        "qvel": np.asarray(obs["agent"]["qvel"]).tolist(),
    }
    state[pose_key] = pose.tolist()
    return state


def success_summary(successes: list[bool]) -> dict[str, Any]:
    count, passed = len(successes), sum(successes)
    proportion = passed / count
    z = 1.959963984540054; z2 = z * z; denominator = 1 + z2 / count
    centre = (proportion + z2 / (2 * count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / count + z2 / (4 * count * count)) / denominator
    return {"n_episodes": count, "n_success": passed, "pc_success": 100 * proportion,
            "pc_success_ci95": [100 * max(0.0, centre - half), 100 * min(1.0, centre + half)]}


class PairedClient:
    def __init__(self, host: str, port: int, condition: str) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        # SpatialVLA generation can legitimately take longer than the transport's
        # 20-second default pong deadline while the synchronous server is inside
        # model.generate(). Keep pings enabled, but do not classify a long policy
        # call as a dead connection.
        self.client._ws.ping_timeout = 600
        self.metadata = self.client.get_server_metadata()
        required_batch = 1 if condition.endswith("singleton") else 4
        if (self.metadata.get("protocol_version") != "2.1"
                or self.metadata.get("policy_type") != "spatialvla"
                or self.metadata.get("task_suite") != "simplerenv_spatialvla_semantic_12_variant"
                or self.metadata.get("paired_condition") != condition
                or self.metadata.get("genuine_batched_policy_call") is not True
                or self.metadata.get("maximum_request_batch") != required_batch
                or self.metadata.get("deterministic_policy") is not True):
            raise RuntimeError(f"SpatialVLA server metadata differs: {self.metadata}")

    def request(self, rows: list[dict[str, Any]]) -> dict[int, dict[str, np.ndarray]]:
        examples = []
        for row in rows:
            examples.append({
                "image": row["image"], "task_description": row["task_description"],
                "policy_setup": row["policy_setup"], "reset": row["reset_policy"],
                "paired_request_id": f"task{row['task']}-init{row['initialization_id']}",
            })
            row["reset_policy"] = False
        response = self.client.predict_action({
            "type": "infer", "request_id": "|".join(e["paired_request_id"] for e in examples),
            "payload": {"examples": examples},
        })
        if not response.get("ok"):
            raise RuntimeError(f"SpatialVLA inference failed: {response}")
        data = response.get("data", {}); ids = [e["paired_request_id"] for e in examples]
        actions = np.asarray(data.get("actions")); selected = np.asarray(data.get("selected_raw_actions"))
        chunks = np.asarray(data.get("raw_action_chunks"))
        if (actions.shape != (len(rows), 7) or selected.shape != (len(rows), 7)
                or chunks.shape != (len(rows), 4, 7) or data.get("request_ids") != ids
                or data.get("physical_batch_size") != len(rows)):
            raise RuntimeError("SpatialVLA response differs from requested physical batch")
        return {row["initialization_id"]: {"action": actions[i], "selected": selected[i],
                                                  "chunk": chunks[i]} for i, row in enumerate(rows)}

    def close(self) -> None:
        self.client.close()


def cache_path(output: Path, task: int, ids: list[int]) -> Path:
    return output / "rollouts" / f"task_{task}_initializations_{ids[0]}_{ids[-1]}.json"


def run_group(
    client: PairedClient, task: dict[str, Any], ids: list[int], *, seed: int,
) -> list[dict[str, Any]]:
    environments, rows = [], {}
    try:
        for initialization in ids:
            reset_seed = seed + initialization
            env, obs, config = make_environment(task, initialization, reset_seed)
            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=config["obs_camera_name"])
            description = env.get_language_instruction()
            row = {
                "task": task["task_id"], "task_name": task["name"],
                "task_group": "simplerenv_spatialvla_semantic_12_variant",
                "task_description": description, "policy_setup": task["policy_setup"],
                "initialization_id": initialization, "reset_seed": reset_seed,
                "initial_observation_sha256": physical_hash(obs, image),
                "deterministic_policy": True, "success": False,
                "trajectory": [state_record(obs)], "action_sequence": [],
                "raw_action_sequence": [], "raw_action_chunks": [],
                "rewards": [], "terminated": [], "truncated": [],
                "policy_batch_sizes": [], "obs": obs, "image": image,
                "active": True, "reset_policy": True,
            }
            rows[initialization] = row; environments.append((initialization, env, config))
        env_by_id = {initialization: (env, config) for initialization, env, config in environments}
        max_steps = max(config["max_steps"] for _, _, config in environments)
        for _step in range(max_steps):
            active = [rows[index] for index in ids if rows[index]["active"]]
            if not active:
                break
            outputs = client.request(active)
            batch_size = len(active)
            for row in active:
                initialization = row["initialization_id"]; env, config = env_by_id[initialization]
                result = outputs[initialization]
                # The msgpack transport may expose a read-only zero-copy view.
                # SimplerEnv passes the rotation slice to SciPy, which requires
                # a writable buffer even though the numerical action is unchanged.
                action = np.array(result["action"], dtype=np.float64, copy=True)
                obs, reward, terminated, truncated, _info = env.step(action)
                row["action_sequence"].append(action.tolist())
                row["raw_action_sequence"].append(result["selected"].tolist())
                row["raw_action_chunks"].append(result["chunk"].tolist())
                row["rewards"].append(float(reward)); row["terminated"].append(bool(terminated))
                row["truncated"].append(bool(truncated)); row["policy_batch_sizes"].append(batch_size)
                row["trajectory"].append(state_record(obs)); row["success"] = row["success"] or bool(terminated)
                new_description = env.get_language_instruction()
                if new_description != row["task_description"]:
                    row["task_description"] = new_description; row["reset_policy"] = True
                row["obs"] = obs
                row["image"] = get_image_from_maniskill2_obs_dict(
                    env, obs, camera_name=config["obs_camera_name"],
                )
                if terminated or truncated:
                    row["active"] = False
        episodes = []
        for initialization in ids:
            row = rows[initialization]
            episodes.append({key: value for key, value in row.items()
                             if key not in {"obs", "image", "active", "reset_policy"}})
        return episodes
    finally:
        for _initialization, env, _config in environments:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--initialization-limit", type=int, default=None)
    args = parser.parse_args()
    if args.task_limit is not None and args.task_limit <= 0:
        raise ValueError("task-limit must be positive")
    if args.initialization_limit is not None and args.initialization_limit <= 0:
        raise ValueError("initialization-limit must be positive")
    manifest = json.loads(TASK_MANIFEST.read_text()); tasks = manifest["tasks"]
    if args.task_limit is not None:
        tasks = tasks[:args.task_limit]
    batch_size, repaired = CONDITIONS[args.condition]
    client = PairedClient(args.host, args.port, args.condition)
    metadata_path = args.output / "server_metadata.json"
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text()) != client.metadata:
            client.close()
            raise ValueError("SpatialVLA paired server metadata differs on resume")
    else:
        write_json_atomic(metadata_path, client.metadata)
    condition_manifest = {
        "schema_version": 1, "model": "spatialvla", "condition": args.condition,
        "batch_size": batch_size, "invariant_ops": repaired, "seed": args.seed,
        "task_suite": manifest["task_suite"], "task_manifest_path": str(TASK_MANIFEST),
        "task_manifest_sha256": sha256(TASK_MANIFEST), "task_count": len(tasks),
        "planned_initializations": sum(
            min(task["initializations"], args.initialization_limit)
            if args.initialization_limit is not None else task["initializations"]
            for task in tasks
        ),
        "deterministic_policy": True, "explicit_physical_configuration": True,
        "websocket_ping_timeout_s": 600,
        "client_source_sha256": sha256(Path(__file__)),
        "server_source_sha256": sha256(Path(__file__).with_name("run_spatialvla_paired_server.py")),
        "server_metadata_path": str(metadata_path.resolve()),
        "server_metadata_sha256": sha256(metadata_path),
    }
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != condition_manifest:
            client.close()
            raise ValueError("SpatialVLA paired resume provenance differs")
    else:
        write_json_atomic(manifest_path, condition_manifest)
    successes = []
    try:
        for task in tasks:
            count = task["initializations"]
            if args.initialization_limit is not None:
                count = min(count, args.initialization_limit)
            for start in range(0, count, batch_size):
                ids = list(range(start, min(start + batch_size, count)))
                path = cache_path(args.output, task["task_id"], ids)
                if path.is_file():
                    cached = json.loads(path.read_text())
                    if (cached.get("task_id") != task["task_id"]
                            or cached.get("explicit_initialization_ids") != ids):
                        raise RuntimeError(f"cached SpatialVLA rollout differs: {path}")
                    episodes = cached["episodes"]
                else:
                    episodes = run_group(client, task, ids, seed=args.seed)
                    write_json_atomic(path, {
                        "task_id": task["task_id"], "task_name": task["name"],
                        "explicit_initialization_ids": ids,
                        "seeds": [args.seed + value for value in ids], "episodes": episodes,
                    })
                successes.extend(bool(episode["success"]) for episode in episodes)
    finally:
        client.close()
    summary = success_summary(successes)
    write_json_atomic(args.output / "official" / "eval_info.json", {
        "schema_version": 1, "overall": summary,
        "task_suite": manifest["task_suite"], "condition": args.condition,
    })
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

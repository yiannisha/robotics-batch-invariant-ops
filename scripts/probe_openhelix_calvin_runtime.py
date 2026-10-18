"""Probe the pinned OpenHelix CALVIN evaluator with real resets and steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


CALVIN_REVISION = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_REVISION = "1431a46bd36bde5903fb6345e68b5ccc30def666"
OPENHELIX_REVISION = "d88867fa56977f401a7c89d62b9ebce87d8078f0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openhelix", type=Path, required=True)
    parser.add_argument("--calvin", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--asset-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if revision(args.openhelix) != OPENHELIX_REVISION:
        raise ValueError("OpenHelix source revision differs")
    if revision(args.calvin) != CALVIN_REVISION:
        raise ValueError("CALVIN source revision differs")
    if revision(args.calvin / "calvin_env") != CALVIN_ENV_REVISION:
        raise ValueError("CALVIN environment revision differs")

    assets = json.loads(args.asset_evidence.read_text())
    if assets.get("calvin_revision") != CALVIN_REVISION or assets.get("calvin_env_revision") != CALVIN_ENV_REVISION:
        raise ValueError("CALVIN asset evidence belongs to another source revision")
    for record in assets.get("files", []):
        path = args.dataset / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"CALVIN evaluator asset changed: {record['path']}")

    from online_evaluation_calvin.evaluate_utils import (
        get_env, get_env_state_for_initial_condition,
    )
    from online_evaluation_calvin.multistep_sequences import get_sequences

    sequences = get_sequences(2, num_workers=1)
    if len(sequences) != 2 or sequences[0][0] == sequences[1][0]:
        raise RuntimeError("official CALVIN sequence generator did not provide two reset identities")
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
    }
    env = get_env(
        args.dataset / "validation", obs_space=observation_space, show_gui=False
    )
    reset_records = []
    try:
        for reset_id, (initial_condition, task_sequence) in enumerate(sequences):
            requested_robot, requested_scene = get_env_state_for_initial_condition(initial_condition)
            env.reset(robot_obs=requested_robot, scene_obs=requested_scene)
            observation = env.get_obs()
            actual_robot = np.asarray(observation["robot_obs"])
            actual_scene = np.asarray(observation["scene_obs"])
            before_static = np.asarray(observation["rgb_obs"]["rgb_static"])
            before_gripper = np.asarray(observation["rgb_obs"]["rgb_gripper"])
            hold_action = (
                actual_robot[:3].copy(), actual_robot[3:6].copy(), actual_robot[[-1]].copy()
            )
            stepped, reward, done, info = env.step(hold_action)
            after_static = np.asarray(stepped["rgb_obs"]["rgb_static"])
            reset_records.append({
                "reset_id": reset_id,
                "initial_condition": initial_condition,
                "initial_condition_sha256": canonical_sha256(initial_condition),
                "task_sequence": list(task_sequence),
                "requested_robot_obs_sha256": array_sha256(requested_robot),
                "requested_scene_obs_sha256": array_sha256(requested_scene),
                "actual_robot_obs_sha256": array_sha256(actual_robot),
                "actual_scene_obs_sha256": array_sha256(actual_scene),
                "robot_reset_max_absolute_error": float(np.max(np.abs(actual_robot - requested_robot))),
                "scene_reset_max_absolute_error": float(np.max(np.abs(actual_scene - requested_scene))),
                "rgb_static_shape": list(before_static.shape),
                "rgb_gripper_shape": list(before_gripper.shape),
                "rgb_static_before_sha256": array_sha256(before_static),
                "rgb_gripper_before_sha256": array_sha256(before_gripper),
                "rgb_static_after_one_step_sha256": array_sha256(after_static),
                "step_reward": float(reward),
                "step_done": bool(done),
                "step_info_keys": sorted(info),
            })
    finally:
        env.close()
        env.ownsPhysicsClient = False

    if len({row["initial_condition_sha256"] for row in reset_records}) != 2:
        raise RuntimeError("CALVIN reset IDs collapsed")
    if len({row["actual_scene_obs_sha256"] for row in reset_records}) != 2:
        raise RuntimeError("CALVIN distinct resets produced the same scene state")
    if any(row["rgb_static_shape"] != [200, 200, 3] or row["rgb_gripper_shape"] != [84, 84, 3]
           for row in reset_records):
        raise RuntimeError("CALVIN policy camera shapes differ")

    evaluator = args.openhelix / "online_evaluation_calvin/evaluate_policy_lcb_pt_act_simple_asy10.py"
    utilities = args.openhelix / "online_evaluation_calvin/evaluate_utils.py"
    result = {
        "schema_version": 1,
        "status": "passed",
        "probe": "two_distinct_official_reset_identities_and_one_headless_step_each",
        "openhelix_revision": OPENHELIX_REVISION,
        "calvin_revision": CALVIN_REVISION,
        "calvin_env_revision": CALVIN_ENV_REVISION,
        "dataset_root": str(args.dataset.resolve()),
        "asset_evidence": str(args.asset_evidence.resolve()),
        "asset_evidence_sha256": sha256(args.asset_evidence),
        "evaluator": str(evaluator.resolve()),
        "evaluator_sha256": sha256(evaluator),
        "evaluate_utils_sha256": sha256(utilities),
        "compatibility_contract": {
            "unused_tactile_camera_removed": True,
            "policy_rgbd_cameras_retained": ["static", "gripper"],
            "physics_and_scene_configuration_unmodified": True,
            "pyhash_fallback": "standard FNV-1 32-bit over UTF-8 bytes",
        },
        "resets": reset_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "passed", "resets": len(reset_records),
        "distinct_scene_states": len({row["actual_scene_obs_sha256"] for row in reset_records}),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

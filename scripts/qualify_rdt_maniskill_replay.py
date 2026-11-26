"""Qualify fixed-action ManiSkill replay for RDT's released evaluator setup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from qualify_simpler_fixed_action_replay import digest


def canonical(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["PickCube-v1"])
    parser.add_argument("--initializations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--reward-mode", default="dense")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or args.initializations < 1 or args.steps < 1:
        parser.error("invalid replay schedule")
    args.output.mkdir(parents=True)
    import gymnasium as gym
    import mani_skill.envs
    import importlib.metadata
    runtime = {"scope": "fixed_action_simulator_replay_only", "model_evaluated": False,
               "reset_protocol": "fresh_scene_same_seed_exact_state_and_observation_checked",
               "environment": {"obs_mode": "rgb", "control_mode": "pd_joint_pos",
                               "render_mode": "rgb_array", "reward_mode": args.reward_mode,
                               "sim_backend": "cpu", "shader_pack": "default",
                               "continue_after_time_limit_until_success_or_400": True},
               "versions": {name: importlib.metadata.version(name) for name in ("mani_skill", "sapien", "torch")},
               "runner_sha256": digest(Path(__file__).read_text()),
               "arguments": {key: str(val) if isinstance(val, Path) else val for key,val in vars(args).items()}}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    blocks = []
    for task in args.tasks:
        for initialization in range(args.initializations):
            reference = None
            for repeat in range(args.repeats):
                seed = 20241201 + initialization
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                env = gym.make(task, obs_mode="rgb", control_mode="pd_joint_pos",
                               render_mode="rgb_array", reward_mode=args.reward_mode,
                               sim_backend="cpu", sensor_configs={"shader_pack": "default"},
                               human_render_camera_configs={"shader_pack": "default"},
                               viewer_camera_configs={"shader_pack": "default"})
                try:
                    obs, info = env.reset(seed=seed)
                    records = [{"observation_sha256": digest(canonical(obs)),
                                "sim_state_sha256": digest(canonical(env.unwrapped.get_state_dict())),
                                "policy_rgb_sha256": digest(canonical(env.render()))}]
                    joint_start = obs["agent"]["qpos"][0, :-2].detach().cpu().numpy().copy()
                    for step in range(args.steps):
                        action = np.concatenate([joint_start + 0.01*np.sin(step/13), [1. if (step//20)%2 else -1.]]).astype(np.float32)
                        obs, reward, terminated, truncated, info = env.step(action)
                        records.append({"step": step, "action": action.tolist(),
                                        "observation_sha256": digest(canonical(obs)),
                                        "sim_state_sha256": digest(canonical(env.unwrapped.get_state_dict())),
                                        "policy_rgb_sha256": digest(canonical(env.render())),
                                        "reward": float(reward), "terminated": bool(terminated),
                                        "truncated": bool(truncated), "info_sha256": digest(canonical(info))})
                        # The released evaluator continues past Gym's time limit
                        # until success (or its own 400-step horizon).
                        if (bool(terminated) or bool(truncated)) and bool(info["success"]):
                            break
                    with (args.output / "replays.jsonl").open("a") as stream:
                        stream.write(json.dumps({"task": task, "initialization": initialization,
                                                 "repeat": repeat, "steps": records}) + "\n")
                    if reference is not None and records != reference:
                        first = next((i for i,(a,b) in enumerate(zip(reference,records)) if a != b), min(len(reference),len(records)))
                        raise RuntimeError(f"ManiSkill replay differs: {task}/{initialization}/{repeat}, record={first}")
                    reference = records
                finally:
                    env.close()
            blocks.append({"task": task, "initialization": initialization, "repeats": args.repeats,
                           "steps": len(reference)-1, "exact": True})
            print(f"EXACT {task}/{initialization}: {len(reference)-1} x {args.repeats}", flush=True)
    (args.output / "summary.json").write_text(json.dumps({"status": "fixed_action_replay_verified",
        "model_evaluated": False, "blocks": blocks}, indent=2) + "\n")


if __name__ == "__main__":
    main()

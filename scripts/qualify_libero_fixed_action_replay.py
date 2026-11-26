"""Qualify fresh-scene LIBERO resets and exact observed fixed-action replay.

Use explicit init-state IDs in a newly constructed official environment per
repeat. No vector environment or policy is used; this is not success-rate data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from qualify_simpler_fixed_action_replay import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--initializations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=520)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or args.initializations < 1 or args.max_steps < 1:
        parser.error("invalid replay schedule")
    args.output.mkdir(parents=True)
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    import importlib.metadata
    import inspect
    runtime = {"scope": "fixed_action_simulator_replay_only", "model_evaluated": False,
               "reset_protocol": "fresh_scene_explicit_init_state_id_before_every_reset",
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "versions": {name: importlib.metadata.version(name) for name in ("lerobot", "hf-libero", "mujoco", "robosuite")},
               "wrapper_source_sha256": digest(inspect.getsource(LiberoEnv)),
               "runner_source_sha256": digest(Path(__file__).read_text())}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    blocks = []
    for task in args.tasks:
        for initialization in range(args.initializations):
            reference = None
            for repeat in range(args.repeats):
                seed = 20261108 + initialization
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                env = LiberoEnv(suite, task, args.suite, obs_type="pixels_agent_pos",
                                episode_index=initialization, n_envs=1)
                try:
                    env.init_state_id = initialization
                    obs, info = env.reset(seed=seed)
                    records = [{"observation_sha256": digest(obs),
                                "sim_state_sha256": digest(env._env.sim.get_state().flatten()),
                                "instruction": env.task_description}]
                    for step in range(args.max_steps):
                        action = np.array([0.01*np.sin(step/7), 0.01*np.cos(step/9),
                                           0.01*np.sin(step/11), 0., 0., 0.,
                                           -1. if (step//20) % 2 else 1.], dtype=np.float32)
                        obs, reward, terminated, truncated, info = env.step(action)
                        records.append({"step": step, "action": action.tolist(),
                                        "observation_sha256": digest(obs),
                                        "sim_state_sha256": digest(env._env.sim.get_state().flatten()),
                                        "reward": float(reward), "terminated": bool(terminated),
                                        "truncated": bool(truncated), "info_sha256": digest(info)})
                        if terminated or truncated:
                            break
                    with (args.output / "replays.jsonl").open("a") as stream:
                        stream.write(json.dumps({"task": task, "initialization": initialization,
                                                 "repeat": repeat, "steps": records}) + "\n")
                    if reference is not None and records != reference:
                        first = next((i for i, (a,b) in enumerate(zip(reference,records)) if a != b), min(len(reference),len(records)))
                        raise RuntimeError(f"inexact LIBERO replay: task={task}, initialization={initialization}, repeat={repeat}, record={first}")
                    reference = records
                finally:
                    env.close()
            blocks.append({"task": task, "initialization": initialization,
                           "repeats": args.repeats, "steps": len(reference)-1, "exact": True})
            print(f"EXACT {args.suite}/{task}/{initialization}: {len(reference)-1} x {args.repeats}", flush=True)
    (args.output / "summary.json").write_text(json.dumps({"status": "fixed_action_replay_verified",
        "model_evaluated": False, "blocks": blocks}, indent=2) + "\n")


if __name__ == "__main__":
    main()

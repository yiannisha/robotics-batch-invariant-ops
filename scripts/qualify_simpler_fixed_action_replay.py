"""Check exact observed SIMPLER replay using the pi-zero snapshot worker.

This is simulator qualification only, never a model success-rate measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def digest(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return {"dtype": str(item.dtype), "shape": list(item.shape),
                    "bytes": hashlib.sha256(item.tobytes(order="C")).hexdigest()}
        if isinstance(item, np.generic):
            return encode(np.asarray(item))
        if isinstance(item, dict):
            return {str(key): encode(val) for key, val in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [encode(val) for val in item]
        return item
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pizero-project", type=Path, required=True)
    parser.add_argument("--simpler-root", type=Path, required=True)
    parser.add_argument("--simpler-python", type=Path, required=True)
    parser.add_argument("--vulkan-icd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initializations", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", default=["pick_can", "move_near", "open_drawer", "close_drawer"])
    args = parser.parse_args()
    if args.repeats < 2 or args.initializations < 1 or args.max_steps < 1:
        parser.error("need >=2 repeats, >=1 initialization and >=1 step")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.pizero_project.resolve()))
    from experiments.simpler_support import make_simpler_env
    from experiments.prepare_replay import TASKS

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    runtime = {"scope": "fixed_action_simulator_replay_only",
               "model_evaluated": False, "arguments": vars(args).copy(),
               "source_sha256": {name: sha(args.pizero_project / "experiments" / name)
                                 for name in ("simpler_support.py", "simpler_worker.py")},
               "runner_sha256": sha(Path(__file__)),
               "simpler_revision": subprocess.check_output(["git", "-C", str(args.simpler_root), "rev-parse", "HEAD"], text=True).strip(),
               "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True)}
    runtime["arguments"] = {key: str(val) if isinstance(val, Path) else val for key, val in runtime["arguments"].items()}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    blocks = []
    for task in args.tasks:
        env = make_simpler_env(TASKS[task], args.simpler_python,
                              args.simpler_root, args.vulkan_icd)
        try:
            for initialization in range(args.initializations):
                env.reset(seed=20261108 + initialization,
                          options={"obj_init_options": {"episode_id": initialization}})
                snapshot = env.capture_snapshot()
                reference = None
                for repeat in range(args.repeats):
                    restored = env.restore_snapshot()
                    if not all(restored["checks"].values()):
                        raise RuntimeError("inexact restore")
                    records = [{"observation_sha256": digest(restored["observation"]),
                                "instruction": env.get_language_instruction()}]
                    for step in range(args.max_steps):
                        # Nonconstant open-loop control probes controller and physics state.
                        action = np.array([0.002 * np.sin(step / 7),
                                           0.002 * np.cos(step / 9),
                                           0.001 * np.sin(step / 11),
                                           0., 0., 0., -1. if (step // 20) % 2 else 1.], dtype=np.float64)
                        obs, reward, success, truncated, info = env.step(action)
                        records.append({"step": step, "action": action.tolist(),
                                        "observation_sha256": digest(obs),
                                        "reward": float(reward), "success": bool(success),
                                        "truncated": bool(truncated), "info": str(info),
                                        "instruction": env.get_language_instruction()})
                        if truncated:
                            break
                    record = {"task": task, "initialization": initialization,
                              "repeat": repeat, "snapshot": snapshot,
                              "restore_checks": restored["checks"], "steps": records}
                    with (args.output / "replays.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    if reference is not None and records != reference:
                        first = next((i for i, (a, b) in enumerate(zip(reference, records)) if a != b), min(len(reference), len(records)))
                        raise RuntimeError(f"fixed-action replay differs: {task}/{initialization}/{repeat}, record={first}")
                    reference = records
                blocks.append({"task": task, "initialization": initialization,
                               "repeats": args.repeats, "steps": len(reference) - 1,
                               "exact": True})
                print(f"EXACT {task}/{initialization}: {len(reference)-1} steps x {args.repeats}", flush=True)
        finally:
            env.close()
    summary = {"status": "fixed_action_replay_verified", "scope": runtime["scope"],
               "model_evaluated": False, "blocks": blocks,
               "replays_sha256": sha(args.output / "replays.jsonl")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

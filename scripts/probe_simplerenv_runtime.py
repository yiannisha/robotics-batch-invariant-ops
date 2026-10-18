"""Pin and probe the official SimplerEnv runtime used by SpatialVLA/CogACT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_REVISION = "ccfe3809766839a2fcfb7a3d3c9abff585189188"
SIMULATOR_REVISION = "cd45dd27dc6bb26d048cb6570cdab4e3f935cc37"
PROBE = r"""
import json
import numpy as np
import gymnasium
import sapien
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

env = simpler_env.make("google_robot_pick_coke_can")
obs, reset_info = env.reset(seed=20261005)
image = get_image_from_maniskill2_obs_dict(
    env.unwrapped, obs, camera_name="overhead_camera"
)
instruction = env.unwrapped.get_language_instruction()
action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
next_obs, reward, terminated, truncated, step_info = env.step(action)
next_image = get_image_from_maniskill2_obs_dict(
    env.unwrapped, next_obs, camera_name="overhead_camera"
)
print(json.dumps({
    "numpy": np.__version__,
    "gymnasium": gymnasium.__version__,
    "sapien": sapien.__version__,
    "environment": "google_robot_pick_coke_can",
    "reset_seed": 20261005,
    "instruction": instruction,
    "image_shape": list(image.shape),
    "image_dtype": str(image.dtype),
    "action_shape": list(action.shape),
    "reward": float(reward),
    "terminated": bool(terminated),
    "truncated": bool(truncated),
    "next_image_shape": list(next_image.shape),
}, sort_keys=True))
env.close()
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path,
        default=Path("/workspace/models/simplerenv-openvla"),
    )
    parser.add_argument(
        "--python", type=Path,
        default=Path("/workspace/envs/simpler_eval/bin/python"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/cross_model/closed_loop/simplerenv_runtime_probe.json"),
    )
    args = parser.parse_args()
    simulator = args.repository / "ManiSkill2_real2sim"
    if revision(args.repository) != REPOSITORY_REVISION:
        raise RuntimeError("SimplerEnv checkout differs from the pinned evaluator revision")
    if revision(simulator) != SIMULATOR_REVISION:
        raise RuntimeError("ManiSkill2 real-to-sim checkout differs from the pinned revision")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)

    environment = os.environ.copy()
    environment.update({
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
    })
    command = [str(args.python), "-c", PROBE]
    completed = subprocess.run(
        command, cwd=args.repository, env=environment,
        capture_output=True, text=True, check=False,
    )
    parsed = None
    if completed.returncode == 0:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        parsed = json.loads(lines[-1]) if lines else None
    sources = [
        args.repository / "simpler_env/main_inference.py",
        args.repository / "simpler_env/evaluation/maniskill2_evaluator.py",
        args.repository / "simpler_env/policies/spatialvla/spatialvla_model.py",
        args.repository / "simpler_env/policies/sim_cogact/cogact_policy.py",
        args.repository / "scripts/run_spatialvla.sh",
        args.repository / "scripts/run_cogact.sh",
    ]
    report = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": (
            "RUNNABLE_SIMULATOR_PROBE_NOT_POLICY_EPISODE"
            if completed.returncode == 0 else "SIMULATOR_PROBE_FAILED"
        ),
        "supports_model_paths": ["spatialvla", "cogact_small"],
        "repository": str(args.repository.resolve()),
        "repository_revision": REPOSITORY_REVISION,
        "simulator_revision": SIMULATOR_REVISION,
        "python": str(args.python.resolve()),
        "graphics_environment": {
            "VK_ICD_FILENAMES": environment["VK_ICD_FILENAMES"],
            "VK_DRIVER_FILES": environment["VK_DRIVER_FILES"],
        },
        "command": [str(args.python), "-c", "<embedded reset/render/step probe>"],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result": parsed,
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in sources
        ],
        "counts_as_closed_loop_episode": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "classification": report["classification"],
        "exit_code": report["exit_code"],
        "result": report["result"],
    }, indent=2))
    raise SystemExit(0 if completed.returncode == 0 else 2)


if __name__ == "__main__":
    main()

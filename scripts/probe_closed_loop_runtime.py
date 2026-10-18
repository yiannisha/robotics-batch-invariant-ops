"""Record whether the official RDT ManiSkill simulator can start on this host."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROBE = r"""
import gymnasium as gym
import mani_skill.envs
env = gym.make(
    "PickCube-v1",
    obs_mode="rgb",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    sim_backend="auto",
)
observation, info = env.reset(seed=20241201)
image = env.render()
print({"image_shape": tuple(image.shape), "image_dtype": str(image.dtype)})
env.close()
"""

MUJOCO_PROBE = r"""
import mujoco
xml = '<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.2"/><camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 .45 .89"/></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 64, 64)
renderer.update_scene(data, camera="cam")
image = renderer.render()
print({"image_shape": tuple(image.shape), "image_dtype": str(image.dtype)})
renderer.close()
"""


def command_output(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/workspace/envs/rdt/bin/python")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cross_model/closed_loop/rdt_maniskill_probe.json"),
    )
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["VK_DRIVER_FILES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    environment["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
    probe = subprocess.run(
        [args.python, "-c", PROBE],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    mujoco_environment = environment.copy()
    mujoco_environment["MUJOCO_GL"] = "egl"
    mujoco_probe = subprocess.run(
        [args.python, "-c", MUJOCO_PROBE],
        capture_output=True,
        text=True,
        check=False,
        env=mujoco_environment,
    )
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "official RDT ManiSkill closed-loop simulator startup gate",
        "python": args.python,
        "graphics_environment": {
            "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "VK_DRIVER_FILES": environment["VK_DRIVER_FILES"],
            "icd_exists": Path(environment["VK_DRIVER_FILES"]).is_file(),
            "dri_devices": sorted(str(path) for path in Path("/dev/dri").glob("*")),
        },
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "simulator_probe": {
            "command": [args.python, "-c", "<embedded probe>"],
            "returncode": probe.returncode,
            "stdout": probe.stdout,
            "stderr": probe.stderr,
            "started": probe.returncode == 0,
        },
        "mujoco_egl_probe": {
            "command": [args.python, "-c", "<embedded probe>"],
            "returncode": mujoco_probe.returncode,
            "stdout": mujoco_probe.stdout,
            "stderr": mujoco_probe.stderr,
            "started": mujoco_probe.returncode == 0,
        },
        "classification": (
            "RUNNABLE"
            if probe.returncode == 0 and mujoco_probe.returncode == 0
            else "BLOCKED: GPU graphics capability unavailable to container"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if probe.returncode == 0 and mujoco_probe.returncode == 0 else 2)


if __name__ == "__main__":
    main()

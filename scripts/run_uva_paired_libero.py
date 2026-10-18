"""Run resumable paired UVA action or joint video/action LIBERO-10 episodes.

This wrapper keeps the released UVA policy, preprocessing, 16-frame history,
absolute-action conversion, task suite, and 500-step horizon.  The only
stochasticity change is to make every VAE and diffusion draw an explicit
request-local input, as in the audited UVA adapter, so the same episode can be
compared across physical policy batch sizes without changing its random stream.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
UVA_ROOT = Path("/workspace/models/uva")
CHECKPOINT = Path("/workspace/assets/uva/libero10.ckpt")
CHECKPOINT_SHA256 = "4554b9bb2f4f3d44a7e51d5c7bd42a7740451874ac6cf1fea91e088a949f0dbf"
UVA_COMMIT = "fe3dec81af565fbb2d686072722f12340be31293"
OVERLAY = Path("/workspace/envs/uva_overlay")
LIBERO_PACKAGE = Path("/workspace/assets/uva/libero_package")
PYTORCH3D_COMPAT = Path("/workspace/assets/uva/pytorch3d_compat")
LIBERO_CONFIG = Path("/workspace/assets/uva/libero_config")
COMPLETE_LIBERO_ASSETS = Path("/root/.cache/libero/assets")

CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}
PINNED_PACKAGES = {
    "transformers": "4.28.0",
    "tokenizers": "0.13.3",
    "huggingface_hub": "0.25.2",
    "zarr": "2.16.1",
    "numcodecs": "0.11.0",
    "gym": "0.21.0",
    "kornia": "0.8.0",
}


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


def install_dependency_paths() -> None:
    # Order is significant: UVA's exact Python package overlay must precede
    # the shared evaluator environment, and its expected libero.libero layout
    # must precede the installed LeRobot LIBERO package.
    for path in reversed((OVERLAY, PYTORCH3D_COMPAT, LIBERO_PACKAGE, UVA_ROOT, ROOT)):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def install_runtime_compatibility() -> None:
    # Gym 0.21 predates Python 3.12's EntryPoints.select API.
    if not hasattr(importlib.metadata.EntryPoints, "get"):
        def get(self, key, default=None):
            values = list(self.select(group=key))
            return values if values else default
        importlib.metadata.EntryPoints.get = get

    # PyTorch 2.6 changed the default for torch.load. UVA's trusted, pinned
    # task init files predate that change and contain NumPy arrays.
    original_load = torch.load
    if not getattr(original_load, "_uva_compat", False):
        def compatible_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)
        compatible_load._uva_compat = True
        torch.load = compatible_load


def verify_inputs() -> dict[str, str]:
    required = (
        CHECKPOINT,
        UVA_ROOT / "eval_sim.py",
        UVA_ROOT / "unified_video_action/policy/unified_video_action_policy.py",
        UVA_ROOT / "unified_video_action/env_runner/libero_image_runner.py",
        UVA_ROOT / "unified_video_action/env/libero/benchmark/__init__.py",
        UVA_ROOT / "unified_video_action/env/libero/envs/env_wrapper.py",
        COMPLETE_LIBERO_ASSETS,
        LIBERO_CONFIG / "config.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"UVA paired evaluator inputs are missing: {missing}")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("UVA paired checkpoint differs from the pinned public checkpoint")
    commit = subprocess.check_output(
        ["git", "-C", str(UVA_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UVA_COMMIT:
        raise ValueError(f"UVA source commit {commit} != {UVA_COMMIT}")
    versions = {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}
    if versions != PINNED_PACKAGES:
        raise RuntimeError(f"UVA dependency overlay differs: {versions}")
    return versions


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


def image_from_observation(obs: dict[str, Any]) -> np.ndarray:
    # EnvRobosuite.get_observation performs this vertical flip before its
    # standard HWC->CHW and /255 processing. The audited adapter owns /255.
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1]
    return np.ascontiguousarray(np.moveaxis(image, -1, 0))


def robot_state(obs: dict[str, Any]) -> list[float]:
    from pytorch3d.transforms import matrix_to_axis_angle
    from unified_video_action.model.common.rotation_transformer import RotationTransformer

    quaternion = torch.as_tensor(obs["robot0_eef_quat"], dtype=torch.float64)
    # Robosuite observations are xyzw; PyTorch3D consumes wxyz.
    quaternion = quaternion[[3, 0, 1, 2]]
    from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
    axis_angle = matrix_to_axis_angle(quaternion_to_matrix(quaternion)).numpy()
    return np.concatenate((
        np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
        axis_angle,
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
    )).tolist()


def request_seed(seed: int, path: str, task: int, initialization: int, chunk: int) -> int:
    value = f"{seed}:{path}:{task}:{initialization}:{chunk}:uva".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63) - 1)


def request_input(adapter, *, history: deque[np.ndarray], task_text: str,
                  seed: int, path: str, task: int, initialization: int,
                  chunk: int) -> tuple[dict[str, torch.Tensor], str]:
    generator = torch.Generator(device="cpu").manual_seed(
        request_seed(seed, path, task, initialization, chunk)
    )
    images = torch.from_numpy(np.stack(tuple(history))).float().div(255)
    images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
    value = {
        # This is the exact resize_image_eval path used by policy.predict_action.
        "images": images.unsqueeze(0),
        **adapter._tokenize(task_text),
        "vae_noise": torch.randn(1, 4, 16, 16, 16, generator=generator),
        "action_initial_noise": torch.randn(1, 16, 10, generator=generator),
        "action_step_noise": torch.randn(
            1, adapter.num_diffusion_steps, 16, 10, generator=generator
        ),
        "context_index": torch.tensor([[initialization]], dtype=torch.int64),
    }
    if path == "joint":
        value["video_initial_noise"] = torch.randn(1, 1_024, 16, generator=generator)
        value["video_step_noise"] = torch.randn(
            1, adapter.num_video_diffusion_steps, 1_024, 16, generator=generator
        )
    digest = hashlib.sha256()
    for name in sorted(key for key in value if key.endswith("noise")):
        tensor = value[name].contiguous()
        digest.update(name.encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return value, digest.hexdigest()


def convert_action(rotation_transformer, action: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (10,):
        raise ValueError(f"UVA action has shape {value.shape}, expected (10,)")
    rotation = rotation_transformer.inverse(value[None, 3:9])[0]
    converted = np.concatenate((value[:3], rotation, value[-1:])).astype(np.float32)
    if converted.shape != (7,) or not np.isfinite(converted).all():
        raise RuntimeError("UVA absolute action conversion produced an invalid action")
    return converted


def make_environment(task_suite, task_id: int):
    import robosuite
    from libero.libero.envs import OffScreenRenderEnv

    original = robosuite.load_controller_config
    def absolute_controller(*args, **kwargs):
        config = original(*args, **kwargs)
        config["control_delta"] = False
        return config
    robosuite.load_controller_config = absolute_controller
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
            camera_heights=128,
            camera_widths=128,
        )
    finally:
        robosuite.load_controller_config = original
    env.env.hard_reset = False
    return env


def infer(adapter, policy, samples: list[dict[str, torch.Tensor]], *, joint: bool):
    batch = adapter.compose_batch(samples)
    output = adapter.run_model(policy, batch)
    if joint:
        actions = output["actions"]
        videos = output["video"]
        video_hashes = [
            hashlib.sha256(row.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for row in videos
        ]
    else:
        actions = output
        video_hashes = [None] * len(samples)
    actions = actions.detach().cpu().numpy()
    if actions.shape != (len(samples), 8, 10) or not np.isfinite(actions).all():
        raise RuntimeError(f"UVA policy returned invalid action tensor {actions.shape}")
    return actions, video_hashes


def cache_path(output: Path, task: int, initializations: list[int]) -> Path:
    return output / "rollouts" / (
        f"libero_10_{task}_initializations_{initializations[0]}_{initializations[-1]}.json"
    )


def load_cached(path: Path, task: int, initializations: list[int], seed: int) -> list[dict] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    expected_seeds = [seed + index for index in initializations]
    if (value.get("task_id") != task or value.get("seeds") != expected_seeds
            or value.get("explicit_init_state_ids") != initializations):
        raise RuntimeError(f"cached UVA rollout has another reset schedule: {path}")
    episodes = value.get("episodes")
    if (not isinstance(episodes, list) or len(episodes) != len(initializations)
            or [row.get("initialization_id") for row in episodes] != initializations):
        raise RuntimeError(f"cached UVA rollout is incomplete: {path}")
    return episodes


def rollout_group(*, adapter, policy, rotation_transformer, task_suite, task_id: int,
                  initial_states, initializations: list[int], seed: int,
                  path: str) -> list[dict]:
    environments: list[tuple[int, Any]] = []
    records: dict[int, dict[str, Any]] = {}
    task = task_suite.get_task(task_id)
    try:
        for initialization in initializations:
            env = make_environment(task_suite, task_id)
            env.seed(seed + initialization)
            env.reset()
            obs = env.set_init_state(initial_states[initialization])
            image = image_from_observation(obs)
            environments.append((initialization, env))
            records[initialization] = {
                "task": task_id,
                "task_group": "libero_10",
                "task_description": task.language,
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
                "prediction_video_sha256": [],
                "inference_s": [],
                "obs": obs,
                "history": deque([image] * 16, maxlen=16),
                "plan": None,
                "active": True,
                "chunk": 0,
            }

        env_by_id = dict(environments)
        for step in range(500):
            active = [records[index] for index in initializations if records[index]["active"]]
            if not active:
                break
            if step % 8 == 0:
                samples, noise_hashes = [], []
                for row in active:
                    sample, digest = request_input(
                        adapter,
                        history=row["history"],
                        task_text=row["task_description"],
                        seed=seed,
                        path=path,
                        task=task_id,
                        initialization=row["initialization_id"],
                        chunk=row["chunk"],
                    )
                    samples.append(sample)
                    noise_hashes.append(digest)
                started = time.time()
                actions, video_hashes = infer(adapter, policy, samples, joint=path == "joint")
                elapsed = time.time() - started
                for index, row in enumerate(active):
                    row["plan"] = actions[index]
                    row["policy_batch_sizes"].append(len(active))
                    row["prediction_video_sha256"].append(video_hashes[index])
                    row["inference_s"].append(elapsed)
                    if row["first_noise_sha256"] is None:
                        row["first_noise_sha256"] = noise_hashes[index]
                    row["chunk"] += 1

            for row in active:
                action = convert_action(rotation_transformer, row["plan"][step % 8])
                row["trajectory"].append(robot_state(row["obs"]))
                obs, reward, done, _ = env_by_id[row["initialization_id"]].step(action)
                success = bool(float(reward) == 1.0 or env_by_id[row["initialization_id"]].check_success())
                row["action_sequence"].append(action.tolist())
                row["rewards"].append(float(reward))
                row["done"].append(bool(done))
                row["obs"] = obs
                row["history"].append(image_from_observation(obs))
                if success or done:
                    row["success"] = success
                    row["active"] = False

        episodes = []
        for initialization in initializations:
            row = records[initialization]
            if row["first_noise_sha256"] is None:
                raise RuntimeError("UVA rollout never made a policy request")
            for key in ("obs", "history", "plan", "active", "chunk"):
                row.pop(key)
            if path != "joint":
                row.pop("prediction_video_sha256")
            episodes.append(row)
        return episodes
    finally:
        for _, env in environments:
            env.close()


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


def summarize(output: Path, task_ids: list[int], episodes_per_task: int, elapsed: float) -> dict:
    per_task, all_successes = [], []
    for task_id in task_ids:
        rows = {}
        task_description = None
        for path in sorted((output / "rollouts").glob(f"libero_10_{task_id}_*.json")):
            for episode in json.loads(path.read_text()).get("episodes", []):
                initialization = episode["initialization_id"]
                if initialization in rows:
                    raise RuntimeError(f"duplicate UVA episode task={task_id} init={initialization}")
                rows[initialization] = episode
                task_description = episode["task_description"]
        expected = set(range(episodes_per_task))
        if set(rows) != expected:
            raise RuntimeError(
                f"UVA condition has {len(rows)}/{episodes_per_task} episodes for task {task_id}"
            )
        successes = [bool(rows[index]["success"]) for index in range(episodes_per_task)]
        all_successes.extend(successes)
        per_task.append({
            "task_group": "libero_10",
            "task_id": task_id,
            "task_description": task_description,
            **success_summary(successes),
        })
    return {
        "per_task": per_task,
        "per_group": {"libero_10": success_summary(all_successes)},
        "overall": {
            **success_summary(all_successes),
            "eval_s": elapsed,
            "eval_ep_s": elapsed / len(all_successes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("action", "joint"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_task <= 50:
        raise ValueError("UVA episodes-per-task must be in [1, 50]")
    if len(set(args.task_ids)) != len(args.task_ids) or any(task not in range(10) for task in args.task_ids):
        raise ValueError("UVA task IDs must be unique values in [0, 9]")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise EnvironmentError("UVA paired evaluator requires MUJOCO_GL=egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG))
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    install_dependency_paths()
    install_runtime_compatibility()
    versions = verify_inputs()
    # The adapter captures these paths at construction time during import.
    os.environ["UVA_LIBERO_CHECKPOINT"] = str(CHECKPOINT)

    from batch_invariant_ops import set_batch_invariant_mode
    from libero.libero import benchmark
    import libero.libero.envs.bddl_base_domain as bddl_domain
    from scripts.model_invariance.adapters.uva_libero10 import adapter
    from unified_video_action.model.common.rotation_transformer import RotationTransformer

    # The vendored environment derives scene paths from this module global;
    # point it at the complete official asset cache used by all other LIBERO runs.
    bddl_domain.DIR_PATH = str(COMPLETE_LIBERO_ASSETS.parent / "envs")

    batch_size, repaired = CONDITIONS[args.condition]
    own_source = Path(__file__).resolve()
    adapter_source = ROOT / "scripts/model_invariance/adapters/uva_libero10.py"
    manifest = {
        "schema_version": 1,
        "model": "uva_joint" if args.path == "joint" else "uva",
        "path": args.path,
        "condition": args.condition,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "official_repo_commit": UVA_COMMIT,
        "official_evaluator_sha256": sha256(UVA_ROOT / "eval_sim.py"),
        "official_policy_sha256": sha256(
            UVA_ROOT / "unified_video_action/policy/unified_video_action_policy.py"
        ),
        "official_libero_runner_sha256": sha256(
            UVA_ROOT / "unified_video_action/env_runner/libero_image_runner.py"
        ),
        "official_libero_environment_sha256": sha256(
            UVA_ROOT / "unified_video_action/env/libero/envs/env_wrapper.py"
        ),
        "paired_adapter_sha256": sha256(adapter_source),
        "paired_wrapper_sha256": sha256(own_source),
        "rotation_conversions_sha256": sha256(
            PYTORCH3D_COMPAT / "pytorch3d/transforms/rotation_conversions.py"
        ),
        "dependency_versions": versions,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "episodes_per_task": args.episodes_per_task,
        "task_ids": args.task_ids,
        "seed": args.seed,
        "batch_size": batch_size,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "episode_length": 500,
        "n_obs_steps": 16,
        "n_action_steps": 8,
        "control_mode": "absolute",
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "genuine_batched_policy_call": True,
        "evidence_tier": (
            "full_paired_condition" if args.episodes_per_task == 50 and args.task_ids == list(range(10))
            else "checkpoint_matched_probe"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("UVA paired resume provenance differs")
    else:
        write_json_atomic(manifest_path, manifest)

    adapter.include_video_output = args.path == "joint"
    adapter.input_images_normalized = True
    adapter.task_mode = "full_dynamic_model" if args.path == "joint" else "policy_model"
    policy = adapter.load_model()
    rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
    task_suite = benchmark.get_benchmark_dict()["libero_10"]()

    started = time.time()
    mode = set_batch_invariant_mode() if repaired else contextlib.nullcontext()
    with mode, torch.inference_mode():
        for task_id in args.task_ids:
            initial_states = task_suite.get_task_init_states(task_id)
            if len(initial_states) < args.episodes_per_task:
                raise RuntimeError(f"UVA task {task_id} has only {len(initial_states)} init states")
            for start in range(0, args.episodes_per_task, batch_size):
                initializations = list(range(start, min(start + batch_size, args.episodes_per_task)))
                output_path = cache_path(args.output, task_id, initializations)
                if load_cached(output_path, task_id, initializations, args.seed) is not None:
                    continue
                episodes = rollout_group(
                    adapter=adapter,
                    policy=policy,
                    rotation_transformer=rotation_transformer,
                    task_suite=task_suite,
                    task_id=task_id,
                    initial_states=initial_states,
                    initializations=initializations,
                    seed=args.seed,
                    path=args.path,
                )
                write_json_atomic(output_path, {
                    "task_id": task_id,
                    "task_group": "libero_10",
                    "seeds": [args.seed + index for index in initializations],
                    "explicit_init_state_ids": initializations,
                    "episodes": episodes,
                })
                print(
                    f"UVA {args.path} {args.condition}: task={task_id} "
                    f"initializations={initializations} "
                    f"success={sum(row['success'] for row in episodes)}",
                    flush=True,
                )

    summary = summarize(
        args.output, args.task_ids, args.episodes_per_task, time.time() - started
    )
    write_json_atomic(args.output / "official/eval_info.json", summary)
    print(
        f"Verified UVA {args.path} {args.condition}: "
        f"{summary['overall']['n_success']}/{summary['overall']['n_episodes']} episodes",
        flush=True,
    )


if __name__ == "__main__":
    main()

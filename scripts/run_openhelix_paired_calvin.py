"""Run one resumable paired OpenHelix condition in official CALVIN ABC-D.

The official evaluator executes a 19-action prediction chunk for at most 60
policy calls per subtask.  Dynamic conditions cycle B={1,2,4,8}; non-target
requests come from the frozen 1,000-observation replay.  Target stochastic
inputs are keyed only by sequence/subtask/policy-call identity and are therefore
identical across all four paired conditions.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "native_singleton": (False, False),
    "native_dynamic": (False, True),
    "patched_singleton": (True, False),
    "patched_dynamic": (True, True),
}
DYNAMIC_BATCH_SCHEDULE = (1, 2, 4, 8)
TARGET_POSITION_SCHEDULE = ("first", "middle", "last")
EP_LEN = 60
EXECUTE_LEN = 20
NUM_SEQUENCES = 1000
BASE_SEED = 20261015
OPENHELIX_REVISION = "d88867fa56977f401a7c89d62b9ebce87d8078f0"
CALVIN_REVISION = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_REVISION = "1431a46bd36bde5903fb6345e68b5ccc30def666"
CHECKPOINT_REVISION = "1a6fabbe148af045ed0710d007ed24c36fa83a53"
POLICY_SHA256 = "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def gzip_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", compresslevel=6) as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    os.replace(temporary, path)


def npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_bundle_sha256(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def observation_sha256(observation: dict[str, Any]) -> str:
    return array_bundle_sha256({
        "rgb_static": np.asarray(observation["rgb_obs"]["rgb_static"]),
        "rgb_gripper": np.asarray(observation["rgb_obs"]["rgb_gripper"]),
        "depth_static": np.asarray(observation["depth_obs"]["depth_static"]),
        "depth_gripper": np.asarray(observation["depth_obs"]["depth_gripper"]),
        "robot_obs": np.asarray(observation["robot_obs"]),
        "scene_obs": np.asarray(observation["scene_obs"]),
    })


def request_seed(sequence_index: int, subtask_index: int, policy_call: int) -> int:
    payload = f"{BASE_SEED}:{sequence_index}:{subtask_index}:{policy_call}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (2**63 - 1)


def request_noise_sha256(sample: dict[str, Any]) -> str:
    return array_bundle_sha256({
        key: sample[key].detach().cpu().numpy()
        for key in ("noise", "position_step_noise", "rotation_step_noise")
    })


def target_position(batch_size: int, policy_call: int) -> int:
    label = TARGET_POSITION_SCHEDULE[policy_call % len(TARGET_POSITION_SCHEDULE)]
    if label == "first" or batch_size == 1:
        return 0
    if label == "middle":
        return batch_size // 2
    return batch_size - 1


def precision_controls() -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Match the released evaluator verbatim.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if revision(args.openhelix) != OPENHELIX_REVISION:
        raise ValueError("OpenHelix source revision differs")
    if revision(args.calvin) != CALVIN_REVISION:
        raise ValueError("CALVIN source revision differs")
    if revision(args.calvin / "calvin_env") != CALVIN_ENV_REVISION:
        raise ValueError("CALVIN environment source revision differs")
    policy = args.checkpoint / "policy.pth"
    if not policy.is_file() or sha256(policy) != POLICY_SHA256:
        raise ValueError("OpenHelix policy checkpoint differs")
    replay = json.loads(args.replay_manifest.read_text())
    if (len(replay.get("observations", [])) != 1000
            or replay.get("split_counts") != {"diagnostic": 200, "heldout": 800}
            or replay.get("unique_physical_observations") != 1000):
        raise ValueError("OpenHelix frozen companion replay differs")
    runtime = json.loads(args.runtime_probe.read_text())
    if runtime.get("status") != "passed" or len(runtime.get("resets", [])) != 2:
        raise ValueError("OpenHelix CALVIN runtime probe has not passed")
    assets = json.loads(args.evaluator_assets.read_text())
    for record in assets.get("files", []):
        path = args.dataset / record["path"]
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]):
            raise ValueError(f"OpenHelix CALVIN evaluator asset changed: {path}")
    return {
        "openhelix_revision": OPENHELIX_REVISION,
        "calvin_revision": CALVIN_REVISION,
        "calvin_env_revision": CALVIN_ENV_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "policy_path": str(policy.resolve()),
        "policy_sha256": POLICY_SHA256,
        "replay_manifest": str(args.replay_manifest.resolve()),
        "replay_manifest_sha256": sha256(args.replay_manifest),
        "runtime_probe": str(args.runtime_probe.resolve()),
        "runtime_probe_sha256": sha256(args.runtime_probe),
        "evaluator_assets": str(args.evaluator_assets.resolve()),
        "evaluator_assets_sha256": sha256(args.evaluator_assets),
    }


def make_manifest(args: argparse.Namespace, provenance: dict[str, Any]) -> dict[str, Any]:
    repaired, dynamic = CONDITIONS[args.condition]
    source = args.openhelix / "online_evaluation_calvin/evaluate_policy_lcb_pt_act_simple_asy10.py"
    return {
        "schema_version": 1,
        "model": "openhelix",
        "condition": args.condition,
        "invariant_ops": repaired,
        "execution_mode": "dynamic_batch" if dynamic else "serial_singleton",
        "dynamic_batch_schedule": list(DYNAMIC_BATCH_SCHEDULE) if dynamic else [1],
        "target_position_schedule": list(TARGET_POSITION_SCHEDULE) if dynamic else ["first"],
        "companion_source": "frozen_1000_observation_CALVIN_ABC_replay",
        "sequence_start": args.sequence_start,
        "sequence_count": args.sequence_count,
        "official_sequence_count": NUM_SEQUENCES,
        "policy_calls_per_subtask": args.max_policy_calls,
        "execute_len": EXECUTE_LEN,
        "paired_noise_seed": BASE_SEED,
        "request_local_noise": True,
        "official_evaluator_source": str(source.resolve()),
        "official_evaluator_sha256": sha256(source),
        "paired_wrapper_sha256": sha256(Path(__file__)),
        "precision_controls": precision_controls(),
        **provenance,
    }


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def companion_batch(adapter: Any, target: dict[str, Any], batch_size: int,
                    sequence_index: int, subtask_index: int, policy_call: int) -> tuple[list[Any], int]:
    if batch_size == 1:
        return [target], 0
    anchor = (sequence_index * 131 + subtask_index * 37 + policy_call * 17) % 1000
    adapter._load_replay_metadata()
    row = adapter.replay_manifest["observations"][anchor]
    companions = [
        adapter.load_replay_input(index, (policy_call + offset) % 3)
        for offset, index in enumerate(row["companion_replay_indices"][:batch_size - 1])
    ]
    position = target_position(batch_size, policy_call)
    samples = companions[:]
    samples.insert(position, target)
    return samples, position


def run_sequence(
    *, adapter: Any, models: Any, env: Any, task_oracle: Any,
    annotations: Any, initial_condition: dict[str, Any], tasks: list[str],
    sequence_index: int, dynamic: bool, repaired: bool, max_policy_calls: int,
    position_min: np.ndarray, position_max: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from batch_invariant_ops import set_batch_invariant_mode
    from online_evaluation_calvin.evaluate_utils import get_env_state_for_initial_condition

    random.seed(BASE_SEED + sequence_index)
    np.random.seed((BASE_SEED + sequence_index) % (2**32))
    torch.manual_seed(BASE_SEED + sequence_index)
    torch.cuda.manual_seed_all(BASE_SEED + sequence_index)
    robot, scene = get_env_state_for_initial_condition(initial_condition)
    env.reset(robot_obs=robot, scene_obs=scene)
    initial_observation = env.get_obs()
    initial_hash = observation_sha256(initial_observation)

    actions: list[np.ndarray] = []
    robot_states: list[np.ndarray] = []
    scene_states: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    subtask_ids: list[int] = []
    policy_call_ids: list[int] = []
    chunk_ids: list[int] = []
    batch_sizes: list[int] = []
    target_positions: list[int] = []
    request_noise_hashes: list[str] = []
    request_schedule: list[dict[str, Any]] = []
    tasks_completed = 0
    termination_reason = "sequence_complete"

    for subtask_index, subtask in enumerate(tasks):
        start_info = env.get_info()
        instruction = annotations[subtask][0]
        solved = False
        for policy_call in range(max_policy_calls):
            observation = env.get_obs()
            seed = request_seed(sequence_index, subtask_index, policy_call)
            target = adapter.load_live_calvin_input(observation, env, instruction, seed)
            noise_hash = request_noise_sha256(target)
            batch_size = (
                DYNAMIC_BATCH_SCHEDULE[policy_call % len(DYNAMIC_BATCH_SCHEDULE)]
                if dynamic else 1
            )
            samples, position = companion_batch(
                adapter, target, batch_size, sequence_index, subtask_index, policy_call
            )
            batch = adapter.compose_batch(samples)
            mode = set_batch_invariant_mode() if repaired else contextlib.nullcontext()
            with torch.inference_mode(), mode:
                output = adapter.run_model(models, batch)
            trajectory = output[position].detach().cpu().numpy()
            trajectory[:, :3] = np.clip(trajectory[:, :3], position_min, position_max)
            request_noise_hashes.append(noise_hash)
            request_schedule.append({
                "subtask_index": subtask_index,
                "policy_call": policy_call,
                "noise_seed": seed,
                "noise_sha256": noise_hash,
                "batch_size": batch_size,
                "target_position": position,
            })
            for chunk_index, action in enumerate(trajectory[:EXECUTE_LEN]):
                stepped, reward, done, current_info = env.step([
                    action[:3], action[3:6], action[[6]],
                ])
                actions.append(np.asarray(action, dtype=np.float32))
                robot_states.append(np.asarray(stepped["robot_obs"], dtype=np.float32))
                scene_states.append(np.asarray(stepped["scene_obs"], dtype=np.float32))
                rewards.append(float(reward))
                dones.append(bool(done))
                subtask_ids.append(subtask_index)
                policy_call_ids.append(policy_call)
                chunk_ids.append(chunk_index)
                batch_sizes.append(batch_size)
                target_positions.append(position)
                if task_oracle.get_task_info_for_set(start_info, current_info, {subtask}):
                    solved = True
                    break
            if solved:
                tasks_completed += 1
                break
        if not solved:
            termination_reason = f"subtask_policy_call_limit:{subtask}"
            break

    arrays = {
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, 7),
        "robot_states": np.asarray(robot_states, dtype=np.float32).reshape(-1, 15),
        "scene_states": np.asarray(scene_states, dtype=np.float32).reshape(-1, 24),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "subtask_indices": np.asarray(subtask_ids, dtype=np.int16),
        "policy_call_indices": np.asarray(policy_call_ids, dtype=np.int16),
        "action_chunk_indices": np.asarray(chunk_ids, dtype=np.int8),
        "batch_sizes": np.asarray(batch_sizes, dtype=np.int8),
        "target_positions": np.asarray(target_positions, dtype=np.int8),
    }
    record = {
        "schema_version": 1,
        "sequence_index": sequence_index,
        "initialization_id": sequence_index,
        "explicit_init_state_id": sequence_index,
        "initial_condition": initial_condition,
        "initial_condition_sha256": canonical_sha256(initial_condition),
        "task_sequence": list(tasks),
        "condition": next(
            name for name, value in CONDITIONS.items() if value == (repaired, dynamic)
        ),
        "initial_observation_sha256": initial_hash,
        "first_noise_sha256": request_noise_hashes[0],
        "request_noise_sha256": request_noise_hashes,
        "request_schedule": request_schedule,
        "request_schedule_sha256": canonical_sha256(request_schedule),
        "success": tasks_completed == len(tasks),
        "tasks_completed": tasks_completed,
        "sequence_length": len(tasks),
        "termination_reason": termination_reason,
        "policy_calls": len(request_schedule),
        "environment_steps": len(actions),
        "trajectory_arrays_sha256": array_bundle_sha256(arrays),
    }
    return record, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-start", type=int, default=0)
    parser.add_argument("--sequence-count", type=int, default=NUM_SEQUENCES)
    parser.add_argument("--max-policy-calls", type=int, default=EP_LEN)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--openhelix", type=Path, default=Path("/workspace/models/openhelix"))
    parser.add_argument("--calvin", type=Path, default=Path("/workspace/models/calvin"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/assets/openhelix/calvin_evaluator/task_ABC_D"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/assets/openhelix/checkpoint/prompt_tuning_aux"))
    parser.add_argument("--base", type=Path, default=Path("/workspace/assets/openhelix/llava_metadata"))
    parser.add_argument("--vision", type=Path, default=Path("/workspace/assets/openhelix/clip_metadata"))
    parser.add_argument("--replay-manifest", type=Path, default=Path("results/cross_model/openhelix/full/replay_manifest.json"))
    parser.add_argument("--runtime-probe", type=Path, default=Path("results/cross_model/openhelix/full/closed_loop/runtime_probe.json"))
    parser.add_argument("--evaluator-assets", type=Path, default=Path("results/cross_model/openhelix/full/closed_loop/evaluator_assets.json"))
    args = parser.parse_args()
    if (args.sequence_start < 0 or args.sequence_count < 1
            or args.sequence_start + args.sequence_count > NUM_SEQUENCES
            or args.max_policy_calls < 1 or args.max_policy_calls > EP_LEN):
        raise ValueError("invalid OpenHelix CALVIN sequence/policy-call range")

    sys.path.insert(0, str(args.openhelix.resolve()))
    os.environ.update({
        "OPENHELIX_SOURCE": str(args.openhelix.resolve()),
        "OPENHELIX_CHECKPOINT": str(args.checkpoint.resolve()),
        "OPENHELIX_BASE": str(args.base.resolve()),
        "OPENHELIX_VISION": str(args.vision.resolve()),
        "OPENHELIX_SAMPLE_DIR": str((args.dataset / "validation").resolve()),
        "OPENHELIX_REPLAY_MANIFEST": str(args.replay_manifest.resolve()),
        "OPENHELIX_STREAM_CHECKPOINT": "1",
        "OPENHELIX_STREAM_CACHE": "/workspace/assets/openhelix/stream_cache",
        "OPENHELIX_CHECKPOINT_RECEIPT": str(
            (ROOT / "results/cross_model/openhelix/full/checkpoint_stream_receipt.json").resolve()
        ),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    provenance = verify_inputs(args)
    from online_evaluation_calvin.multistep_sequences import get_sequences

    sequences = get_sequences(NUM_SEQUENCES, num_workers=1)
    if len(sequences) != NUM_SEQUENCES:
        raise RuntimeError("official CALVIN sequence generator cardinality differs")
    manifest = make_manifest(args, provenance)
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("OpenHelix paired resume manifest differs")
    else:
        json_atomic(manifest_path, manifest)
    if args.preflight_only:
        json_atomic(args.output / "preflight.json", {
            "schema_version": 1, "passed": True,
            "sequence_generator_count": len(sequences),
            "first_initial_condition_sha256": canonical_sha256(sequences[0][0]),
            "last_initial_condition_sha256": canonical_sha256(sequences[-1][0]),
            "condition_manifest_sha256": sha256(manifest_path),
        })
        print(json.dumps({"preflight_passed": True, "sequences": len(sequences)}))
        return

    from omegaconf import OmegaConf
    import hydra
    from online_evaluation_calvin.evaluate_utils import get_env
    from scripts.model_invariance.adapters.openhelix_calvin import adapter

    task_cfg = OmegaConf.load(
        args.calvin / "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(
        args.calvin / "calvin_models/conf/annotations/new_playtable_validation.yaml"
    )
    statistics = yaml.safe_load((args.dataset / "validation/statistics.yaml").read_text())
    position_min = np.asarray(statistics["act_min_bound"][:3], dtype=np.float32)
    position_max = np.asarray(statistics["act_max_bound"][:3], dtype=np.float32)
    environment = get_env(
        args.dataset / "validation",
        obs_space={
            "rgb_obs": ["rgb_static", "rgb_gripper"],
            "depth_obs": ["depth_static", "depth_gripper"],
        },
        show_gui=False,
    )
    models = adapter.load_model()
    repaired, dynamic = CONDITIONS[args.condition]
    try:
        for sequence_index in range(
            args.sequence_start, args.sequence_start + args.sequence_count
        ):
            metadata_path = args.output / "rollouts" / f"sequence_{sequence_index:04d}.json.gz"
            arrays_path = args.output / "rollouts" / f"sequence_{sequence_index:04d}.npz"
            if metadata_path.is_file() and arrays_path.is_file():
                existing = load_json_gz(metadata_path)
                if (existing.get("sequence_index") != sequence_index
                        or existing.get("condition") != args.condition
                        or existing.get("trajectory_npz_sha256") != sha256(arrays_path)):
                    raise ValueError(f"OpenHelix paired resume evidence differs: {metadata_path}")
                print(f"SKIP sequence {sequence_index}: complete", flush=True)
                continue
            if metadata_path.exists() or arrays_path.exists():
                raise RuntimeError(f"partial OpenHelix rollout requires inspection: {sequence_index}")
            initial_condition, tasks = sequences[sequence_index]
            record, arrays = run_sequence(
                adapter=adapter, models=models, env=environment,
                task_oracle=task_oracle, annotations=annotations,
                initial_condition=initial_condition, tasks=list(tasks),
                sequence_index=sequence_index, dynamic=dynamic, repaired=repaired,
                max_policy_calls=args.max_policy_calls,
                position_min=position_min, position_max=position_max,
            )
            npz_atomic(arrays_path, arrays)
            record["trajectory_npz_path"] = str(arrays_path.resolve())
            record["trajectory_npz_sha256"] = sha256(arrays_path)
            gzip_json_atomic(metadata_path, record)
            print(json.dumps({
                "condition": args.condition, "sequence": sequence_index,
                "tasks_completed": record["tasks_completed"],
                "steps": record["environment_steps"],
            }, sort_keys=True), flush=True)
    finally:
        environment.close()
        environment.ownsPhysicsClient = False


if __name__ == "__main__":
    main()

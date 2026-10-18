"""Run one pinned official SmolVLA/LIBERO paired condition.

The shared rollout hooks remain frozen for the active X-VLA campaign.  This
model-specific entry point pins SmolVLA's relative-action LIBERO-Long recipe
and verifies every public checkpoint tensor after policy dtype conversion.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts import run_lerobot_paired_libero as shared


def verify_checkpoint_loaded(policy, checkpoint_weight: Path, checkpoint_sha256: str) -> dict:
    state = policy.state_dict()
    checked = 0
    source_bytes = 0
    with safe_open(checkpoint_weight, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        if set(keys) != set(state):
            raise RuntimeError(
                "official SmolVLA policy/checkpoint key sets differ: "
                f"missing={sorted(set(keys) - set(state))[:5]}, "
                f"unexpected={sorted(set(state) - set(keys))[:5]}"
            )
        for key in keys:
            source = weights.get_tensor(key)
            loaded = state[key].detach().to("cpu")
            source_bytes += source.numel() * source.element_size()
            if source.shape != loaded.shape or not torch.equal(source.to(loaded.dtype), loaded):
                raise RuntimeError(f"official SmolVLA policy did not load pinned tensor {key}")
            checked += 1
    if checked == 0:
        raise RuntimeError("SmolVLA checkpoint contains no tensors")
    return {
        "schema_version": 1,
        "model_id": "smolvla",
        "checkpoint_sha256": checkpoint_sha256,
        "tensors_checked": checked,
        "source_bytes_checked": source_bytes,
        "exact_after_policy_dtype_conversion": True,
    }


def precision_controls() -> dict:
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(shared.CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-weight", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--smoke-episode-length", type=int)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--source", type=Path, default=Path("/workspace/models/lerobot"))
    args = parser.parse_args()
    if (args.episodes_per_task < 1 or args.seed < 0 or not args.task_ids
            or len(set(args.task_ids)) != len(args.task_ids)
            or any(task not in range(10) for task in args.task_ids)):
        raise ValueError("invalid paired episode count or seed")
    if args.smoke_episode_length is not None and (
        args.smoke_episode_length < 1
        or args.episodes_per_task != shared.ENVIRONMENT_BATCH_SIZE
        or args.task_ids == list(range(10))
    ):
        raise ValueError("short smoke horizons require four episodes on a strict task subset")
    if not args.checkpoint_weight.is_file() or shared.sha256(args.checkpoint_weight) != args.checkpoint_sha256:
        raise ValueError("SmolVLA checkpoint differs from pinned weight hash")
    if not os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("MUJOCO_GL") != "egl":
        raise EnvironmentError("paired LIBERO evaluator requires LIBERO_CONFIG_PATH and MUJOCO_GL=egl")

    evaluator_source = args.source / "src/lerobot/scripts/lerobot_eval.py"
    policy_source = args.source / "src/lerobot/policies/smolvla/modeling_smolvla.py"
    if not evaluator_source.is_file() or not policy_source.is_file():
        raise FileNotFoundError("official LeRobot evaluator or SmolVLA source is missing")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    policy_batch_size, repaired = shared.CONDITIONS[args.condition]
    environment_batch_size = shared.ENVIRONMENT_BATCH_SIZE
    execution_mode = "serial_singleton" if policy_batch_size == 1 else "dynamic_batch"
    episode_length = args.smoke_episode_length or 520
    manifest = {
        "schema_version": 2,
        "paired_protocol_version": shared.PAIRED_PROTOCOL_VERSION,
        "model": "smolvla", "condition": args.condition,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight": str(args.checkpoint_weight.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "official_evaluator_sha256": shared.sha256(evaluator_source),
        "official_policy_sha256": shared.sha256(policy_source),
        "paired_wrapper_sha256": shared.sha256(Path(__file__)),
        "shared_rollout_wrapper_sha256": shared.sha256(Path(shared.__file__)),
        "episodes_per_task": args.episodes_per_task, "seed": args.seed,
        "batch_size": policy_batch_size,
        "policy_batch_size": policy_batch_size,
        "environment_batch_size": environment_batch_size,
        "execution_mode": execution_mode,
        "invariant_ops": repaired,
        "task_suite": "libero_10", "control_mode": "relative", "episode_length": episode_length,
        "smoke_test": args.smoke_episode_length is not None,
        "task_ids": args.task_ids,
        "explicit_init_state_by_initialization_id": True,
        "smolvla_canonical_prompt_padding": repaired,
        "precision_controls": precision_controls(),
    }
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("SmolVLA paired resume provenance differs from existing output")
    else:
        shared.write_json_atomic(manifest_path, manifest)

    sys.path.insert(0, str((args.source / "src").resolve()))
    from lerobot.scripts import lerobot_eval

    original_make_policy = lerobot_eval.make_policy

    def make_verified_policy(*positional, **keyword):
        policy = original_make_policy(*positional, **keyword)
        report = verify_checkpoint_loaded(policy, args.checkpoint_weight, args.checkpoint_sha256)
        shared.write_json_atomic(args.output / "checkpoint_load_verification.json", report)
        return policy

    lerobot_eval.make_policy = make_verified_policy
    shared.install_evaluator_hooks(
        lerobot_eval, model_name="smolvla", repaired=repaired,
        output=args.output, seed=args.seed,
        policy_batch_size=policy_batch_size,
        environment_batch_size=environment_batch_size,
    )
    sys.argv = [
        str(Path(__file__)),
        f"--policy.path={args.checkpoint.resolve()}",
        "--env.type=libero", "--env.task=libero_10",
        f"--env.task_ids={json.dumps(args.task_ids, separators=(',', ':'))}",
        "--env.control_mode=relative", f"--env.episode_length={episode_length}",
        f"--eval.batch_size={environment_batch_size}",
        f"--eval.n_episodes={args.episodes_per_task}",
        "--env.max_parallel_tasks=1",
        f"--seed={args.seed}",
        f"--output_dir={(args.output / 'official').resolve()}",
    ]
    mode = set_batch_invariant_mode() if repaired else contextlib.nullcontext()
    with mode:
        lerobot_eval.eval_main()


if __name__ == "__main__":
    main()

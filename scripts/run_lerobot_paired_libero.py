"""Run one resumable, checkpoint-matched LeRobot LIBERO paired condition.

The official evaluator still builds the environment, preprocessors, policy,
rollouts, and success metrics. These narrow hooks give each reset seed its own
noise stream, preserve raw state/action trajectories, and cache whole rollouts
for safe per-rollout resume. Four conditions must be run separately and then
compared by task and reset seed; one invocation is never paired completion.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode


CONDITIONS = {
    "native_singleton": (1, False),
    "native_dynamic": (4, False),
    "patched_singleton": (1, True),
    "patched_dynamic": (4, True),
}

# The simulator batch must be identical in every paired condition.  LIBERO's
# renderer is not invariant to the physical vector slot: constructing B=1 and
# B=4 environments can yield different camera pixels for the same task/init
# state.  Singleton therefore means four independent serial B=1 policy calls
# over one B=4 environment step, not a B=1 vector environment.
ENVIRONMENT_BATCH_SIZE = 4
PAIRED_PROTOCOL_VERSION = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def save_rollout_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sensor_hash(batch: dict, index: int) -> str:
    """Hash only physical observations, unaffected by prompt padding."""
    digest = hashlib.sha256()
    matched = False
    for key in sorted(batch):
        if key != "observation.state" and not key.startswith("observation.images."):
            continue
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        matched = True
        sample = value[index].detach().to("cpu").contiguous()
        digest.update(key.encode())
        digest.update(str(sample.dtype).encode())
        digest.update(json.dumps(list(sample.shape)).encode())
        digest.update(sample.numpy().tobytes())
    if not matched:
        raise RuntimeError("preprocessed LIBERO policy input contains no physical sensors")
    return digest.hexdigest()


def raw_sensor_hash(observation: Any, index: int) -> str:
    """Hash every raw array field for one physical vector-environment slot."""
    digest = hashlib.sha256()
    matched = False

    def visit(value: Any, prefix: str) -> None:
        nonlocal matched
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{prefix}/{key}")
            return
        if isinstance(value, np.ndarray):
            if value.ndim < 1 or value.shape[0] <= index:
                raise RuntimeError(f"raw observation field {prefix} lacks vector slot {index}")
            sample = np.ascontiguousarray(value[index])
        elif isinstance(value, torch.Tensor):
            if value.ndim < 1 or value.shape[0] <= index:
                raise RuntimeError(f"raw observation field {prefix} lacks vector slot {index}")
            sample = value[index].detach().to("cpu").contiguous().numpy()
        else:
            return
        matched = True
        digest.update(prefix.encode())
        digest.update(str(sample.dtype).encode())
        digest.update(json.dumps(list(sample.shape)).encode())
        digest.update(sample.tobytes())

    visit(observation, "")
    if not matched:
        raise RuntimeError("raw LIBERO reset contains no array-valued observation fields")
    return digest.hexdigest()


def slice_policy_batch(value: Any, index: int, batch_size: int) -> Any:
    """Extract one request while preserving a leading batch dimension."""
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, dict):
        return {key: slice_policy_batch(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        return [value[index]]
    if isinstance(value, tuple) and len(value) == batch_size:
        return (value[index],)
    return value


def _policy_queue_state(policy: Any) -> dict[str, Any]:
    names = ("_queues", "_action_queue")
    state = {name: getattr(policy, name) for name in names if hasattr(policy, name)}
    if not state:
        raise RuntimeError("serial singleton policy exposes no action-queue state")
    return state


def _restore_policy_queue_state(policy: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        setattr(policy, name, value)


def attach_serial_singleton_execution(policy: Any, environment_batch_size: int) -> None:
    """Run one B=1 model request per environment while retaining a B=4 world."""
    if environment_batch_size < 2:
        raise ValueError("serial singleton execution requires a multi-environment physical batch")
    original_select = policy.select_action
    original_reset = policy.reset

    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        template = _policy_queue_state(policy)
        policy._paired_request_queue_states = [
            copy.deepcopy(template) for _ in range(environment_batch_size)
        ]
        _restore_policy_queue_state(policy, policy._paired_request_queue_states[0])
        policy._paired_active_request_index = None
        return result

    def select_action(batch, *args, **kwargs):
        state = batch.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.shape[0] != environment_batch_size:
            raise RuntimeError("serial singleton received a policy batch unlike the physical B=4 batch")
        request_states = getattr(policy, "_paired_request_queue_states", None)
        if not isinstance(request_states, list) or len(request_states) != environment_batch_size:
            raise RuntimeError("serial singleton policy was not reset before rollout")
        outputs = []
        try:
            for index in range(environment_batch_size):
                _restore_policy_queue_state(policy, request_states[index])
                policy._paired_active_request_index = index
                output = original_select(
                    slice_policy_batch(batch, index, environment_batch_size), *args, **kwargs
                )
                if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
                    raise RuntimeError("serial singleton policy did not return one batched action")
                request_states[index] = _policy_queue_state(policy)
                outputs.append(output)
        finally:
            policy._paired_active_request_index = None
            _restore_policy_queue_state(policy, request_states[0])
        return torch.cat(outputs, dim=0)

    policy.reset = reset
    policy.select_action = select_action
    # Policy construction calls reset before hooks are attached; initialize the
    # request-local state now as well as on every official rollout reset.
    reset()


def attach_policy_observation_recorder(policy: Any, environment_batch_size: int) -> None:
    """Record B=4 processed sensors/state once, outside any serial policy calls."""
    original_select = policy.select_action

    def select_action(batch, *args, **kwargs):
        state = batch.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.shape[0] != environment_batch_size:
            raise RuntimeError("paired LIBERO rollout lacks the fixed physical policy batch")
        if not policy._paired_first_sensor_hashes:
            policy._paired_first_sensor_hashes = [
                sensor_hash(batch, index) for index in range(environment_batch_size)
            ]
        policy._paired_state_trajectory.append(state.detach().to("cpu").tolist())
        return original_select(batch, *args, **kwargs)

    policy.select_action = select_action


def set_explicit_init_state_ids(env, seeds: list[int], base_seed: int) -> list[int]:
    """Make LIBERO physical initialization independent of vector batch size.

    LiberoEnv increments ``init_state_id`` on every reset, including Gym's
    terminal auto-reset. Without this assignment, the next manual rollout
    selects a different task init state for B1 and B4 even with identical
    requested reset seeds.
    """
    if len(seeds) != env.num_envs:
        raise RuntimeError("paired reset seed count differs from physical vector batch")
    identifiers = [int(reset_seed) - base_seed for reset_seed in seeds]
    if any(identifier < 0 for identifier in identifiers):
        raise RuntimeError("paired reset seed precedes the campaign base seed")
    # Gymnasium vector envs expose set_attr directly, but LeRobot wraps its
    # AsyncVectorEnv in _LazyAsyncVectorEnv and (as of the pinned revision)
    # forwards get_attr without forwarding set_attr.  Materialize that wrapper
    # and use the underlying public AsyncVectorEnv API when necessary.
    setter = getattr(env, "set_attr", None)
    if setter is None:
        ensure = getattr(env, "_ensure", None)
        if not callable(ensure):
            raise RuntimeError("LIBERO vector env does not expose set_attr or a lazy-env initializer")
        ensure()
        inner = getattr(env, "_env", None)
        setter = getattr(inner, "set_attr", None)
        if not callable(setter):
            raise RuntimeError("LIBERO lazy vector env did not materialize a set_attr-capable env")
    setter("init_state_id", identifiers)
    observed = list(env.get_attr("init_state_id"))
    if observed != identifiers:
        raise RuntimeError(
            f"LIBERO vector env did not accept explicit init-state IDs: {observed} != {identifiers}"
        )
    return identifiers


def attach_policy_hooks(policy, model_name: str, repaired: bool) -> None:
    def next_noise(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        active = getattr(policy, "_paired_active_request_index", None)
        if active is None:
            if len(policy._paired_generators) != shape[0]:
                raise RuntimeError("per-request noise generators differ from rollout batch")
            indices = list(range(shape[0]))
        else:
            if shape[0] != 1 or active not in range(len(policy._paired_generators)):
                raise RuntimeError("serial singleton noise request has invalid request identity")
            indices = [active]
        values = [
            torch.randn(
                shape[1:], generator=policy._paired_generators[index], device=device, dtype=dtype
            )
            for index in indices
        ]
        if len(policy._paired_first_noise_hashes) != len(policy._paired_generators):
            raise RuntimeError("paired first-noise evidence was not initialized for every request")
        for index, value in zip(indices, values):
            if policy._paired_first_noise_hashes[index] is None:
                policy._paired_first_noise_hashes[index] = hashlib.sha256(
                    value.detach().to("cpu").contiguous().numpy().tobytes()
                ).hexdigest()
        return torch.stack(values)

    if model_name == "smolvla":
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        original_chunk = policy._get_action_chunk
        pad_token = policy.model.vlm_with_expert.processor.tokenizer.pad_token_id
        if pad_token is None:
            raise RuntimeError("pinned SmolVLA tokenizer lacks a pad token")
        canonical_length = policy.config.tokenizer_max_length

        def action_chunk(batch, noise=None, **kwargs):
            if repaired:
                tokens = batch[OBS_LANGUAGE_TOKENS]
                mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
                remaining = canonical_length - tokens.shape[1]
                if remaining < 0:
                    raise RuntimeError("SmolVLA prompt exceeds canonical repaired length")
                if remaining:
                    batch = dict(batch)
                    batch[OBS_LANGUAGE_TOKENS] = torch.nn.functional.pad(tokens, (0, remaining), value=pad_token)
                    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.nn.functional.pad(mask, (0, remaining), value=0)
            if noise is not None:
                raise RuntimeError("official evaluator supplied unexpected SmolVLA noise")
            state = batch["observation.state"]
            noise = next_noise(
                (state.shape[0], policy.model.config.chunk_size, policy.model.config.max_action_dim),
                device=state.device, dtype=torch.float32,
            )
            return original_chunk(batch, noise=noise, **kwargs)

        policy._get_action_chunk = action_chunk
    elif model_name == "xvla":
        model = policy.model

        def generate_actions(input_ids, image_input, image_mask, domain_id, proprio, steps):
            # The released method ignores XVLAPolicy.select_action(noise=...),
            # so this is its exact inference body with only x1 made request-local.
            model.eval()
            target_dtype = model._get_target_dtype()
            image_input = image_input.to(dtype=target_dtype)
            proprio = proprio.to(dtype=target_dtype)
            encoded = model.forward_vlm(input_ids, image_input, image_mask)
            batch_size = input_ids.shape[0]
            x1 = next_noise(
                (batch_size, model.chunk_size, model.dim_action),
                device=proprio.device, dtype=target_dtype,
            )
            action = torch.zeros_like(x1)
            steps = max(1, int(steps))
            for step in range(steps, 0, -1):
                t = torch.full((batch_size,), step / steps, device=proprio.device, dtype=target_dtype)
                x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
                proprio_m, x_t_m = model.action_space.preprocess(proprio, x_t)
                action = model.transformer(
                    domain_id=domain_id,
                    action_with_noise=x_t_m,
                    proprio=proprio_m,
                    t=t,
                    **encoded,
                )
            return model.action_space.postprocess(action)

        model.generate_actions = generate_actions
    else:
        raise ValueError(model_name)


def install_evaluator_hooks(
    module,
    *,
    model_name: str,
    repaired: bool,
    output: Path,
    seed: int,
    policy_batch_size: int,
    environment_batch_size: int = ENVIRONMENT_BATCH_SIZE,
) -> None:
    original_make_policy = module.make_policy
    original_run_one = module.run_one
    original_rollout = module.rollout
    original_eval_all = module.eval_policy_all
    current_task: dict[str, int | str | None] = {"id": None, "group": None}

    def make_policy(*args, **kwargs):
        policy = original_make_policy(*args, **kwargs)
        attach_policy_hooks(policy, model_name, repaired)
        if policy_batch_size == 1:
            attach_serial_singleton_execution(policy, environment_batch_size)
        elif policy_batch_size != environment_batch_size:
            raise RuntimeError("dynamic policy batch differs from the fixed environment batch")
        attach_policy_observation_recorder(policy, environment_batch_size)
        return policy

    def run_one(task_group, task_id, env, **kwargs):
        current_task.update({"id": task_id, "group": task_group})
        try:
            return original_run_one(task_group, task_id, env, **kwargs)
        finally:
            current_task.update({"id": None, "group": None})

    def rollout(*args, **kwargs):
        policy = kwargs.get("policy")
        seeds = kwargs.get("seeds")
        task_id = current_task["id"]
        task_group = current_task["group"]
        if policy is None or not seeds or task_id is None or task_group is None:
            raise RuntimeError("paired rollout requires policy, task ID, and explicit reset seeds")
        seeds = list(seeds)
        stem = f"{task_group}_{task_id}_seeds_{seeds[0]}_{seeds[-1]}"
        rollout_path = output / "rollouts" / f"{stem}.pt"
        evidence_path = output / "rollouts" / f"{stem}.json"
        if rollout_path.is_file() and evidence_path.is_file():
            evidence = json.loads(evidence_path.read_text())
            if evidence.get("task_id") != task_id or evidence.get("seeds") != seeds:
                raise RuntimeError("cached paired rollout has another task or reset schedule")
            return torch.load(rollout_path, map_location="cpu", weights_only=True)

        env = kwargs.get("env", args[0] if args else None)
        if env is None:
            raise RuntimeError("paired rollout lacks the official vector environment")
        if env.num_envs != environment_batch_size or len(seeds) != environment_batch_size:
            raise RuntimeError("paired rollout did not use the fixed B=4 physical environment")
        init_state_ids = set_explicit_init_state_ids(env, seeds, seed)
        policy._paired_generators = [
            torch.Generator(device="cuda").manual_seed(int(value)) for value in seeds
        ]
        policy._paired_first_sensor_hashes = []
        deterministic_contract = getattr(policy, "_paired_deterministic_contract_sha256", None)
        policy._paired_first_noise_hashes = (
            [deterministic_contract] * len(seeds)
            if deterministic_contract is not None else [None] * len(seeds)
        )
        policy._paired_first_raw_sensor_hashes = []
        policy._paired_state_trajectory = []
        original_reset = env.reset

        def reset_with_raw_evidence(*reset_args, **reset_kwargs):
            observation, info = original_reset(*reset_args, **reset_kwargs)
            policy._paired_first_raw_sensor_hashes = [
                raw_sensor_hash(observation, index) for index in range(environment_batch_size)
            ]
            return observation, info

        env.reset = reset_with_raw_evidence
        try:
            result = original_rollout(*args, **kwargs)
        finally:
            env.reset = original_reset
        actions = result[module.ACTION].detach().to("cpu")
        dones = result["done"].detach().to("cpu")
        rewards = result["reward"].detach().to("cpu")
        successes = result["success"].detach().to("cpu")
        if (len(policy._paired_first_sensor_hashes) != len(seeds)
                or len(policy._paired_first_raw_sensor_hashes) != len(seeds)
                or len(policy._paired_first_noise_hashes) != len(seeds)
                or any(value is None for value in policy._paired_first_noise_hashes)):
            raise RuntimeError("paired rollout did not record every request-local reset/noise")
        episodes = []
        for index, reset_seed in enumerate(seeds):
            done_positions = torch.nonzero(dones[index], as_tuple=False)
            length = int(done_positions[0, 0]) + 1 if len(done_positions) else actions.shape[1]
            states = [step[index] for step in policy._paired_state_trajectory[:length]]
            episodes.append({
                "task": task_id,
                "task_group": task_group,
                "initialization_id": int(reset_seed) - seed,
                "reset_seed": int(reset_seed),
                "initial_raw_observation_sha256": policy._paired_first_raw_sensor_hashes[index],
                "initial_observation_sha256": policy._paired_first_sensor_hashes[index],
                "first_noise_sha256": policy._paired_first_noise_hashes[index],
                "success": bool(successes[index, :length].any()),
                "trajectory": states,
                "action_sequence": actions[index, :length].tolist(),
                "rewards": rewards[index, :length].tolist(),
                "done": dones[index, :length].tolist(),
            })
        save_rollout_atomic(rollout_path, result)
        write_json_atomic(evidence_path, {
            "task_id": task_id, "task_group": task_group, "seeds": seeds,
            "explicit_init_state_ids": init_state_ids, "episodes": episodes,
        })
        return result

    def eval_policy_all(*args, **kwargs):
        # Videos are unnecessary for the paired raw trajectory/metric contract,
        # and cached rollouts have no render frames to encode on resume.
        kwargs["max_episodes_rendered"] = 0
        return original_eval_all(*args, **kwargs)

    module.make_policy = make_policy
    module.run_one = run_one
    module.rollout = rollout
    module.eval_policy_all = eval_policy_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("smolvla", "xvla"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
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
        or args.episodes_per_task != ENVIRONMENT_BATCH_SIZE
        or args.task_ids == list(range(10))
    ):
        raise ValueError("short smoke horizons require four episodes on a strict task subset")
    if not args.checkpoint_weight.is_file() or sha256(args.checkpoint_weight) != args.checkpoint_sha256:
        raise ValueError("paired evaluator checkpoint differs from pinned weight hash")
    if not os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("MUJOCO_GL") != "egl":
        raise EnvironmentError("paired LIBERO evaluator requires pinned LIBERO_CONFIG_PATH and MUJOCO_GL=egl")
    source_file = args.source / "src" / "lerobot" / "scripts" / "lerobot_eval.py"
    model_file = args.source / "src" / "lerobot" / "policies" / ("smolvla" if args.model == "smolvla" else "xvla") / ("modeling_smolvla.py" if args.model == "smolvla" else "modeling_xvla.py")
    if not source_file.is_file() or not model_file.is_file():
        raise FileNotFoundError("official LeRobot evaluator or policy source is missing")
    # Match the public evaluator's FP32/cuDNN execution policy explicitly so
    # four paired conditions cannot silently inherit different process flags.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    precision_controls = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    policy_batch_size, repaired = CONDITIONS[args.condition]
    environment_batch_size = ENVIRONMENT_BATCH_SIZE
    execution_mode = "serial_singleton" if policy_batch_size == 1 else "dynamic_batch"
    # The released X-VLA LIBERO checkpoint was trained/evaluated with absolute
    # end-effector control and an 800-step horizon.  LeRobot's generic LIBERO
    # defaults are relative control and the task-native horizon (520 for
    # LIBERO-Long), which produces a valid-looking but checkpoint-mismatched
    # evaluation.  SmolVLA keeps the generic relative-control recipe.
    control_mode = "absolute" if args.model == "xvla" else "relative"
    official_episode_length = 800 if args.model == "xvla" else 520
    episode_length = args.smoke_episode_length or official_episode_length
    manifest = {
        "schema_version": 2, "model": args.model, "condition": args.condition,
        "paired_protocol_version": PAIRED_PROTOCOL_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight": str(args.checkpoint_weight.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "official_evaluator_sha256": sha256(source_file),
        "official_policy_sha256": sha256(model_file),
        "paired_wrapper_sha256": sha256(Path(__file__)),
        "episodes_per_task": args.episodes_per_task, "seed": args.seed,
        "batch_size": policy_batch_size,
        "policy_batch_size": policy_batch_size,
        "environment_batch_size": environment_batch_size,
        "execution_mode": execution_mode,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "task_ids": args.task_ids,
        "control_mode": control_mode,
        "episode_length": episode_length,
        "smoke_test": args.smoke_episode_length is not None,
        "explicit_init_state_by_initialization_id": True,
        "smolvla_canonical_prompt_padding": repaired and args.model == "smolvla",
        "precision_controls": precision_controls,
    }
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("paired evaluator resume provenance differs from existing output")
    else:
        write_json_atomic(manifest_path, manifest)

    sys.path.insert(0, str((args.source / "src").resolve()))
    from lerobot.scripts import lerobot_eval

    install_evaluator_hooks(
        lerobot_eval, model_name=args.model, repaired=repaired,
        output=args.output, seed=args.seed,
        policy_batch_size=policy_batch_size,
        environment_batch_size=environment_batch_size,
    )
    sys.argv = [
        str(Path(__file__)),
        f"--policy.path={args.checkpoint.resolve()}",
        "--env.type=libero", "--env.task=libero_10",
        f"--env.task_ids={json.dumps(args.task_ids, separators=(',', ':'))}",
        f"--eval.batch_size={environment_batch_size}",
        f"--eval.n_episodes={args.episodes_per_task}",
        "--env.max_parallel_tasks=1",
        f"--seed={args.seed}",
        f"--output_dir={(args.output / 'official').resolve()}",
    ]
    if args.model == "xvla":
        sys.argv.extend([
            "--env.control_mode=absolute",
            f"--env.episode_length={episode_length}",
        ])
    elif args.smoke_episode_length is not None:
        sys.argv.append(f"--env.episode_length={episode_length}")
    mode = set_batch_invariant_mode() if repaired else contextlib.nullcontext()
    with mode:
        lerobot_eval.eval_main()


if __name__ == "__main__":
    main()

"""Run one exact-checkpoint MolmoAct2/LIBERO paired success condition.

The official LeRobot evaluator supplies the LIBERO worlds, reset schedule, and
success metric.  The public five-shard checkpoint is streamed one shard at a
time into a meta-allocated model because the workspace cannot retain a 22 GB
snapshot.  Four separate conditions are required for paired completion.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts import run_lerobot_paired_libero as shared
from scripts.model_invariance.adapters.molmoact2 import (
    CHECKPOINT_REVISION,
    CHECKPOINT_SHARDS,
    MolmoAct2Adapter,
)
from scripts.run_molmoact2_full_campaign import SUPPORT_HASHES
from lerobot.configs import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.policies.pretrained import PreTrainedPolicy


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/workspace/assets/molmoact2/checkpoint")
LEROBOT = Path("/workspace/models/lerobot")
OFFICIAL_MODEL = Path("/workspace/models/molmoact2-official")
PREFLIGHT = ROOT / (
    "results/cross_model/molmoact2/full/closed_loop/attempts/"
    "checkpoint_memory_stream_20261015T1230Z/preflight.json"
)
RECEIPT = ROOT / "results/cross_model/molmoact2/full/checkpoint_stream_receipt.json"
LEROBOT_REVISION = "e595b7902714ba51f91e47523f66f89c5181b649"
OFFICIAL_MODEL_REVISION = "66b87e64efd99dfd103241418113955cf64dfa9c"
PREFLIGHT_SHA256 = "818a8c4826b5349db51e5aaaee8d640d2d9cbfc214dd4a4f08297636b8848255"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def verify_assets() -> dict[str, Any]:
    if git_revision(LEROBOT) != LEROBOT_REVISION:
        raise ValueError("pinned official LeRobot evaluator revision differs")
    if git_revision(OFFICIAL_MODEL) != OFFICIAL_MODEL_REVISION:
        raise ValueError("pinned official MolmoAct2 repository revision differs")
    for name, expected in SUPPORT_HASHES.items():
        path = CHECKPOINT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"MolmoAct2 support artifact differs: {name}")
    if not PREFLIGHT.is_file() or sha256(PREFLIGHT) != PREFLIGHT_SHA256:
        raise ValueError("MolmoAct2 checkpoint-stream preflight differs")
    preflight = json.loads(PREFLIGHT.read_text())
    streaming = preflight.get("memory_streaming") or {}
    remote = streaming.get("remote_exact_shards") or []
    if (preflight.get("runnable") is not True or preflight.get("blockers") != []
            or len(remote) != 5 or any(not row.get("range_probe") for row in remote)):
        raise ValueError("MolmoAct2 checkpoint-stream preflight did not pass")
    evaluator = LEROBOT / "src/lerobot/scripts/lerobot_eval.py"
    environment = LEROBOT / "src/lerobot/envs/libero.py"
    model_source = CHECKPOINT / "modeling_molmoact2.py"
    for path in (evaluator, environment, model_source, Path(shared.__file__)):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "official_evaluator_path": str(evaluator.resolve()),
        "official_evaluator_sha256": sha256(evaluator),
        "official_environment_path": str(environment.resolve()),
        "official_environment_sha256": sha256(environment),
        "official_checkpoint_model_source_path": str(model_source.resolve()),
        "official_checkpoint_model_source_sha256": sha256(model_source),
        "official_model_repository_revision": OFFICIAL_MODEL_REVISION,
        "official_lerobot_revision": LEROBOT_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_shard_sha256": [row[2] for row in CHECKPOINT_SHARDS],
        "checkpoint_stream_preflight_path": str(PREFLIGHT.resolve()),
        "checkpoint_stream_preflight_sha256": PREFLIGHT_SHA256,
    }


@PreTrainedConfig.register_subclass("molmoact2_paired_eval")
@dataclass
class MolmoAct2EvalConfig(PreTrainedConfig):
    n_action_steps: int = 10

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError("evaluation-only MolmoAct2 configuration")

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        return None


class MolmoAct2PairedPolicy(PreTrainedPolicy):
    """Thin policy facade around the exact public Transformers checkpoint."""

    config_class = MolmoAct2EvalConfig
    name = "molmoact2_paired_eval"

    def __init__(self) -> None:
        super().__init__(MolmoAct2EvalConfig(device="cuda", n_action_steps=10))
        os.environ.update({
            "MOLMOACT2_CHECKPOINT": str(CHECKPOINT),
            "MOLMOACT2_STREAM_CHECKPOINT": "1",
            "MOLMOACT2_CHECKPOINT_RECEIPT": str(RECEIPT),
            "MOLMOACT2_DISABLE_CANONICAL_PROMPT_PADDING": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        self.adapter = MolmoAct2Adapter()
        self.model = self.adapter.load_model()
        self._action_queue: deque[torch.Tensor] = deque(maxlen=10)

    def get_optim_params(self):
        return self.parameters()

    def forward(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError("MolmoAct2 paired facade is evaluation-only")

    def reset(self) -> None:
        self._action_queue = deque(maxlen=10)

    def _request_trajectories(self, batch_size: int) -> list[torch.Tensor]:
        generators = getattr(self, "_paired_generators", None)
        hashes = getattr(self, "_paired_first_noise_hashes", None)
        active = getattr(self, "_paired_active_request_index", None)
        if not isinstance(generators, list) or not isinstance(hashes, list):
            raise RuntimeError("MolmoAct2 paired rollout lacks request-local generators")
        if active is None:
            indices = list(range(batch_size))
            if len(generators) != batch_size:
                raise RuntimeError("MolmoAct2 dynamic batch/generator cardinality differs")
        else:
            if batch_size != 1 or active not in range(len(generators)):
                raise RuntimeError("MolmoAct2 serial request identity differs")
            indices = [active]
        weight = self.model.model.action_expert.action_embed.weight
        dtype = weight.dtype
        device = weight.device
        result = []
        for index in indices:
            value = torch.randn(
                (10, 32), generator=generators[index], device=device, dtype=dtype
            )
            if hashes[index] is None:
                hashes[index] = hashlib.sha256(
                    value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest()
            result.append(value)
        return result

    @staticmethod
    def _batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            return value[index]
        if isinstance(value, (list, tuple)):
            return value[index]
        raise TypeError(f"MolmoAct2 paired input {key} has unsupported type {type(value)}")

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Any], **_kwargs: Any
    ) -> torch.Tensor:
        state = batch.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 8:
            raise RuntimeError("MolmoAct2 paired policy requires a Bx8 LIBERO state")
        batch_size = int(state.shape[0])
        image_key = "observation.images.image"
        wrist_key = (
            "observation.images.image2"
            if "observation.images.image2" in batch
            else "observation.images.wrist_image"
        )
        if image_key not in batch or wrist_key not in batch or "task" not in batch:
            raise RuntimeError("MolmoAct2 paired policy lacks two cameras or task text")
        trajectories = self._request_trajectories(batch_size)
        samples = [
            self.adapter.prepare_runtime_sample(
                image=self._batch_value(batch, image_key, index),
                wrist_image=self._batch_value(batch, wrist_key, index),
                state=state[index],
                task=str(self._batch_value(batch, "task", index)),
                trajectory=trajectories[index],
            )
            for index in range(batch_size)
        ]
        inputs = self.adapter.compose_batch(samples)
        actions = self.adapter.run_model(self.model, inputs)
        if tuple(actions.shape) != (batch_size, 10, 7):
            raise RuntimeError(f"MolmoAct2 action chunk shape differs: {actions.shape}")
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **_kwargs: Any) -> torch.Tensor:
        if self._action_queue:
            return self._action_queue.popleft()
        actions = self.predict_action_chunk(batch, **_kwargs)
        self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


def attach_molmoact2_hooks(policy: Any, model_name: str, _repaired: bool) -> None:
    if model_name != "molmoact2" or not isinstance(policy, MolmoAct2PairedPolicy):
        raise RuntimeError("paired evaluator did not construct the exact MolmoAct2 policy")


def precision_controls() -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--smoke-episode-length", type=int)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (args.episodes_per_task < 1 or args.seed < 0 or not args.task_ids
            or len(set(args.task_ids)) != len(args.task_ids)
            or any(task not in range(10) for task in args.task_ids)):
        raise ValueError("invalid MolmoAct2 paired episode schedule")
    if args.smoke_episode_length is not None and (
        args.smoke_episode_length < 1
        or args.episodes_per_task != shared.ENVIRONMENT_BATCH_SIZE
        or args.task_ids == list(range(10))
    ):
        raise ValueError("short smoke horizons require four episodes on a strict task subset")
    if not os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("MUJOCO_GL") != "egl":
        raise EnvironmentError("paired LIBERO evaluator requires LIBERO_CONFIG_PATH and MUJOCO_GL=egl")
    provenance = verify_assets()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        adapter_path = ROOT / "scripts/model_invariance/adapters/molmoact2.py"
        report = {
            "schema_version": 1, "model_id": "molmoact2", "passed": True,
            "paired_wrapper_path": str(Path(__file__).resolve()),
            "paired_wrapper_sha256": sha256(Path(__file__)),
            "adapter_path": str(adapter_path.resolve()),
            "adapter_sha256": sha256(adapter_path),
            **provenance,
        }
        shared.write_json_atomic(args.output / "preflight.json", report)
        print(json.dumps(report, sort_keys=True))
        return

    policy_batch_size, repaired = shared.CONDITIONS[args.condition]
    episode_length = args.smoke_episode_length or 520
    index_path = CHECKPOINT / "model.safetensors.index.json"
    manifest = {
        "schema_version": 2,
        "paired_protocol_version": shared.PAIRED_PROTOCOL_VERSION,
        "model": "molmoact2", "condition": args.condition,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_weight": str(index_path.resolve()),
        "checkpoint_sha256": sha256(index_path),
        "official_evaluator_sha256": provenance["official_evaluator_sha256"],
        "official_policy_sha256": provenance["official_checkpoint_model_source_sha256"],
        "paired_wrapper_sha256": sha256(Path(__file__)),
        "shared_rollout_wrapper_sha256": sha256(Path(shared.__file__)),
        "episodes_per_task": args.episodes_per_task, "seed": args.seed,
        "batch_size": policy_batch_size,
        "policy_batch_size": policy_batch_size,
        "environment_batch_size": shared.ENVIRONMENT_BATCH_SIZE,
        "execution_mode": "serial_singleton" if policy_batch_size == 1 else "dynamic_batch",
        "invariant_ops": repaired,
        "task_suite": "libero_10", "task_ids": args.task_ids,
        "control_mode": "relative", "episode_length": episode_length,
        "smoke_test": args.smoke_episode_length is not None,
        "explicit_init_state_by_initialization_id": True,
        "request_local_noise": True,
        "checkpoint_loading": "five_hash_verified_shards_streamed_from_RAM_to_meta_allocated_GPU_model",
        "checkpoint_shard_sha256": provenance["checkpoint_shard_sha256"],
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_model_repository_revision": OFFICIAL_MODEL_REVISION,
        "official_lerobot_revision": LEROBOT_REVISION,
        "official_environment_sha256": provenance["official_environment_sha256"],
        "paired_preflight_path": str(
            (CLOSED_PREFLIGHT := ROOT / (
                "results/cross_model/molmoact2/full/closed_loop/paired_preflight/"
                "exact_ram_stream_v5/preflight.json"
            )).resolve()
        ),
        "paired_preflight_sha256": sha256(CLOSED_PREFLIGHT),
        "task_manifest_path": str(
            (TASK_MANIFEST := ROOT / (
                "results/cross_model/molmoact2/full/closed_loop/paired/"
                "evaluator_task_manifest.json"
            )).resolve()
        ),
        "task_manifest_sha256": sha256(TASK_MANIFEST),
        "precision_controls": precision_controls(),
    }
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("MolmoAct2 paired resume provenance differs")
    else:
        shared.write_json_atomic(manifest_path, manifest)

    sys.path.insert(0, str((LEROBOT / "src").resolve()))
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.scripts import lerobot_eval

    lerobot_eval.make_policy = lambda *args, **kwargs: MolmoAct2PairedPolicy()
    lerobot_eval.make_pre_post_processors = lambda *args, **kwargs: (
        PolicyProcessorPipeline(steps=[]),
        PolicyProcessorPipeline(
            steps=[],
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
    shared.attach_policy_hooks = attach_molmoact2_hooks
    shared.install_evaluator_hooks(
        lerobot_eval, model_name="molmoact2", repaired=repaired,
        output=args.output, seed=args.seed,
        policy_batch_size=policy_batch_size,
        environment_batch_size=shared.ENVIRONMENT_BATCH_SIZE,
    )
    sys.argv = [
        str(Path(__file__)), "--policy.type=molmoact2",
        f"--policy.checkpoint_path={CHECKPOINT.resolve()}",
        "--policy.checkpoint_revision=" + CHECKPOINT_REVISION,
        "--policy.chunk_size=10", "--policy.n_action_steps=10",
        "--policy.norm_tag=libero", "--policy.inference_action_mode=continuous",
        "--policy.enable_inference_cuda_graph=false", "--policy.dtype=bfloat16",
        "--env.type=libero", "--env.task=libero_10",
        f"--env.task_ids={json.dumps(args.task_ids, separators=(',', ':'))}",
        "--env.control_mode=relative", f"--env.episode_length={episode_length}",
        f"--eval.batch_size={shared.ENVIRONMENT_BATCH_SIZE}",
        f"--eval.n_episodes={args.episodes_per_task}",
        "--env.max_parallel_tasks=1", f"--seed={args.seed}",
        f"--output_dir={(args.output / 'official').resolve()}",
    ]
    mode = set_batch_invariant_mode() if repaired else contextlib.nullcontext()
    with mode:
        lerobot_eval.eval_main()


if __name__ == "__main__":
    main()

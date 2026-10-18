"""Run one exact-checkpoint UniVLA paired LIBERO-10 success condition.

The released UniVLA image policy is wrapped in the pinned LeRobot LIBERO
environment so all four conditions share the same B=4 physical simulator,
explicit initialization IDs, and raw trajectory recorder.  Policy singleton
conditions execute four independent B=1 requests; dynamic conditions make one
genuine B=4 policy call.  UniVLA is greedy, so its request-local stochastic
input is the documented deterministic decoding contract.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers  # Load the pinned UniVLA version before extending sys.path.
from torch import nn

# The UniVLA environment has the required model stack; the already-pinned
# SmolVLA evaluation environment supplies only the simulator packages it lacks.
sys.path.insert(0, "/workspace/models/lerobot/src")
SMOLVLA_SITE = "/workspace/envs/smolvla_eval/lib/python3.12/site-packages"
# PyAV 19 removed ``av.option`` while the pinned LeRobot dataset annotations
# still reference it at import time.  Load the simulator environment's pinned
# PyAV 15 first, then restore UniVLA's package precedence.
sys.path.insert(0, SMOLVLA_SITE)
import av  # noqa: E402,F401
sys.path.remove(SMOLVLA_SITE)
sys.path.append(SMOLVLA_SITE)

import huggingface_hub  # noqa: E402

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
if not hasattr(huggingface_hub, "sync_bucket"):
    def _evaluation_only_sync_bucket(*_args, **_kwargs):
        raise RuntimeError("Hub bucket synchronization is unavailable in evaluation-only UniVLA")

    huggingface_hub.sync_bucket = _evaluation_only_sync_bucket

# The evaluator imports ``lerobot.policies`` only for its base type and two
# factory callables, both of which this runner replaces below.  Avoid importing
# every unrelated policy backend (notably FastWAM) into UniVLA's pinned model
# environment by exposing precisely that public surface.
_policies = types.ModuleType("lerobot.policies")
_policies.__path__ = ["/workspace/models/lerobot/src/lerobot/policies"]
_policies.__package__ = "lerobot.policies"
sys.modules["lerobot.policies"] = _policies

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.optim import OptimizerConfig  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402

_policies.PreTrainedPolicy = PreTrainedPolicy
_policies.make_policy = lambda *_args, **_kwargs: (_ for _ in ()).throw(
    RuntimeError("UniVLA runner did not install its exact policy factory")
)
_policies.make_pre_post_processors = _policies.make_policy

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts import run_lerobot_paired_libero as shared
from scripts.model_invariance.adapters.univla_baai import (
    EOA_TOKEN_ID,
    GENERATION_COMPATIBILITY,
    UniVLAAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts/model_invariance/adapters/univla_baai.py"
LEROBOT = Path("/workspace/models/lerobot")
OFFICIAL = Path("/workspace/models/univla")
FULL = ROOT / "results/cross_model/univla_baai/full"
RUNTIME = FULL / "runtime_checkpoint"
CHECKPOINT = RUNTIME / "policy/UNIVLA_LIBERO_IMG_BS192_8K"
VISION = RUNTIME / "vision"
RECEIPT = FULL / "checkpoint_cache_receipt.json"
TASK_MANIFEST = FULL / "closed_loop/paired/evaluator_task_manifest.json"
LEROBOT_REVISION = "e595b7902714ba51f91e47523f66f89c5181b649"
UNIVLA_REVISION = "a91ee5f269d7ad46d46635ee6481d2eb655b4af9"
POLICY_REVISION = "df6c94ff872f4fc056693e507c033ca390694118"
VISION_REVISION = "c81f916ad371289e205310a7539255e8a9396488"
SEED = 20261015
DETERMINISTIC_CONTRACT = hashlib.sha256(
    json.dumps(
        {
            "action_decoding": "greedy_constrained_FAST",
            "do_sample": False,
            "eoa_token_id": EOA_TOKEN_ID,
            "maximum_new_tokens": 80,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def verify_assets() -> dict[str, Any]:
    if revision(LEROBOT) != LEROBOT_REVISION or revision(OFFICIAL) != UNIVLA_REVISION:
        raise ValueError("UniVLA or LeRobot official source revision differs")
    if not RECEIPT.is_file() or not TASK_MANIFEST.is_file():
        raise FileNotFoundError("UniVLA checkpoint receipt or evaluator task manifest is missing")
    receipt = json.loads(RECEIPT.read_text())
    files = receipt.get("files") or {}
    if (
        receipt.get("all_pinned_files_verified") is not True
        or receipt.get("storage") != "sealed_memfd"
        or len(files) != 5
    ):
        raise ValueError("UniVLA exact checkpoint memory receipt is incomplete")
    required_seals = (
        fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    )
    for key, row in files.items():
        path = Path(row.get("path", ""))
        expected_revision = POLICY_REVISION if key.startswith("policy/") else VISION_REVISION
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or row.get("revision") != expected_revision
            or len(str(row.get("sha256", ""))) != 64
            or row.get("verification") != "full_stream_sha256_then_memfd_sealed_immutable"
        ):
            raise ValueError(f"UniVLA sealed checkpoint record differs: {key}")
        with path.open("rb") as stream:
            seals = fcntl.fcntl(stream.fileno(), fcntl.F_GET_SEALS)
        if seals & required_seals != required_seals:
            raise ValueError(f"UniVLA checkpoint is not immutable: {key}")
    evaluator = OFFICIAL / "reference/RoboVLMs/eval/libero/evaluate_libero_emu.py"
    wrapper = OFFICIAL / "reference/RoboVLMs/eval/libero/model_wrapper_emu.py"
    return {
        "checkpoint_receipt_path": str(RECEIPT.resolve()),
        "checkpoint_receipt_sha256": sha256(RECEIPT),
        "task_manifest_path": str(TASK_MANIFEST.resolve()),
        "task_manifest_sha256": sha256(TASK_MANIFEST),
        "official_evaluator_path": str(evaluator.resolve()),
        "official_evaluator_sha256": sha256(evaluator),
        "official_policy_wrapper_path": str(wrapper.resolve()),
        "official_policy_wrapper_sha256": sha256(wrapper),
        "official_univla_revision": UNIVLA_REVISION,
        "official_lerobot_revision": LEROBOT_REVISION,
        "policy_checkpoint_revision": POLICY_REVISION,
        "vision_checkpoint_revision": VISION_REVISION,
    }


@PreTrainedConfig.register_subclass("univla_baai")
@dataclass
class UniVLAEvalConfig(PreTrainedConfig):
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
        raise NotImplementedError("evaluation-only UniVLA configuration")

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        return None


class UniVLAPairedPolicy(PreTrainedPolicy):
    """LeRobot policy facade around the exact released UniVLA wrapper."""

    config_class = UniVLAEvalConfig
    name = "univla_baai"

    def __init__(self) -> None:
        super().__init__(UniVLAEvalConfig(device="cuda", n_action_steps=10))
        os.environ.update({
            "UNIVLA_CHECKPOINT": str(CHECKPOINT),
            "UNIVLA_VISION_TOKENIZER": str(VISION),
            "UNIVLA_SOURCE": str(OFFICIAL),
            "UNIVLA_SAMPLE_DIR": "/workspace/assets/lerobot_libero/fixture",
            "UNIVLA_REPLAY_MANIFEST": str(FULL / "replay_manifest.json"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        self.adapter = UniVLAAdapter()
        self.model = self.adapter.load_model()
        self._action_queue: deque[torch.Tensor] = deque(maxlen=10)
        self._paired_deterministic_contract_sha256 = DETERMINISTIC_CONTRACT

    def get_optim_params(self):
        return self.parameters()

    def forward(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError("UniVLA paired facade is evaluation-only")

    def reset(self) -> None:
        self._action_queue = deque(maxlen=10)

    @staticmethod
    def _batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            return value[index]
        if isinstance(value, (list, tuple)):
            return value[index]
        raise TypeError(f"UniVLA paired input {key} has unsupported type {type(value)}")

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Any], **_kwargs: Any
    ) -> torch.Tensor:
        state = batch.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 8:
            raise RuntimeError("UniVLA paired policy requires the official Bx8 LIBERO state")
        image_key = "observation.images.image"
        wrist_key = (
            "observation.images.image2"
            if "observation.images.image2" in batch
            else "observation.images.wrist_image"
        )
        if image_key not in batch or wrist_key not in batch or "task" not in batch:
            raise RuntimeError("UniVLA paired policy lacks two cameras or task text")
        batch_size = int(state.shape[0])
        samples = [
            self.adapter.prepare_runtime_sample(
                image=self._batch_value(batch, image_key, index),
                wrist_image=self._batch_value(batch, wrist_key, index),
                task=str(self._batch_value(batch, "task", index)),
            )
            for index in range(batch_size)
        ]
        inputs = self.adapter.compose_batch(samples)
        actions = self.adapter.run_model(self.model, inputs)
        if tuple(actions.shape) != (batch_size, 10, 7):
            raise RuntimeError(f"UniVLA action chunk shape differs: {actions.shape}")
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **_kwargs: Any) -> torch.Tensor:
        if self._action_queue:
            return self._action_queue.popleft()
        actions = self.predict_action_chunk(batch, **_kwargs)
        self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


def attach_univla_hooks(policy: Any, model_name: str, _repaired: bool) -> None:
    if model_name != "univla_baai" or not isinstance(policy, UniVLAPairedPolicy):
        raise RuntimeError("paired evaluator did not construct the exact UniVLA policy")
    if policy._paired_deterministic_contract_sha256 != DETERMINISTIC_CONTRACT:
        raise RuntimeError("UniVLA deterministic request contract differs")


def precision_controls() -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")
    return {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (
        args.episodes_per_task < 1 or args.seed < 0 or not args.task_ids
        or len(set(args.task_ids)) != len(args.task_ids)
        or any(task not in range(10) for task in args.task_ids)
    ):
        raise ValueError("invalid UniVLA paired episode schedule")
    if args.smoke_episode_length is not None and (
        args.smoke_episode_length < 1
        or args.episodes_per_task != shared.ENVIRONMENT_BATCH_SIZE
        or args.task_ids == list(range(10))
    ):
        raise ValueError("short smoke horizons require four episodes on a strict task subset")
    if not os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("MUJOCO_GL") != "egl":
        raise EnvironmentError("paired LIBERO evaluator requires LIBERO_CONFIG_PATH and MUJOCO_GL=egl")
    provenance = verify_assets()
    checkpoint_index = CHECKPOINT / "model.safetensors.index.json"
    official_policy = OFFICIAL / "reference/RoboVLMs/eval/libero/model_wrapper_emu.py"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        report = {
            "schema_version": 1, "model_id": "univla_baai", "passed": True,
            "paired_wrapper_path": str(Path(__file__).resolve()),
            "paired_wrapper_sha256": sha256(Path(__file__)),
            "adapter_path": str(ADAPTER.resolve()),
            "adapter_sha256": sha256(ADAPTER),
            "generation_compatibility": GENERATION_COMPATIBILITY,
            **provenance,
        }
        shared.write_json_atomic(args.output / "preflight.json", report)
        print(json.dumps(report, sort_keys=True))
        return

    policy_batch_size, repaired = shared.CONDITIONS[args.condition]
    episode_length = args.smoke_episode_length or 520
    manifest = {
        "schema_version": 2,
        "paired_protocol_version": shared.PAIRED_PROTOCOL_VERSION,
        "model": "univla_baai", "condition": args.condition,
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
        "request_local_noise": False,
        "deterministic_equivalent": True,
        "deterministic_contract_sha256": DETERMINISTIC_CONTRACT,
        "genuine_batched_policy_call": policy_batch_size == 4,
        "checkpoint_loading": "five_hash_verified_sealed_memfd_artifacts",
        "checkpoint_weight": str(checkpoint_index.resolve()),
        "checkpoint_sha256": sha256(checkpoint_index),
        "official_evaluator_sha256": provenance["official_evaluator_sha256"],
        "official_policy_sha256": sha256(official_policy),
        "adapter_path": str(ADAPTER.resolve()),
        "adapter_sha256": sha256(ADAPTER),
        "generation_compatibility": GENERATION_COMPATIBILITY,
        "paired_wrapper_path": str(Path(__file__).resolve()),
        "paired_wrapper_sha256": sha256(Path(__file__)),
        "shared_rollout_wrapper_sha256": sha256(Path(shared.__file__)),
        "task_manifest_path": str(TASK_MANIFEST.resolve()),
        "task_manifest_sha256": sha256(TASK_MANIFEST),
        "precision_controls": precision_controls(),
        **provenance,
    }
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("UniVLA paired resume provenance differs")
    else:
        shared.write_json_atomic(manifest_path, manifest)

    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.scripts import lerobot_eval

    lerobot_eval.make_policy = lambda *args, **kwargs: UniVLAPairedPolicy()
    lerobot_eval.make_pre_post_processors = lambda *args, **kwargs: (
        PolicyProcessorPipeline(steps=[]),
        PolicyProcessorPipeline(
            steps=[],
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
    shared.attach_policy_hooks = attach_univla_hooks
    shared.install_evaluator_hooks(
        lerobot_eval, model_name="univla_baai", repaired=repaired,
        output=args.output, seed=args.seed, policy_batch_size=policy_batch_size,
        environment_batch_size=shared.ENVIRONMENT_BATCH_SIZE,
    )
    sys.argv = [
        str(Path(__file__)), "--policy.type=univla_baai", "--policy.device=cuda",
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

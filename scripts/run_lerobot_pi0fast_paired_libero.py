"""Run one resumable official PI0-FAST/LIBERO paired condition.

This reuses the pinned LeRobot evaluator and rollout cache from the SmolVLA/
X-VLA runner. The public PI0-FAST checkpoint uses greedy temperature-zero
decoding, so a hash of that deterministic contract replaces a noise-tensor
hash. Local tokenizer files must first pass byte-for-byte comparison with
the pinned official Google/FAST repository file hashes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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


CONTRACT = {
    "temperature": 0.0,
    "use_kv_cache": True,
    "max_decoding_steps": 256,
    "stochastic_request_inputs": "none_greedy_argmax",
}
CONTRACT_SHA256 = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()


def attach_pi0fast_policy_hooks(policy, _model_name: str, _repaired: bool) -> None:
    if (policy.config.temperature != 0.0 or not policy.config.use_kv_cache
            or policy.config.max_decoding_steps != 256):
        raise RuntimeError("PI0-FAST official evaluator differs from pinned deterministic decoding")
    policy._paired_deterministic_contract_sha256 = CONTRACT_SHA256


def precision_controls() -> dict:
    # Match the official LeRobot evaluator's inference flags explicitly.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def verify_checkpoint_loaded(policy, checkpoint_weight: Path, checkpoint_sha256: str) -> dict:
    """Reject silent partial/random loads in the official PI0-FAST loader."""
    state = policy.state_dict()
    tensors_checked = 0
    source_bytes_checked = 0
    with safe_open(checkpoint_weight, framework="pt", device="cpu") as weights:
        source_keys = set(weights.keys())
        state_keys = set(state)
        if source_keys != state_keys:
            missing = sorted(source_keys - state_keys)
            unexpected = sorted(state_keys - source_keys)
            raise RuntimeError(
                "official PI0-FAST policy/checkpoint key sets differ: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        if not source_keys:
            raise RuntimeError("PI0-FAST checkpoint contains no tensors")
        for key in sorted(source_keys):
            source = weights.get_tensor(key)
            loaded = state[key].detach().to("cpu")
            if not torch.equal(loaded, source.to(dtype=loaded.dtype)):
                raise RuntimeError(f"official PI0-FAST policy did not load pinned checkpoint tensor {key}")
            tensors_checked += 1
            source_bytes_checked += source.numel() * source.element_size()
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "tensors_checked": tensors_checked,
        "source_bytes_checked": source_bytes_checked,
        "exact_after_policy_dtype_conversion": True,
    }


def install_verified_processor_overrides(lerobot_eval, text_tokenizer: Path, action_tokenizer: Path) -> None:
    """Route serialized processor steps to the same hash-verified local tokenizers."""
    original = lerobot_eval.make_pre_post_processors
    required = {
        "tokenizer_processor": {"tokenizer_name": str(text_tokenizer.resolve())},
        "action_tokenizer_processor": {
            "action_tokenizer_name": str(action_tokenizer.resolve()),
            "paligemma_tokenizer_name": str(text_tokenizer.resolve()),
        },
    }

    def make_pre_post_processors_with_verified_tokenizers(*positional, **keyword):
        overrides = {
            key: dict(value)
            for key, value in (keyword.get("preprocessor_overrides") or {}).items()
        }
        for step, fields in required.items():
            target = overrides.setdefault(step, {})
            for field, value in fields.items():
                existing = target.get(field)
                if existing is not None and existing != value:
                    raise RuntimeError(f"conflicting PI0-FAST processor override {step}.{field}")
                target[field] = value
        keyword["preprocessor_overrides"] = overrides
        return original(*positional, **keyword)

    lerobot_eval.make_pre_post_processors = make_pre_post_processors_with_verified_tokenizers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(shared.CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/assets/pi_fast/checkpoint"))
    parser.add_argument("--checkpoint-weight", type=Path, default=Path("/workspace/assets/pi_fast/checkpoint/model.safetensors"))
    parser.add_argument("--checkpoint-sha256", default="f1f8236d44016a533daba7b8249756e84df047f46e20ca17d3fc34d8f2f691ba")
    parser.add_argument("--text-tokenizer", type=Path, default=Path("/workspace/assets/pi_fast/text_tokenizer"))
    parser.add_argument("--action-tokenizer", type=Path, default=Path("/workspace/assets/pi_fast/action_tokenizer"))
    parser.add_argument("--tokenizer-provenance", type=Path, default=Path("results/cross_model/pi_fast/full/tokenizer_provenance.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--smoke-episode-length", type=int)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--source", type=Path, default=Path("/workspace/models/lerobot"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (args.episodes_per_task < 1 or args.seed < 0 or not args.task_ids
            or len(set(args.task_ids)) != len(args.task_ids)
            or any(task not in range(10) for task in args.task_ids)):
        raise ValueError("invalid PI0-FAST paired episode count or seed")
    if args.smoke_episode_length is not None and (
        args.smoke_episode_length < 1
        or args.episodes_per_task != shared.ENVIRONMENT_BATCH_SIZE
        or args.task_ids == list(range(10))
    ):
        raise ValueError("short smoke horizons require four episodes on a strict task subset")
    if not args.checkpoint_weight.is_file() or shared.sha256(args.checkpoint_weight) != args.checkpoint_sha256:
        raise ValueError("PI0-FAST checkpoint differs from pinned weight hash")
    if not os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("MUJOCO_GL") != "egl":
        raise EnvironmentError("paired LIBERO evaluator requires LIBERO_CONFIG_PATH and MUJOCO_GL=egl")
    tokenizer_report = json.loads(args.tokenizer_provenance.read_text())
    if (tokenizer_report.get("model_id") != "pi_fast" or tokenizer_report.get("passed") is not True
            or len(tokenizer_report.get("tokenizers", [])) != 2
            or any(not item.get("complete_byte_match") for item in tokenizer_report["tokenizers"])):
        raise ValueError("PI0-FAST local tokenizers lack passed official-byte verification")
    roots = {item["repository"]: item["local_root"] for item in tokenizer_report["tokenizers"]}
    if (roots.get("google/paligemma-3b-pt-224") != str(args.text_tokenizer.resolve())
            or roots.get("jadechoghari/tokenizer-lib-mean") != str(args.action_tokenizer.resolve())):
        raise ValueError("PI0-FAST tokenizer paths differ from verified files")
    for item in tokenizer_report["tokenizers"]:
        for file_record in item["files"]:
            path = Path(item["local_root"]) / file_record["file"]
            if not path.is_file() or shared.sha256(path) != file_record.get("local_sha256"):
                raise ValueError(f"verified PI0-FAST tokenizer changed: {path}")
    source_file = args.source / "src" / "lerobot" / "scripts" / "lerobot_eval.py"
    model_file = args.source / "src" / "lerobot" / "policies" / "pi0_fast" / "modeling_pi0_fast.py"
    if not source_file.is_file() or not model_file.is_file():
        raise FileNotFoundError("official LeRobot evaluator or PI0-FAST policy source is missing")
    config = json.loads((args.checkpoint / "config.json").read_text())
    if any(config.get(key) != value for key, value in CONTRACT.items() if key != "stochastic_request_inputs"):
        raise ValueError("PI0-FAST checkpoint does not declare pinned greedy decoding")
    if (config.get("text_tokenizer_name") != "google/paligemma-3b-pt-224"
            or config.get("action_tokenizer_name") != "jadechoghari/tokenizer-lib-mean"):
        raise ValueError("PI0-FAST checkpoint tokenizer identifiers differ from verified repositories")
    if args.preflight_only:
        report = {
            "model_id": "pi_fast", "checkpoint_sha256": args.checkpoint_sha256,
            "tokenizer_provenance_sha256": shared.sha256(args.tokenizer_provenance),
            "official_evaluator_sha256": shared.sha256(source_file),
            "official_policy_sha256": shared.sha256(model_file),
            "preflight_passed": True,
        }
        report_path = args.output / "preflight.json"
        if report_path.is_file() and json.loads(report_path.read_text()) != report:
            raise FileExistsError("existing PI0-FAST paired preflight differs")
        shared.write_json_atomic(report_path, report)
        print(json.dumps(report))
        return
    policy_batch_size, repaired = shared.CONDITIONS[args.condition]
    environment_batch_size = shared.ENVIRONMENT_BATCH_SIZE
    execution_mode = "serial_singleton" if policy_batch_size == 1 else "dynamic_batch"
    episode_length = args.smoke_episode_length or 520
    manifest = {
        "schema_version": 2, "model": "pi_fast", "condition": args.condition,
        "paired_protocol_version": shared.PAIRED_PROTOCOL_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight": str(args.checkpoint_weight.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "official_evaluator_sha256": shared.sha256(source_file),
        "official_policy_sha256": shared.sha256(model_file),
        "paired_wrapper_sha256": shared.sha256(Path(__file__)),
        "shared_rollout_wrapper_sha256": shared.sha256(Path(shared.__file__)),
        "tokenizer_provenance_sha256": shared.sha256(args.tokenizer_provenance),
        "text_tokenizer": str(args.text_tokenizer.resolve()),
        "action_tokenizer": str(args.action_tokenizer.resolve()),
        "deterministic_decoding_contract": CONTRACT,
        "deterministic_decoding_sha256": CONTRACT_SHA256,
        "episodes_per_task": args.episodes_per_task, "seed": args.seed,
        "batch_size": policy_batch_size,
        "policy_batch_size": policy_batch_size,
        "environment_batch_size": environment_batch_size,
        "execution_mode": execution_mode,
        "invariant_ops": repaired,
        "task_suite": "libero_10",
        "task_ids": args.task_ids,
        "control_mode": "relative",
        "episode_length": episode_length,
        "smoke_test": args.smoke_episode_length is not None,
        "explicit_init_state_by_initialization_id": True,
        "precision_controls": precision_controls(),
    }
    manifest_path = args.output / "condition_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("PI0-FAST paired resume provenance differs from existing output")
    else:
        shared.write_json_atomic(manifest_path, manifest)

    sys.path.insert(0, str((args.source / "src").resolve()))
    from lerobot.scripts import lerobot_eval

    original_make_policy = lerobot_eval.make_policy

    def make_policy_with_verified_tokenizers(*positional, **keyword):
        cfg = keyword.get("cfg", positional[0] if positional else None)
        if cfg is None or cfg.type != "pi0_fast":
            raise RuntimeError("official evaluator did not request a PI0-FAST policy")
        if (cfg.text_tokenizer_name != "google/paligemma-3b-pt-224"
                or cfg.action_tokenizer_name != "jadechoghari/tokenizer-lib-mean"):
            raise RuntimeError("official policy config tokenizer sources changed")
        cfg.text_tokenizer_name = str(args.text_tokenizer.resolve())
        cfg.action_tokenizer_name = str(args.action_tokenizer.resolve())
        policy = original_make_policy(*positional, **keyword)
        # The released PI0FastPolicy.from_pretrained can return a randomly
        # initialized or partially loaded model after a caught load error.
        # Compare every checkpoint tensor, after the policy's declared dtype
        # conversion, rather than trusting that method's return value.
        report = verify_checkpoint_loaded(policy, args.checkpoint_weight, args.checkpoint_sha256)
        shared.write_json_atomic(args.output / "checkpoint_load_verification.json", report)
        return policy

    lerobot_eval.make_policy = make_policy_with_verified_tokenizers
    install_verified_processor_overrides(
        lerobot_eval, args.text_tokenizer, args.action_tokenizer,
    )
    shared.attach_policy_hooks = attach_pi0fast_policy_hooks
    shared.install_evaluator_hooks(
        lerobot_eval, model_name="pi_fast", repaired=repaired,
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

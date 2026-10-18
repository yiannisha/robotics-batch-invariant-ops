"""Attempt the checkpoint-matched official LeRobot LIBERO evaluator once.

A successful one-episode-per-task probe is not a paired success-rate campaign. A
failed probe retains exact command, checkpoint hash, stdout, and stderr for
the per-model closed-loop applicability gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(output: str, timeout: bool) -> str:
    if timeout:
        return "evaluator_rollout_timeout" if "Running rollout" in output else "evaluator_startup_timeout"
    lower = output.lower()
    if "eglinitialize" in lower or "eglquerystring" in lower or "egl" in lower and "error" in lower:
        return "egl_graphics_unavailable"
    if "vulkan" in lower and ("error" in lower or "driver" in lower):
        return "vulkan_graphics_unavailable"
    if "modulenotfounderror" in lower or "importerror" in lower:
        return "missing_evaluator_dependency"
    if "file not found" in lower or "filenotfounderror" in lower:
        return "missing_evaluator_asset"
    return "evaluator_runtime_failure"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-weight", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/workspace/models/lerobot"))
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--control-mode", choices=("relative", "absolute"))
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.checkpoint_weight.is_file() or sha256(args.checkpoint_weight) != args.expected_weight_sha256:
        raise ValueError("checkpoint weight is missing or differs from the pinned pilot SHA")
    source_evidence = args.source / "docs" / "source" / "libero.mdx"
    if not source_evidence.is_file() or not args.python.is_file():
        raise FileNotFoundError("official evaluator code/environment is missing")
    control_mode = args.control_mode
    episode_length = args.episode_length
    recipe_evidence = source_evidence
    if args.model_id == "xvla":
        control_mode = control_mode or "absolute"
        episode_length = episode_length or 800
        if control_mode != "absolute" or episode_length != 800:
            raise ValueError("released X-VLA LIBERO checkpoint requires absolute control and 800 steps")
        recipe_evidence = args.source / "docs" / "source" / "xvla.mdx"
        if not recipe_evidence.is_file():
            raise FileNotFoundError("X-VLA evaluator recipe evidence is missing")
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "evaluator_probe.stdout.log"
    stderr_path = output_dir / "evaluator_probe.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("evaluator probe logs already exist")
    command = [
        str(args.python), "-m", "lerobot.scripts.lerobot_eval",
        f"--policy.path={args.checkpoint}",
        "--env.type=libero", f"--env.task={args.task}",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        "--env.max_parallel_tasks=1",
        f"--output_dir={output_dir / 'probe_output'}",
    ]
    if control_mode is not None:
        command.append(f"--env.control_mode={control_mode}")
    if episode_length is not None:
        command.append(f"--env.episode_length={episode_length}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.source / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    attempted_at = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment, start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        stderr += f"\nKilled evaluator process group after {args.timeout_seconds}s timeout.\n"
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    exit_code = process.returncode
    eval_info_path = output_dir / "probe_output" / "eval_info.json"
    eval_info = json.loads(eval_info_path.read_text()) if exit_code == 0 and eval_info_path.is_file() else None
    overall = eval_info.get("overall", {}) if isinstance(eval_info, dict) else {}
    report = {
        "schema_version": 1,
        "model_id": args.model_id,
        "evaluator": "official LeRobot LIBERO",
        "status": "blocked" if exit_code != 0 else "probe_succeeded_requires_paired_campaign",
        "attempt_timestamp_utc": attempted_at,
        "attempted_command": command,
        "probe_episodes_requested_per_task": 1,
        "python_executable": str(args.python.resolve()),
        "evaluator_environment": {
            key: environment.get(key) for key in ("LIBERO_CONFIG_PATH", "MUJOCO_GL")
        },
        "exit_code": exit_code,
        "failure_class": classify(stdout + "\n" + stderr, timed_out) if exit_code != 0 else None,
        "checkpoint_weight_path": str(args.checkpoint_weight.resolve()),
        "checkpoint_weight_sha256": args.expected_weight_sha256,
        "source_evidence_path": str(source_evidence.resolve()),
        "source_evidence_sha256": sha256(source_evidence),
        "policy_recipe_evidence_path": str(recipe_evidence.resolve()),
        "policy_recipe_evidence_sha256": sha256(recipe_evidence),
        "task_suite": args.task,
        "control_mode": control_mode,
        "episode_length": episode_length,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "probe_eval_info_path": str(eval_info_path.resolve()) if eval_info is not None else None,
        "probe_eval_info_sha256": sha256(eval_info_path) if eval_info is not None else None,
        "probe_tasks_completed": len(eval_info.get("per_task", [])) if eval_info is not None else 0,
        "probe_episodes_completed": overall.get("n_episodes", 0),
        "probe_successes": overall.get("n_success") if eval_info is not None else None,
        "paired_episodes_completed": 0,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("model_id", "status", "exit_code", "failure_class")}))


if __name__ == "__main__":
    main()

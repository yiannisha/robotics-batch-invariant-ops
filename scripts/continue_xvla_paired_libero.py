"""Finish the four exact-checkpoint X-VLA LIBERO conditions, one GPU job at a time.

Attach to an already-running native-singleton PID. After it exits, verify its
raw reset prefix and official metrics before starting each remaining condition.
Existing incomplete conditions resume through the rollout cache; completed
conditions are never silently overwritten or rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from audit_lerobot_paired_prefix import audit


ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "results/cross_model/xvla/full/closed_loop/paired"
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
MAX_CONDITION_ATTEMPTS = 20
# Protocol v2 writes 130 physical batch files per condition (500 planned
# episodes plus task-boundary padding), rather than one file per episode.
EXPECTED_ROLLOUT_FILES = 4 * 130
CHECKPOINT = Path("/workspace/assets/xvla/checkpoint")
WEIGHT = CHECKPOINT / "model.safetensors"
WEIGHT_SHA256 = "2a0e29448111df56b9485cb1f964db799205794c49baeda48ef960589ce649ab"
EVAL_PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
CGROUP_MEMORY = Path("/sys/fs/cgroup/memory.current")


def gpu_free_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def wait_for_resources(minimum_gpu_free_mib: int, maximum_cgroup_bytes: int) -> None:
    """Wait outside the retry budget until a batch-4 launch has safe headroom."""
    while True:
        free = gpu_free_mib()
        memory = int(CGROUP_MEMORY.read_text())
        if free >= minimum_gpu_free_mib and memory <= maximum_cgroup_bytes:
            print(
                f"X-VLA resource gate passed: gpu_free_mib={free} "
                f"cgroup_memory_bytes={memory}",
                flush=True,
            )
            return
        print(
            f"X-VLA waiting: gpu_free_mib={free}/{minimum_gpu_free_mib} "
            f"cgroup_memory_bytes={memory}/{maximum_cgroup_bytes}",
            flush=True,
        )
        time.sleep(30)


def validate_condition(condition: str) -> bool:
    folder = PAIRED / condition
    if not (folder / "condition_manifest.json").is_file():
        return False
    prefix = audit(folder)
    if (prefix["condition"] != condition or prefix["model_id"] != "xvla"
            or prefix.get("causal_pairing_eligible") is not True
            or prefix.get("paired_protocol_version") != 2
            or not prefix["all_ten_tasks_have_50_resets"]
            or prefix["planned_episodes_cached"] != 500):
        return False
    metrics_path = folder / "official/eval_info.json"
    if not metrics_path.is_file():
        return False
    info = json.loads(metrics_path.read_text())
    by_task = {row.get("task_id"): row for row in info.get("per_task", [])}
    return (len(info.get("per_task", [])) == 10
            and set(by_task) == set(range(10))
            and all(row.get("n_episodes") == 50 for row in by_task.values())
            and info.get("overall", {}).get("n_episodes") == 500)


def wait_for_current(pid: int, condition: str = "native_singleton") -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"invalid X-VLA condition: {condition}")
    expected = ("scripts/run_lerobot_paired_libero.py", "--model", "xvla",
                "--condition", condition)
    while True:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            return
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if any(part not in cmdline for part in expected):
            raise RuntimeError(f"PID {pid} no longer identifies the pinned X-VLA condition")
        time.sleep(20)


def run_condition(condition: str) -> None:
    folder = PAIRED / condition
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": "/workspace/models/lerobot/src:/workspace/robotics-batch-invariant-ops",
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl",
    })
    command = [
        str(EVAL_PYTHON), "-u", "scripts/run_lerobot_paired_libero.py",
        "--model", "xvla", "--condition", condition,
        "--checkpoint", str(CHECKPOINT), "--checkpoint-weight", str(WEIGHT),
        "--checkpoint-sha256", WEIGHT_SHA256,
        "--output", str(folder), "--episodes-per-task", "50",
    ]
    print(f"Starting {condition}; log={folder / 'continuation.log'}", flush=True)
    with (folder / "continuation.log").open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{condition} evaluator exited {result.returncode}; see continuation.log")
    if not validate_condition(condition):
        raise RuntimeError(f"{condition} finished without 500 verified resets and official metrics")
    print(f"Verified {condition}: 500 planned resets and official metrics", flush=True)


def run_condition_resumable(
    condition: str,
    *,
    minimum_gpu_free_mib: int,
    maximum_cgroup_bytes: int,
) -> None:
    retry_path = PAIRED / condition / "supervisor_retries.json"
    failures: list[str] = []
    if retry_path.is_file():
        previous = json.loads(retry_path.read_text())
        if (previous.get("condition") != condition
                or previous.get("maximum_attempts") != MAX_CONDITION_ATTEMPTS
                or not isinstance(previous.get("failures"), list)):
            raise RuntimeError(f"invalid existing retry ledger: {retry_path}")
        failures = [str(failure) for failure in previous["failures"]]
    if len(failures) >= MAX_CONDITION_ATTEMPTS:
        raise RuntimeError(f"{condition} exhausted its persisted retry budget")
    for attempt in range(len(failures) + 1, MAX_CONDITION_ATTEMPTS + 1):
        wait_for_resources(minimum_gpu_free_mib, maximum_cgroup_bytes)
        try:
            run_condition(condition)
            return
        except Exception as error:
            if validate_condition(condition):
                return
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            retry_path.write_text(json.dumps({
                "schema_version": 1, "condition": condition,
                "maximum_attempts": MAX_CONDITION_ATTEMPTS,
                "attempts_failed": len(failures), "failures": failures,
                "append_resume_safe": True,
            }, indent=2) + "\n")
            if attempt == MAX_CONDITION_ATTEMPTS:
                raise
            delay = min(60, 5 * attempt)
            print(
                f"X-VLA {condition} attempt {attempt} failed; preserving caches and "
                f"retrying in {delay}s: {error}", flush=True,
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-pid", type=int)
    parser.add_argument("--current-condition", choices=CONDITIONS, default="native_singleton")
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=20_000)
    parser.add_argument("--maximum-cgroup-bytes", type=int, default=115_000_000_000)
    args = parser.parse_args()
    if ((args.current_pid is not None and args.current_pid <= 0)
            or args.minimum_gpu_free_mib <= 0 or args.maximum_cgroup_bytes <= 0):
        raise ValueError("invalid PID or resource threshold")
    if args.current_pid is not None:
        wait_for_current(args.current_pid, args.current_condition)
    else:
        print("X-VLA starting a fresh protocol-v2 paired campaign", flush=True)
    for condition in CONDITIONS:
        if validate_condition(condition):
            print(f"Already verified {condition}; preserving existing evidence", flush=True)
            continue
        run_condition_resumable(
            condition,
            minimum_gpu_free_mib=args.minimum_gpu_free_mib,
            maximum_cgroup_bytes=args.maximum_cgroup_bytes,
        )
    pairing_report = PAIRED / "reset_pairing_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_lerobot_reset_pairing.py",
        "--root", str(PAIRED), "--output", str(pairing_report),
    ], cwd=ROOT, check=True)
    pairing = json.loads(pairing_report.read_text())
    if pairing.get("fully_paired") is not True:
        raise RuntimeError(
            "four conditions finished but physical reset/noise pairing failed; "
            "preserving raw evidence without reduction"
        )
    subprocess.run([
        sys.executable, "scripts/compress_preserved_rollouts.py",
        "--root", str(PAIRED), "--allow-fully-paired",
        "--expected-files", str(EXPECTED_ROLLOUT_FILES),
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(PAIRED), "--gzip-episodes",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/finalize_lerobot_paired_record.py",
        "--model-id", "xvla",
        "--paired-root", str(PAIRED),
        "--attempt-record", str(PAIRED.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(WEIGHT),
        "--checkpoint-sha256", WEIGHT_SHA256,
        "--source-evidence", "/workspace/models/lerobot/docs/source/libero.mdx",
        "--official-evaluator-source",
        "/workspace/models/lerobot/src/lerobot/scripts/lerobot_eval.py",
    ], cwd=ROOT, check=True)
    print(
        "All four X-VLA paired conditions, reducer, and completed evaluator record are verified",
        flush=True,
    )


if __name__ == "__main__":
    main()

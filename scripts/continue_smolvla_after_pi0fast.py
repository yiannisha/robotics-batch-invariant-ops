"""Run SmolVLA's four paired LIBERO conditions after the PI0-FAST worker exits.

The predecessor result is recorded but is not a gate: one blocked evaluator
must not prevent the next independently runnable model from being attempted.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from audit_lerobot_paired_prefix import audit


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cross_model/smolvla/full"
PAIRED = FULL / "closed_loop/paired"
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
MAX_CONDITION_ATTEMPTS = 20
# Protocol v2 writes 130 physical batch files per condition (500 planned
# episodes plus task-boundary padding), rather than one file per episode.
EXPECTED_ROLLOUT_FILES = 4 * 130
CHECKPOINT = Path("/workspace/assets/smolvla/checkpoint")
WEIGHT = CHECKPOINT / "model.safetensors"
WEIGHT_SHA256 = "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"
EVAL_PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
CGROUP_MEMORY = Path("/sys/fs/cgroup/memory.current")
SUPERVISOR_LOCK = PAIRED / ".supervisor.lock"


def gpu_free_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def wait_for_resources(minimum_gpu_free_mib: int, maximum_cgroup_bytes: int) -> None:
    while True:
        free = gpu_free_mib()
        memory = int(CGROUP_MEMORY.read_text())
        if free >= minimum_gpu_free_mib and memory <= maximum_cgroup_bytes:
            print(
                f"SmolVLA resource gate passed: gpu_free_mib={free} "
                f"cgroup_memory_bytes={memory}",
                flush=True,
            )
            return
        print(
            f"SmolVLA waiting: gpu_free_mib={free}/{minimum_gpu_free_mib} "
            f"cgroup_memory_bytes={memory}/{maximum_cgroup_bytes}",
            flush=True,
        )
        time.sleep(30)


def wait_for_pi_worker(pid: int) -> dict:
    while True:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            break
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        if "continue_pi0fast_after_xvla.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies the PI0-FAST worker")
        time.sleep(20)
    attempt_path = ROOT / "results/cross_model/pi_fast/full/closed_loop/evaluator_attempt.json"
    return json.loads(attempt_path.read_text()) if attempt_path.is_file() else {}


def wait_for_active_condition(pid: int) -> None:
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
        if "run_lerobot_smolvla_paired_libero.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies the active SmolVLA condition")
        time.sleep(20)


def condition_worker_pids(
    condition: str, proc_root: Path = Path("/proc"),
) -> list[int]:
    """Find live workers for this exact condition/output pair.

    A condition may be started early to use otherwise idle evaluator capacity.
    The sequential supervisor must join that worker instead of launching a
    second writer into the same append/resume cache.
    """
    expected_output = str((PAIRED / condition).resolve())
    matches: list[int] = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            arguments = [
                part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if state == "Z" or not any(
            argument.endswith("run_lerobot_smolvla_paired_libero.py")
            for argument in arguments
        ):
            continue
        try:
            observed_condition = arguments[arguments.index("--condition") + 1]
            observed_output = str(Path(arguments[arguments.index("--output") + 1]).resolve())
        except (ValueError, IndexError):
            continue
        if observed_condition == condition and observed_output == expected_output:
            matches.append(int(proc.name))
    return sorted(matches)


def join_external_condition(condition: str) -> None:
    announced: tuple[int, ...] | None = None
    while True:
        workers = tuple(condition_worker_pids(condition))
        if not workers:
            return
        if workers != announced:
            print(
                f"Joining external SmolVLA {condition} worker(s) {workers}; "
                "a duplicate cache writer will not be launched",
                flush=True,
            )
            announced = workers
        time.sleep(20)


def validate_condition(condition: str) -> bool:
    folder = PAIRED / condition
    if not (folder / "condition_manifest.json").is_file():
        return False
    prefix = audit(folder)
    if (prefix["condition"] != condition or prefix["model_id"] != "smolvla"
            or prefix.get("causal_pairing_eligible") is not True
            or prefix.get("paired_protocol_version") != 2
            or not prefix["all_ten_tasks_have_50_resets"]
            or prefix["planned_episodes_cached"] != 500):
        return False
    metrics_path = folder / "official/eval_info.json"
    verification_path = folder / "checkpoint_load_verification.json"
    if not metrics_path.is_file() or not verification_path.is_file():
        return False
    info = json.loads(metrics_path.read_text())
    verification = json.loads(verification_path.read_text())
    by_task = {row.get("task_id"): row for row in info.get("per_task", [])}
    return (len(info.get("per_task", [])) == 10
            and set(by_task) == set(range(10))
            and all(row.get("n_episodes") == 50 for row in by_task.values())
            and info.get("overall", {}).get("n_episodes") == 500
            and verification.get("checkpoint_sha256") == WEIGHT_SHA256
            and verification.get("tensors_checked") == 788
            and verification.get("source_bytes_checked") == 1_217_932_096
            and verification.get("exact_after_policy_dtype_conversion") is True)


def run_condition(condition: str) -> None:
    folder = PAIRED / condition
    folder.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": "/workspace/models/lerobot/src:/workspace/robotics-batch-invariant-ops",
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl",
    })
    command = [
        str(EVAL_PYTHON), "-u", "scripts/run_lerobot_smolvla_paired_libero.py",
        "--condition", condition,
        "--checkpoint", str(CHECKPOINT), "--checkpoint-weight", str(WEIGHT),
        "--checkpoint-sha256", WEIGHT_SHA256,
        "--output", str(folder), "--episodes-per-task", "50",
    ]
    print(f"Starting SmolVLA {condition}; log={folder / 'continuation.log'}", flush=True)
    with (folder / "continuation.log").open("a") as log:
        result = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"SmolVLA {condition} exited {result.returncode}; see continuation.log")
    if not validate_condition(condition):
        raise RuntimeError(f"SmolVLA {condition} lacks verified resets/metrics/checkpoint evidence")
    print(f"Verified SmolVLA {condition}: 500 planned resets and official metrics", flush=True)


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
                or not isinstance(previous.get("failures"), list)
                or previous.get("attempts_failed") != len(previous["failures"])):
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
                f"SmolVLA {condition} attempt {attempt} failed; preserving caches and "
                f"retrying in {delay}s: {error}", flush=True,
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-worker-pid", type=int, required=True)
    parser.add_argument("--active-condition-pid", type=int)
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=12_000)
    parser.add_argument("--maximum-cgroup-bytes", type=int, default=115_000_000_000)
    args = parser.parse_args()
    if (args.pi_worker_pid <= 0 or args.minimum_gpu_free_mib <= 0
            or args.maximum_cgroup_bytes <= 0):
        raise ValueError("invalid predecessor PID or resource threshold")
    SUPERVISOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    supervisor_lock = SUPERVISOR_LOCK.open("a+")
    try:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another SmolVLA paired supervisor already holds the campaign lock") from error
    predecessor = wait_for_pi_worker(args.pi_worker_pid)
    if args.active_condition_pid is not None:
        if args.active_condition_pid <= 0:
            raise ValueError("invalid active SmolVLA condition PID")
        wait_for_active_condition(args.active_condition_pid)
    print(
        "PI0-FAST predecessor ended; "
        f"status={predecessor.get('status', 'missing')} (SmolVLA proceeds independently)",
        flush=True,
    )
    for condition in CONDITIONS:
        if validate_condition(condition):
            print(f"Already verified SmolVLA {condition}; preserving evidence", flush=True)
            continue
        join_external_condition(condition)
        if validate_condition(condition):
            print(
                f"Verified externally completed SmolVLA {condition}; preserving evidence",
                flush=True,
            )
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
    if json.loads(pairing_report.read_text()).get("fully_paired") is not True:
        raise RuntimeError("SmolVLA physical reset/noise pairing failed")
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
        "--model-id", "smolvla", "--paired-root", str(PAIRED),
        "--attempt-record", str(PAIRED.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(WEIGHT), "--checkpoint-sha256", WEIGHT_SHA256,
        "--source-evidence", "/workspace/models/lerobot/docs/source/libero.mdx",
        "--official-evaluator-source",
        "/workspace/models/lerobot/src/lerobot/scripts/lerobot_eval.py",
    ], cwd=ROOT, check=True)
    print("SmolVLA paired campaign and completed evaluator record verified", flush=True)


if __name__ == "__main__":
    main()

"""Continue all UVA paired closed-loop conditions after a live native run."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_uva_paired_prefix import audit


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
RUNNER = ROOT / "scripts/run_uva_paired_libero.py"
RESULT_ROOT = ROOT / "results/cross_model"
CHECKPOINT = Path("/workspace/assets/uva/libero10.ckpt")
CHECKPOINT_SHA256 = "4554b9bb2f4f3d44a7e51d5c7bd42a7740451874ac6cf1fea91e088a949f0dbf"
SOURCE_EVIDENCE = ROOT / "research/uva/investigation.md"
OFFICIAL_EVALUATOR = Path("/workspace/models/uva/eval_sim.py")
CONDITIONS = (
    "native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic",
)
MAX_CONDITION_ATTEMPTS = 20


def condition_worker_pids(
    path_kind: str, condition: str, proc_root: Path = Path("/proc"),
) -> list[int]:
    if path_kind not in ("action", "joint") or condition not in CONDITIONS:
        raise ValueError("invalid UVA worker identity")
    expected_output = str(condition_root(path_kind, condition).resolve())
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
            argument.endswith("run_uva_paired_libero.py") for argument in arguments
        ):
            continue
        try:
            observed_path = arguments[arguments.index("--path") + 1]
            observed_condition = arguments[arguments.index("--condition") + 1]
            observed_output = str(Path(arguments[arguments.index("--output") + 1]).resolve())
        except (ValueError, IndexError):
            continue
        if (
            observed_path == path_kind
            and observed_condition == condition
            and observed_output == expected_output
        ):
            matches.append(int(proc.name))
    return sorted(matches)


def join_external_condition(path_kind: str, condition: str) -> None:
    announced: tuple[int, ...] | None = None
    while True:
        workers = tuple(condition_worker_pids(path_kind, condition))
        if len(workers) > 1:
            raise RuntimeError(
                f"multiple UVA {path_kind} {condition} workers are writing one cache: {workers}"
            )
        if not workers:
            return
        if workers != announced:
            print(
                f"Joining external UVA {path_kind} {condition} worker {workers[0]}; "
                "a duplicate writer will not be launched",
                flush=True,
            )
            announced = workers
        time.sleep(20)


@contextlib.contextmanager
def path_lock(path_kind: str):
    if path_kind not in ("action", "joint"):
        raise ValueError(f"invalid UVA path {path_kind}")
    model = "uva_joint" if path_kind == "joint" else "uva"
    lock_path = RESULT_ROOT / model / "full/closed_loop/paired/.supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def condition_root(path_kind: str, condition: str) -> Path:
    model = "uva_joint" if path_kind == "joint" else "uva"
    return RESULT_ROOT / model / "full/closed_loop/paired" / condition


def require_complete(path_kind: str, condition: str) -> dict:
    result = audit(condition_root(path_kind, condition))
    if (result.get("complete") is not True
            or result.get("episodes_cached") != 500
            or result.get("evidence_tier") != "full_paired_condition"):
        raise RuntimeError(f"UVA {path_kind} {condition} is not a full audited condition: {result}")
    return result


def condition_complete(path_kind: str, condition: str) -> bool:
    try:
        require_complete(path_kind, condition)
    except Exception:
        return False
    return True


def run_condition(path_kind: str, condition: str) -> None:
    output = condition_root(path_kind, condition)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), "-u", str(RUNNER),
        "--path", path_kind,
        "--condition", condition,
        "--output", str(output),
        "--episodes-per-task", "50",
        "--task-ids", *[str(value) for value in range(10)],
    ]
    environment = os.environ.copy()
    environment.update({
        "MUJOCO_GL": "egl",
        "LIBERO_CONFIG_PATH": "/workspace/assets/uva/libero_config",
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    with (output / "worker.log").open("a") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"UVA {path_kind} {condition} exited with {completed.returncode}; see {output / 'worker.log'}"
        )
    require_complete(path_kind, condition)


def run_condition_resumable(path_kind: str, condition: str) -> None:
    failures = []
    output = condition_root(path_kind, condition)
    for attempt in range(1, MAX_CONDITION_ATTEMPTS + 1):
        try:
            run_condition(path_kind, condition)
            return
        except Exception as error:
            try:
                require_complete(path_kind, condition)
                return
            except Exception:
                pass
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            (output / "supervisor_retries.json").write_text(json.dumps({
                "schema_version": 1, "path": path_kind, "condition": condition,
                "maximum_attempts": MAX_CONDITION_ATTEMPTS,
                "attempts_failed": len(failures), "failures": failures,
                "append_resume_safe": True,
            }, indent=2) + "\n")
            if attempt == MAX_CONDITION_ATTEMPTS:
                raise
            delay = min(60, 5 * attempt)
            print(
                f"UVA {path_kind} {condition} attempt {attempt} failed; preserving caches "
                f"and retrying in {delay}s: {error}", flush=True,
            )
            time.sleep(delay)


def write_state(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def write_if_absent_or_identical(path: Path, value: dict) -> None:
    content = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.is_file():
        if path.read_text() != content:
            raise RuntimeError(f"existing UVA task manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def finalize_path(path_kind: str) -> None:
    model_id = "uva_joint" if path_kind == "joint" else "uva"
    paired = RESULT_ROOT / model_id / "full/closed_loop/paired"
    task_manifest = paired / "evaluator_task_manifest.json"
    write_if_absent_or_identical(task_manifest, {
        "schema_version": 1,
        "task_suite": "libero_10",
        "task_ids": list(range(10)),
        "task_descriptions": [
            "put both the alphabet soup and the tomato sauce in the basket",
            "put both the cream cheese box and the butter in the basket",
            "turn on the stove and put the moka pot on it",
            "put the black bowl in the bottom drawer of the cabinet and close it",
            "put the white mug on the left plate and put the yellow and white mug on the right plate",
            "pick up the book and place it in the back compartment of the caddy",
            "put the white mug on the plate and put the chocolate pudding to the right of the plate",
            "put both the alphabet soup and the cream cheese box in the basket",
            "put both moka pots on the stove",
            "put the yellow and white mug in the microwave and close it",
        ],
        "initializations_per_task": 50,
        "checkpoint_weight_path": str(CHECKPOINT.resolve()),
        "checkpoint_weight_sha256": CHECKPOINT_SHA256,
        "checkpoint_repository_revision": "fe3dec81af565fbb2d686072722f12340be31293",
    })
    pairing = paired / "reset_pairing_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_lerobot_reset_pairing.py",
        "--root", str(paired), "--output", str(pairing),
    ], cwd=ROOT, check=True)
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError(f"UVA {path_kind} physical reset/noise pairing failed")
    subprocess.run([
        sys.executable, "scripts/compress_preserved_rollouts.py",
        "--root", str(paired), "--allow-fully-paired", "--expected-files", "1260",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(paired), "--gzip-episodes",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/finalize_lerobot_paired_record.py",
        "--model-id", model_id,
        "--paired-root", str(paired),
        "--attempt-record", str(paired.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(CHECKPOINT),
        "--checkpoint-sha256", CHECKPOINT_SHA256,
        "--source-evidence", str(SOURCE_EVIDENCE),
        "--official-evaluator-source", str(OFFICIAL_EVALUATOR),
        "--evaluator-task-manifest", str(task_manifest),
    ], cwd=ROOT, check=True)
    print(f"UVA {path_kind} paired campaign independently verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-native-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.active_native_pid <= 1 or args.poll_seconds < 1:
        raise ValueError("invalid UVA continuation PID or poll interval")

    state_path = RESULT_ROOT / "uva/full/closed_loop/continuation_state.json"
    while pid_alive(args.active_native_pid):
        write_state(state_path, {
            "schema_version": 1,
            "state": "waiting_for_action_native_singleton",
            "active_pid": args.active_native_pid,
            "updated_unix": time.time(),
        })
        time.sleep(args.poll_seconds)
    with path_lock("action"):
        join_external_condition("action", "native_singleton")
        try:
            require_complete("action", "native_singleton")
        except Exception:
            run_condition_resumable("action", "native_singleton")

        for condition in CONDITIONS[1:]:
            if condition_complete("action", condition):
                print(f"Already verified UVA action {condition}; preserving evidence", flush=True)
                continue
            join_external_condition("action", condition)
            if condition_complete("action", condition):
                continue
            write_state(state_path, {
                "schema_version": 1,
                "state": "running",
                "path": "action",
                "condition": condition,
                "updated_unix": time.time(),
            })
            run_condition_resumable("action", condition)
        finalize_path("action")

    with path_lock("joint"):
        for condition in CONDITIONS:
            if condition_complete("joint", condition):
                print(f"Already verified UVA joint {condition}; preserving evidence", flush=True)
                continue
            join_external_condition("joint", condition)
            if condition_complete("joint", condition):
                continue
            write_state(state_path, {
                "schema_version": 1,
                "state": "running",
                "path": "joint",
                "condition": condition,
                "updated_unix": time.time(),
            })
            run_condition_resumable("joint", condition)
        finalize_path("joint")

    write_state(state_path, {
        "schema_version": 1,
        "state": "all_eight_conditions_audited",
        "updated_unix": time.time(),
    })


if __name__ == "__main__":
    main()

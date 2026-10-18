"""Run both full four-condition InternVLA LIBERO campaigns after smoke probes."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/workspace/models/internvla-a-series")
SERVER_PYTHON = Path("/workspace/envs/internvla_a15/bin/python")
CLIENT_PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
CHECKPOINT = Path("/workspace/assets/internvla_a15/checkpoint")
WEIGHT = CHECKPOINT / "model.safetensors"
WEIGHT_SHA256 = "222b6ed5dbd3224f441ad2f9165838a890aad5934282209fb635ca4daf073de0"
VLM = Path("/workspace/assets/internvla_a15/qwen")
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
BACKENDS = ("standard", "optimized")
MAX_CONDITION_ATTEMPTS = 20


def _process_arguments(proc: Path) -> tuple[list[str], str] | None:
    try:
        arguments = [
            part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0")
            if part
        ]
        state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None
    return arguments, state


def condition_worker_pids(
    backend: str, condition: str, kind: str, proc_root: Path = Path("/proc"),
) -> list[int]:
    """Find exact live InternVLA client or server processes for one condition."""
    if backend not in BACKENDS or condition not in CONDITIONS or kind not in ("client", "server"):
        raise ValueError("invalid InternVLA worker identity")
    script = f"run_internvla_paired_{kind}.py"
    expected_output = str((
        ROOT / f"results/cross_model/internvla_a15_{backend}/full/closed_loop/paired/{condition}"
    ).resolve())
    matches: list[int] = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        observed = _process_arguments(proc)
        if observed is None:
            continue
        arguments, state = observed
        if state == "Z" or not any(argument.endswith(script) for argument in arguments):
            continue
        try:
            observed_backend = arguments[arguments.index("--backend") + 1]
            observed_condition = arguments[arguments.index("--condition") + 1]
            observed_output = (
                str(Path(arguments[arguments.index("--output") + 1]).resolve())
                if kind == "client" else expected_output
            )
        except (ValueError, IndexError):
            continue
        if (
            observed_backend == backend
            and observed_condition == condition
            and observed_output == expected_output
        ):
            matches.append(int(proc.name))
    return sorted(matches)


def _stop_adopted_server(pid: int) -> None:
    if not Path(f"/proc/{pid}").exists():
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(150):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def join_external_condition(backend: str, condition: str) -> None:
    announced: tuple[int, ...] | None = None
    while True:
        clients = tuple(condition_worker_pids(backend, condition, "client"))
        if len(clients) > 1:
            raise RuntimeError(
                f"multiple InternVLA {backend} {condition} clients are writing one cache: {clients}"
            )
        if not clients:
            break
        if clients != announced:
            print(
                f"Joining external InternVLA {backend} {condition} client {clients[0]}; "
                "a duplicate writer will not be launched",
                flush=True,
            )
            announced = clients
        time.sleep(20)
    servers = condition_worker_pids(backend, condition, "server")
    if len(servers) > 1:
        raise RuntimeError(
            f"multiple InternVLA {backend} {condition} servers remain: {servers}"
        )
    for pid in servers:
        _stop_adopted_server(pid)


@contextlib.contextmanager
def backend_lock(backend: str):
    if backend not in BACKENDS:
        raise ValueError(f"invalid InternVLA backend {backend}")
    lock_path = ROOT / f"results/cross_model/internvla_a15_{backend}/full/closed_loop/paired/.supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wait_for_probe_worker(pid: int) -> dict:
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
        if "continue_internvla_probes_after_smolvla.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies the InternVLA probe worker")
        time.sleep(20)
    return {
        backend: json.loads((ROOT / f"results/cross_model/internvla_a15_{backend}/full/closed_loop/evaluator_attempt.json").read_text())
        if (ROOT / f"results/cross_model/internvla_a15_{backend}/full/closed_loop/evaluator_attempt.json").is_file()
        else {}
        for backend in BACKENDS
    }


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_server(process: subprocess.Popen, port: int, log: Path, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"paired InternVLA server exited {process.returncode}; see {log}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"paired InternVLA server startup timed out; see {log}")


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)


def rollout_paths(path: Path) -> list[Path]:
    return sorted((*path.glob("*.json"), *path.glob("*.json.gz")))


def read_rollout(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def validate_condition(backend: str, condition: str) -> bool:
    folder = ROOT / f"results/cross_model/internvla_a15_{backend}/full/closed_loop/paired/{condition}"
    manifest_path = folder / "condition_manifest.json"
    info_path = folder / "official/eval_info.json"
    if not manifest_path.is_file() or not info_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    required_batch = 1 if condition.endswith("singleton") else 4
    if (manifest.get("model") != f"internvla_a15_{backend}"
            or manifest.get("backend") != backend or manifest.get("condition") != condition
            or manifest.get("batch_size") != required_batch
            or manifest.get("invariant_ops") != condition.startswith("patched")
            or manifest.get("episodes_per_task") != 50
            or manifest.get("checkpoint_sha256") != WEIGHT_SHA256
            or manifest.get("genuine_batched_policy_call") is not True
            or manifest.get("request_local_noise") is not True):
        return False
    rows = {}
    for path in rollout_paths(folder / "rollouts"):
        value = read_rollout(path)
        for episode in value.get("episodes", []):
            key = (episode.get("task"), episode.get("initialization_id"))
            if key in rows:
                return False
            rows[key] = episode
    expected = {(task, initialization) for task in range(10) for initialization in range(50)}
    if set(rows) != expected:
        return False
    if any(
        not isinstance(row.get("success"), bool)
        or not row.get("trajectory") or not row.get("action_sequence")
        or len(row.get("initial_observation_sha256", "")) != 64
        or len(row.get("first_noise_sha256", "")) != 64
        for row in rows.values()
    ):
        return False
    info = json.loads(info_path.read_text())
    per_task = {row.get("task_id"): row for row in info.get("per_task", [])}
    if set(per_task) != set(range(10)) or info.get("overall", {}).get("n_episodes") != 500:
        return False
    return all(
        per_task[task].get("metrics", {}).get("successes")
        == [rows[task, initialization]["success"] for initialization in range(50)]
        for task in range(10)
    )


def run_condition(backend: str, condition: str) -> None:
    folder = ROOT / f"results/cross_model/internvla_a15_{backend}/full/closed_loop/paired/{condition}"
    folder.mkdir(parents=True, exist_ok=True)
    port = free_local_port()
    server_log = folder / "server.log"
    client_log = folder / "client.log"
    server_command = [
        str(SERVER_PYTHON), "-u", "scripts/run_internvla_paired_server.py",
        "--backend", backend, "--condition", condition,
        "--checkpoint", str(CHECKPOINT), "--vlm", str(VLM),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    client_command = [
        str(CLIENT_PYTHON), "-u", "scripts/run_internvla_paired_client.py",
        "--backend", backend, "--condition", condition,
        "--host", "127.0.0.1", "--port", str(port),
        "--output", str(folder), "--checkpoint", str(CHECKPOINT), "--vlm", str(VLM),
        "--episodes-per-task", "50", "--seed", "20261005",
        "--num-steps-wait", "10", "--replan-steps", "8",
    ]
    server_environment = os.environ.copy()
    server_environment.update({
        "PYTHONPATH": f"{ROOT}:{MODEL_ROOT}:{MODEL_ROOT / 'src'}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    client_environment = os.environ.copy()
    client_environment.update({
        "PYTHONPATH": f"{ROOT}:{MODEL_ROOT}:{MODEL_ROOT / 'src'}",
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "TOKENIZERS_PARALLELISM": "false",
    })
    print(f"Starting InternVLA {backend} {condition}; logs={server_log},{client_log}", flush=True)
    server = None
    try:
        with server_log.open("a") as server_stream:
            server = subprocess.Popen(
                server_command, cwd=ROOT, env=server_environment,
                stdout=server_stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
            wait_for_server(server, port, server_log)
            with client_log.open("a") as client_stream:
                result = subprocess.run(
                    client_command, cwd=ROOT, env=client_environment,
                    stdout=client_stream, stderr=subprocess.STDOUT, check=False,
                )
        if result.returncode != 0:
            raise RuntimeError(
                f"InternVLA {backend} {condition} client exited {result.returncode}; see {client_log}"
            )
    finally:
        if server is not None:
            stop_process_group(server)
    if not validate_condition(backend, condition):
        raise RuntimeError(f"InternVLA {backend} {condition} lacks 500 verified raw episodes")
    print(f"Verified InternVLA {backend} {condition}: 500 episodes", flush=True)


def run_condition_resumable(backend: str, condition: str) -> None:
    failures = []
    identifier = f"internvla_a15_{backend}"
    folder = ROOT / f"results/cross_model/{identifier}/full/closed_loop/paired/{condition}"
    for attempt in range(1, MAX_CONDITION_ATTEMPTS + 1):
        try:
            run_condition(backend, condition)
            return
        except Exception as error:
            if validate_condition(backend, condition):
                return
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            (folder / "supervisor_retries.json").write_text(json.dumps({
                "schema_version": 1, "backend": backend, "condition": condition,
                "maximum_attempts": MAX_CONDITION_ATTEMPTS,
                "attempts_failed": len(failures), "failures": failures,
                "append_resume_safe": True,
            }, indent=2) + "\n")
            if attempt == MAX_CONDITION_ATTEMPTS:
                raise
            delay = min(60, 5 * attempt)
            print(
                f"InternVLA {backend} {condition} attempt {attempt} failed; "
                f"preserving caches and retrying in {delay}s: {error}", flush=True,
            )
            time.sleep(delay)


def finalize_backend(backend: str) -> None:
    identifier = f"internvla_a15_{backend}"
    paired = ROOT / f"results/cross_model/{identifier}/full/closed_loop/paired"
    pairing = paired / "reset_pairing_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_lerobot_reset_pairing.py",
        "--root", str(paired), "--output", str(pairing),
    ], cwd=ROOT, check=True)
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError(f"InternVLA {backend} physical reset/noise pairing failed")
    subprocess.run([
        sys.executable, "scripts/compress_preserved_rollouts.py",
        "--root", str(paired), "--allow-fully-paired",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(paired), "--gzip-episodes",
    ], cwd=ROOT, check=True)
    task_manifest = paired / "evaluator_task_manifest.json"
    expected_task_manifest = {
        "schema_version": 1,
        "task_suite": "libero_10",
        "task_ids": list(range(10)),
        "episodes_per_task": 50,
    }
    expected_task_text = json.dumps(expected_task_manifest, sort_keys=True, indent=2) + "\n"
    if task_manifest.is_file() and task_manifest.read_text() != expected_task_text:
        raise RuntimeError(f"InternVLA {backend} evaluator task manifest changed")
    if not task_manifest.is_file():
        task_manifest.write_text(expected_task_text)
    subprocess.run([
        sys.executable, "scripts/finalize_lerobot_paired_record.py",
        "--model-id", identifier,
        "--paired-root", str(paired),
        "--attempt-record", str(paired.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(WEIGHT), "--checkpoint-sha256", WEIGHT_SHA256,
        "--source-evidence", str(MODEL_ROOT / "evaluation/LIBERO/README.md"),
        "--official-evaluator-source",
        str(MODEL_ROOT / "evaluation/LIBERO/eval_libero_server_client.py"),
        "--evaluator-task-manifest", str(task_manifest),
    ], cwd=ROOT, check=True)
    print(f"InternVLA {backend} paired campaign independently verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-worker-pid", type=int, required=True)
    args = parser.parse_args()
    if args.probe_worker_pid <= 0:
        raise ValueError("invalid InternVLA probe worker PID")
    predecessors = wait_for_probe_worker(args.probe_worker_pid)
    print(
        "InternVLA probes ended; "
        + ", ".join(f"{backend}={record.get('status', 'missing')}" for backend, record in predecessors.items())
        + "; paired campaigns proceed independently",
        flush=True,
    )
    for backend in BACKENDS:
        with backend_lock(backend):
            for condition in CONDITIONS:
                if validate_condition(backend, condition):
                    print(f"Already verified InternVLA {backend} {condition}; preserving evidence", flush=True)
                    continue
                join_external_condition(backend, condition)
                if validate_condition(backend, condition):
                    print(
                        f"Verified externally completed InternVLA {backend} {condition}; "
                        "preserving evidence",
                        flush=True,
                    )
                    continue
                run_condition_resumable(backend, condition)
            finalize_backend(backend)


if __name__ == "__main__":
    main()

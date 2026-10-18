"""Run SpatialVLA's four paired SimplerEnv conditions after queued RDT work."""

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
MODEL_ROOT = Path("/workspace/models/simplerenv-openvla")
TRANSPORT_ROOT = Path("/workspace/models/internvla-a-series")
CHECKPOINT = Path("/workspace/assets/spatialvla/checkpoint")
SERVER_PYTHON = Path("/workspace/envs/spatialvla/bin/python")
CLIENT_PYTHON = Path("/workspace/envs/simpler_eval/bin/python")
PAIRED = ROOT / "results/cross_model/spatialvla/full/closed_loop/paired"
TASK_MANIFEST = PAIRED / "evaluator_task_manifest.json"
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
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
    condition: str, kind: str, proc_root: Path = Path("/proc"),
) -> list[int]:
    """Find exact live SpatialVLA client or server processes for one condition."""
    if condition not in CONDITIONS or kind not in ("client", "server"):
        raise ValueError("invalid SpatialVLA worker identity")
    script = f"run_spatialvla_paired_{kind}.py"
    expected_output = str((PAIRED / condition).resolve())
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
            observed_condition = arguments[arguments.index("--condition") + 1]
            observed_output = (
                str(Path(arguments[arguments.index("--output") + 1]).resolve())
                if kind == "client" else expected_output
            )
        except (ValueError, IndexError):
            continue
        if observed_condition == condition and observed_output == expected_output:
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


def join_external_condition(condition: str) -> None:
    announced: tuple[int, ...] | None = None
    while True:
        clients = tuple(condition_worker_pids(condition, "client"))
        if len(clients) > 1:
            raise RuntimeError(
                f"multiple SpatialVLA {condition} clients are writing one cache: {clients}"
            )
        if not clients:
            break
        if clients != announced:
            print(
                f"Joining external SpatialVLA {condition} client {clients[0]}; "
                "a duplicate writer will not be launched",
                flush=True,
            )
            announced = clients
        time.sleep(20)
    servers = condition_worker_pids(condition, "server")
    if len(servers) > 1:
        raise RuntimeError(f"multiple SpatialVLA {condition} servers remain: {servers}")
    for pid in servers:
        _stop_adopted_server(pid)


@contextlib.contextmanager
def supervisor_lock():
    lock_path = PAIRED / ".supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wait_for_rdt(pid: int) -> None:
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
        if "continue_rdt_paired_after_wsa.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies the RDT paired worker")
        time.sleep(20)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_server(process: subprocess.Popen, port: int, log: Path, timeout: int = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SpatialVLA server exited {process.returncode}; see {log}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"SpatialVLA server startup timed out; see {log}")


def stop_group(process: subprocess.Popen) -> None:
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


def read_rollout(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def validate_condition(condition: str) -> bool:
    folder = PAIRED / condition
    manifest_path = folder / "condition_manifest.json"
    info_path = folder / "official/eval_info.json"
    if not manifest_path.is_file() or not info_path.is_file() or not TASK_MANIFEST.is_file():
        return False
    task_manifest = json.loads(TASK_MANIFEST.read_text())
    manifest = json.loads(manifest_path.read_text())
    batch = 1 if condition.endswith("singleton") else 4
    if (manifest.get("model") != "spatialvla" or manifest.get("condition") != condition
            or manifest.get("batch_size") != batch
            or manifest.get("invariant_ops") != condition.startswith("patched")
            or manifest.get("task_suite") != task_manifest.get("task_suite")
            or manifest.get("task_count") != 12
            or manifest.get("planned_initializations") != 496):
        return False
    rows = {}
    for path in sorted((*folder.glob("rollouts/*.json"), *folder.glob("rollouts/*.json.gz"))):
        for episode in read_rollout(path).get("episodes", []):
            key = (episode.get("task"), episode.get("initialization_id"))
            if key in rows:
                return False
            rows[key] = episode
    expected = {
        (task["task_id"], initialization)
        for task in task_manifest["tasks"]
        for initialization in range(task["initializations"])
    }
    info = json.loads(info_path.read_text())
    return (set(rows) == expected and info.get("overall", {}).get("n_episodes") == 496
            and info.get("overall", {}).get("n_success")
            == sum(bool(episode.get("success")) for episode in rows.values()))


def verify_task_manifest() -> None:
    value = json.loads(TASK_MANIFEST.read_text())
    if (value.get("repository_revision") != "ccfe3809766839a2fcfb7a3d3c9abff585189188"
            or value.get("simulator_revision") != "cd45dd27dc6bb26d048cb6570cdab4e3f935cc37"
            or value.get("task_count") != 12
            or value.get("initializations_per_condition") != 496):
        raise RuntimeError("SpatialVLA pinned task manifest differs")
    import hashlib
    for item in value.get("source_files", []):
        path = Path(item["path"]); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not path.is_file() or digest != item["sha256"]:
            raise RuntimeError(f"SpatialVLA pinned evaluator source differs: {path}")


def run_condition(condition: str) -> None:
    folder = PAIRED / condition; folder.mkdir(parents=True, exist_ok=True)
    port = free_port(); server_log = folder / "server.log"; client_log = folder / "client.log"
    server_command = [
        str(SERVER_PYTHON), "-u", "scripts/run_spatialvla_paired_server.py",
        "--condition", condition, "--checkpoint", str(CHECKPOINT),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    client_command = [
        str(CLIENT_PYTHON), "-u", "scripts/run_spatialvla_paired_client.py",
        "--condition", condition, "--host", "127.0.0.1", "--port", str(port),
        "--output", str(folder),
    ]
    common = os.environ.copy(); common.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
    })
    server_env = common.copy(); server_env["PYTHONPATH"] = f"{ROOT}:{TRANSPORT_ROOT}"
    client_env = common.copy(); client_env["PYTHONPATH"] = f"{ROOT}:{MODEL_ROOT}:{TRANSPORT_ROOT}"
    print(f"Starting SpatialVLA {condition}; logs={server_log},{client_log}", flush=True)
    server = None
    try:
        with server_log.open("a") as stream:
            server = subprocess.Popen(
                server_command, cwd=ROOT, env=server_env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            wait_server(server, port, server_log)
            with client_log.open("a") as stream:
                result = subprocess.run(
                    client_command, cwd=ROOT, env=client_env,
                    stdout=stream, stderr=subprocess.STDOUT, check=False,
                )
        if result.returncode != 0:
            raise RuntimeError(f"SpatialVLA {condition} exited {result.returncode}; see {client_log}")
    finally:
        if server is not None:
            stop_group(server)
    if not validate_condition(condition):
        raise RuntimeError(f"SpatialVLA {condition} lacks 496 verified episodes")


def run_condition_resumable(condition: str) -> None:
    failures = []
    for attempt in range(1, MAX_CONDITION_ATTEMPTS + 1):
        try:
            run_condition(condition)
            return
        except Exception as error:
            if validate_condition(condition):
                return
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            (PAIRED / condition / "supervisor_retries.json").write_text(json.dumps({
                "schema_version": 1, "condition": condition,
                "maximum_attempts": MAX_CONDITION_ATTEMPTS,
                "attempts_failed": len(failures), "failures": failures,
                "append_resume_safe": True,
            }, indent=2) + "\n")
            if attempt == MAX_CONDITION_ATTEMPTS:
                raise
            delay = min(60, 5 * attempt)
            print(
                f"SpatialVLA {condition} attempt {attempt} failed; preserving caches and "
                f"retrying in {delay}s: {error}", flush=True,
            )
            time.sleep(delay)


def finalize() -> None:
    pairing = PAIRED / "reset_pairing_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_spatialvla_reset_pairing.py",
        "--root", str(PAIRED), "--task-manifest", str(TASK_MANIFEST),
        "--output", str(pairing),
    ], cwd=ROOT, check=True)
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError("SpatialVLA physical configuration pairing failed")
    subprocess.run([
        sys.executable, "scripts/compress_preserved_rollouts.py", "--root", str(PAIRED),
        "--allow-fully-paired", "--expected-files", "1248",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/audit_spatialvla_reset_pairing.py",
        "--root", str(PAIRED), "--task-manifest", str(TASK_MANIFEST),
        "--output", str(pairing),
    ], cwd=ROOT, check=True)
    # Reduction must follow the lossless storage transform so every retained
    # source_rollout_path names the authoritative .json.gz artifact.
    subprocess.run([
        sys.executable, "scripts/summarize_spatialvla_paired.py",
        "--root", str(PAIRED), "--task-manifest", str(TASK_MANIFEST),
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/finalize_lerobot_paired_record.py",
        "--model-id", "spatialvla", "--paired-root", str(PAIRED),
        "--attempt-record", str(PAIRED.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(CHECKPOINT / "model.safetensors.index.json"),
        "--checkpoint-sha256", "6c351156136b2452db86d3a426ff30adc9d11363b43f46fa69fe79d76b6d5c79",
        "--source-evidence", "/workspace/models/spatialvla/README.md",
        "--official-evaluator-source",
        str(MODEL_ROOT / "simpler_env/evaluation/maniskill2_evaluator.py"),
        "--evaluator-task-manifest", str(TASK_MANIFEST),
    ], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdt-paired-pid", type=int, required=True)
    args = parser.parse_args()
    if args.rdt_paired_pid <= 0:
        raise ValueError("invalid RDT paired worker PID")
    wait_for_rdt(args.rdt_paired_pid); verify_task_manifest()
    with supervisor_lock():
        for condition in CONDITIONS:
            if validate_condition(condition):
                print(f"Already verified SpatialVLA {condition}; preserving evidence", flush=True)
                continue
            join_external_condition(condition)
            if not validate_condition(condition):
                run_condition_resumable(condition)
        finalize()
    print("SpatialVLA paired 12-task SimplerEnv campaign independently verified", flush=True)


if __name__ == "__main__":
    main()

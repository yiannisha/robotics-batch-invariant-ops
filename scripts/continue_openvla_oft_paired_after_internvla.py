"""Run full paired OpenVLA-OFT LIBERO-Spatial after InternVLA campaigns."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/workspace/models/openvla-oft")
TRANSPORT_ROOT = Path("/workspace/models/internvla-a-series")
CHECKPOINT = Path("/workspace/assets/openvla_oft/checkpoint")
SERVER_PYTHON = Path("/workspace/envs/openvla_oft/bin/python")
CLIENT_PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
PAIRED = ROOT / "results/cross_model/openvla_oft/full/closed_loop/paired"
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
MAX_CONDITION_ATTEMPTS = 20
INDEX = CHECKPOINT / "model.safetensors.index.json"
INDEX_SHA256 = "ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659"
CHECKPOINT_HASHES = {
    "model-00001-of-00004.safetensors": "2809bd7be9422315c5ecbe91eea612f5b02925f37e0051728ad45bc993c79251",
    "model-00002-of-00004.safetensors": "a00a7c5f2b6586ccfc89c693a9c36f3552ff455ff2b1bfea3e92642e4cd2b6d3",
    "model-00003-of-00004.safetensors": "a894b7230a08b471af55c57dd7385fd3b51fae2c4d9307a36a8017ef57abf22a",
    "model-00004-of-00004.safetensors": "a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60",
    "action_head--150000_checkpoint.pt": "809858636cf0a65009dd567d2f4e116249442790f02b8fe31f24500ea6118908",
    "proprio_projector--150000_checkpoint.pt": "438d28e81e125166d0771762424aaf017de3a8daaebde06a5cb71157e62b3bf3",
    "config.json": "bcb688a66f3e94a42311b77d37b7e1484886701f7b2aa8e36d165dfccac92ae6",
    "dataset_statistics.json": "3f8b7854c8634ea6ccfd6ef6b8f983e3b30b067c2cdaf690b152c0899d5d9282",
}


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
    """Find exact live OpenVLA-OFT client or server processes for one condition."""
    if condition not in CONDITIONS or kind not in ("client", "server"):
        raise ValueError("invalid OpenVLA-OFT worker identity")
    script = f"run_openvla_oft_paired_{kind}.py"
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
                f"multiple OpenVLA-OFT {condition} clients are writing one cache: {clients}"
            )
        if not clients:
            break
        if clients != announced:
            print(
                f"Joining external OpenVLA-OFT {condition} client {clients[0]}; "
                "a duplicate writer will not be launched",
                flush=True,
            )
            announced = clients
        time.sleep(20)
    servers = condition_worker_pids(condition, "server")
    if len(servers) > 1:
        raise RuntimeError(f"multiple OpenVLA-OFT {condition} servers remain: {servers}")
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
TASK_DESCRIPTIONS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_internvla(pid: int) -> None:
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
        if "continue_internvla_paired_after_probes.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies InternVLA paired worker")
        time.sleep(20)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_server(process: subprocess.Popen, port: int, log: Path, timeout: int = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OpenVLA-OFT server exited {process.returncode}; see {log}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"OpenVLA-OFT server startup timed out; see {log}")


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
    manifest_path, info_path = folder / "condition_manifest.json", folder / "official/eval_info.json"
    if not manifest_path.is_file() or not info_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    required_batch = 1 if condition.endswith("singleton") else 4
    if (manifest.get("model") != "openvla_oft" or manifest.get("condition") != condition
            or manifest.get("batch_size") != required_batch
            or manifest.get("invariant_ops") != condition.startswith("patched")
            or manifest.get("task_suite") != "libero_spatial"
            or manifest.get("episodes_per_task") != 50
            or manifest.get("genuine_batched_policy_call") is not True):
        return False
    rows = {}
    paths = sorted((*folder.glob("rollouts/*.json"), *folder.glob("rollouts/*.json.gz")))
    for path in paths:
        for episode in read_rollout(path).get("episodes", []):
            key = (episode.get("task"), episode.get("initialization_id"))
            if key in rows:
                return False
            rows[key] = episode
    expected = {(task, initialization) for task in range(10) for initialization in range(50)}
    if set(rows) != expected:
        return False
    info = json.loads(info_path.read_text())
    per_task = {row.get("task_id"): row for row in info.get("per_task", [])}
    return (set(per_task) == set(range(10)) and info.get("overall", {}).get("n_episodes") == 500
            and all(per_task[t]["metrics"]["successes"] == [rows[t, i]["success"] for i in range(50)]
                    for t in range(10)))


def make_task_manifest() -> Path:
    for name, expected in CHECKPOINT_HASHES.items():
        if sha256(CHECKPOINT / name) != expected:
            raise ValueError(f"OpenVLA-OFT checkpoint changed: {name}")
    if sha256(INDEX) != INDEX_SHA256:
        raise ValueError("OpenVLA-OFT checkpoint index changed")
    value = {
        "schema_version": 1,
        "task_suite": "libero_spatial",
        "task_ids": list(range(10)),
        "task_descriptions": TASK_DESCRIPTIONS,
        "initializations_per_task": 50,
        "checkpoint_index_path": str(INDEX.resolve()),
        "checkpoint_index_sha256": INDEX_SHA256,
        "checkpoint_files": [
            {"path": str((CHECKPOINT / name).resolve()), "sha256": digest}
            for name, digest in CHECKPOINT_HASHES.items()
        ],
    }
    path = PAIRED / "evaluator_task_manifest.json"
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.is_file() and path.read_text() != text:
        raise RuntimeError("OpenVLA-OFT task/checkpoint manifest changed")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def run_condition(condition: str, task_manifest: Path) -> None:
    folder = PAIRED / condition
    folder.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server_log, client_log = folder / "server.log", folder / "client.log"
    server_command = [
        str(SERVER_PYTHON), "-u", "scripts/run_openvla_oft_paired_server.py",
        "--condition", condition, "--checkpoint", str(CHECKPOINT),
        "--task-manifest", str(task_manifest), "--host", "127.0.0.1", "--port", str(port),
    ]
    client_command = [
        str(CLIENT_PYTHON), "-u", "scripts/run_openvla_oft_paired_client.py",
        "--condition", condition, "--host", "127.0.0.1", "--port", str(port),
        "--output", str(folder), "--task-manifest", str(task_manifest),
        "--seed", "20261005", "--num-steps-wait", "10",
    ]
    server_env = os.environ.copy()
    server_env.update({
        "PYTHONPATH": f"{ROOT}:{MODEL_ROOT}:{TRANSPORT_ROOT}:{TRANSPORT_ROOT / 'src'}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    client_env = os.environ.copy()
    client_env.update({
        "PYTHONPATH": f"{ROOT}:{TRANSPORT_ROOT}:{TRANSPORT_ROOT / 'src'}",
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
    })
    print(f"Starting OpenVLA-OFT {condition}; logs={server_log},{client_log}", flush=True)
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
            raise RuntimeError(f"OpenVLA-OFT {condition} exited {result.returncode}; see {client_log}")
    finally:
        if server is not None:
            stop_group(server)
    if not validate_condition(condition):
        raise RuntimeError(f"OpenVLA-OFT {condition} lacks 500 verified episodes")


def run_condition_resumable(condition: str, task_manifest: Path) -> None:
    failures = []
    for attempt in range(1, MAX_CONDITION_ATTEMPTS + 1):
        try:
            run_condition(condition, task_manifest)
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
                f"OpenVLA-OFT {condition} attempt {attempt} failed; preserving caches and "
                f"retrying in {delay}s: {error}", flush=True,
            )
            time.sleep(delay)


def finalize(task_manifest: Path) -> None:
    pairing = PAIRED / "reset_pairing_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_lerobot_reset_pairing.py",
        "--root", str(PAIRED), "--output", str(pairing),
    ], cwd=ROOT, check=True)
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError("OpenVLA-OFT physical reset/deterministic contract pairing failed")
    subprocess.run([
        sys.executable, "scripts/compress_preserved_rollouts.py",
        "--root", str(PAIRED), "--allow-fully-paired",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(PAIRED), "--gzip-episodes",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/finalize_lerobot_paired_record.py",
        "--model-id", "openvla_oft", "--paired-root", str(PAIRED),
        "--attempt-record", str(PAIRED.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(INDEX), "--checkpoint-sha256", INDEX_SHA256,
        "--source-evidence", str(MODEL_ROOT / "LIBERO.md"),
        "--official-evaluator-source", str(MODEL_ROOT / "experiments/robot/libero/run_libero_eval.py"),
        "--evaluator-task-manifest", str(task_manifest),
    ], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internvla-paired-pid", type=int, required=True)
    args = parser.parse_args()
    wait_for_internvla(args.internvla_paired_pid)
    task_manifest = make_task_manifest()
    with supervisor_lock():
        for condition in CONDITIONS:
            if validate_condition(condition):
                print(f"Already verified OpenVLA-OFT {condition}; preserving evidence", flush=True)
                continue
            join_external_condition(condition)
            if not validate_condition(condition):
                run_condition_resumable(condition, task_manifest)
        finalize(task_manifest)
    print("OpenVLA-OFT paired LIBERO-Spatial campaign independently verified", flush=True)


if __name__ == "__main__":
    main()

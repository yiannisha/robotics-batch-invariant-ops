"""Run full paired WSA-Base LIBERO-10 after the OpenVLA-OFT campaign."""

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
MODEL_ROOT = Path("/workspace/models/wsa")
TRANSPORT_ROOT = Path("/workspace/models/internvla-a-series")
ASSET_ROOT = Path("/workspace/assets/wsa")
CHECKPOINT = ASSET_ROOT / "checkpoint"
QWEN = Path("/workspace/assets/groot_n17/processor")
SERVER_PYTHON = Path("/workspace/envs/wsa/bin/python")
CLIENT_PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
PAIRED = ROOT / "results/cross_model/wsa_base/full/closed_loop/paired"
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
MAX_CONDITION_ATTEMPTS = 20
CHECKPOINT_SHA256 = "603eacbb1a7da80dd2d41cbbd814a833bd57dec30601da5fda35651d6dd46e67"
ARTIFACT_HASHES = {
    ASSET_ROOT / "checkpoint/model.safetensors": CHECKPOINT_SHA256,
    ASSET_ROOT / "checkpoint/config.json": "e0f3e9043e97bf84569c8f430a3e666417f00d454286e83289ef010e5e507b7e",
    ASSET_ROOT / "checkpoint/stats.json": "5f6b2728c416bf63908a802aef8c96885d31c6fbdd7e625ec1ca84d306361f3a",
    ASSET_ROOT / "checkpoint/train_config.json": "16d21b3fd4fa49601cf3bb31c8176f42b8dc14834ceb04d4d84dad2ffa48e548",
    ASSET_ROOT / "cosmos_tokenizer/encoder.jit": "84e4c12c10568faff20da744b51c45a6b1e4385af86e80ae56bf3d1b13cda413",
    ASSET_ROOT / "cosmos_tokenizer/decoder.jit": "8e50c647a04decc7e939670ef7b51bc0500e0ffb4836a306a93100b0b2e957de",
    ASSET_ROOT / "cosmos_tokenizer/config.json": "ebf911575616266497325c38329b4212b2dfe223a6ad6c9b578fcbd7f7552aa3",
    ASSET_ROOT / "cosmos_tokenizer/model_config.yaml": "69bc6fabb030be2ddba76c0a6944882348c3db0426baf31ced29ca02aeaba679",
    QWEN / "config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    QWEN / "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    QWEN / "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    QWEN / "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    QWEN / "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    QWEN / "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    QWEN / "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
TASK_DESCRIPTIONS = [
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
]


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
    """Find exact live WSA client or server processes for one condition."""
    if condition not in CONDITIONS or kind not in ("client", "server"):
        raise ValueError("invalid WSA worker identity")
    script = f"run_wsa_paired_{kind}.py"
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
                f"multiple WSA {condition} clients are writing one cache: {clients}"
            )
        if not clients:
            break
        if clients != announced:
            print(
                f"Joining external WSA {condition} client {clients[0]}; "
                "a duplicate writer will not be launched",
                flush=True,
            )
            announced = clients
        time.sleep(20)
    servers = condition_worker_pids(condition, "server")
    if len(servers) > 1:
        raise RuntimeError(f"multiple WSA {condition} servers remain: {servers}")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_openvla(pid: int) -> None:
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
        if "continue_openvla_oft_paired_after_internvla.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies OpenVLA-OFT paired worker")
        time.sleep(20)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_server(process: subprocess.Popen, port: int, log: Path, timeout: int = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"WSA server exited {process.returncode}; see {log}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"WSA server startup timed out; see {log}")


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
    if (manifest.get("model") != "wsa_base" or manifest.get("condition") != condition
            or manifest.get("batch_size") != required_batch
            or manifest.get("invariant_ops") != condition.startswith("patched")
            or manifest.get("task_suite") != "libero_10"
            or manifest.get("episodes_per_task") != 50
            or manifest.get("request_local_noise") is not True
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
            and all(per_task[task]["metrics"]["successes"] == [rows[task, i]["success"] for i in range(50)]
                    for task in range(10)))


def make_task_manifest() -> Path:
    records = []
    for path, expected in ARTIFACT_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"WSA exact artifact changed: {path}")
        records.append({"path": str(path.resolve()), "sha256": expected})
    value = {
        "schema_version": 1,
        "task_suite": "libero_10",
        "task_ids": list(range(10)),
        "task_descriptions": TASK_DESCRIPTIONS,
        "initializations_per_task": 50,
        "artifact_files": records,
        "checkpoint_files": records,
    }
    path = PAIRED / "evaluator_task_manifest.json"
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.is_file() and path.read_text() != text:
        raise RuntimeError("WSA task/checkpoint manifest changed")
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
        str(SERVER_PYTHON), "-u", "scripts/run_wsa_paired_server.py",
        "--condition", condition, "--asset-root", str(ASSET_ROOT),
        "--qwen-metadata", str(QWEN), "--task-manifest", str(task_manifest),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    client_command = [
        str(CLIENT_PYTHON), "-u", "scripts/run_wsa_paired_client.py",
        "--condition", condition, "--host", "127.0.0.1", "--port", str(port),
        "--output", str(folder), "--task-manifest", str(task_manifest),
        "--seed", "20261005", "--num-steps-wait", "10",
    ]
    server_env = os.environ.copy()
    server_env.update({
        "PYTHONPATH": f"{ROOT}:{MODEL_ROOT / 'src'}:{MODEL_ROOT}:{TRANSPORT_ROOT}:{TRANSPORT_ROOT / 'src'}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    client_env = os.environ.copy()
    client_env.update({
        "PYTHONPATH": f"{ROOT}:{TRANSPORT_ROOT}:{TRANSPORT_ROOT / 'src'}",
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
    })
    print(f"Starting WSA {condition}; logs={server_log},{client_log}", flush=True)
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
            raise RuntimeError(f"WSA {condition} exited {result.returncode}; see {client_log}")
    finally:
        if server is not None:
            stop_group(server)
    if not validate_condition(condition):
        raise RuntimeError(f"WSA {condition} lacks 500 verified episodes")


def run_condition_resumable(
    condition: str, task_manifest: Path, before_attempt=None,
) -> None:
    failures = []
    for attempt in range(1, MAX_CONDITION_ATTEMPTS + 1):
        try:
            if before_attempt is not None:
                before_attempt()
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
                f"WSA {condition} attempt {attempt} failed; preserving caches and "
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
        raise RuntimeError("WSA physical reset/noise pairing failed")
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
        "--model-id", "wsa_base", "--paired-root", str(PAIRED),
        "--attempt-record", str(PAIRED.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(CHECKPOINT / "model.safetensors"),
        "--checkpoint-sha256", CHECKPOINT_SHA256,
        "--source-evidence", str(MODEL_ROOT / "evaluation/Libero/README.md"),
        "--official-evaluator-source", str(MODEL_ROOT / "evaluation/Libero/inference.py"),
        "--evaluator-task-manifest", str(task_manifest),
    ], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-paired-pid", type=int, required=True)
    args = parser.parse_args()
    wait_for_openvla(args.openvla_paired_pid)
    task_manifest = make_task_manifest()
    with supervisor_lock():
        for condition in CONDITIONS:
            if validate_condition(condition):
                print(f"Already verified WSA {condition}; preserving evidence", flush=True)
                continue
            join_external_condition(condition)
            if not validate_condition(condition):
                run_condition_resumable(condition, task_manifest)
        finalize(task_manifest)
    print("WSA paired LIBERO-10 campaign independently verified", flush=True)


if __name__ == "__main__":
    main()

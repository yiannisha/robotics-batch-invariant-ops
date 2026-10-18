"""Run the resumable exact GR00T N1.7 paired LIBERO-10 campaign."""

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
MODEL_ROOT = Path("/workspace/models/isaac-groot")
CHECKPOINT = Path("/workspace/assets/groot_n17/repo/libero_10")
METADATA = Path("/workspace/assets/groot_n17/processor")
SERVER_PYTHON = Path("/workspace/envs/groot_n17/bin/python")
CLIENT_PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
PAIRED = ROOT / "results/cross_model/groot_n17/full/closed_loop/paired"
PROBE = ROOT / "results/cross_model/groot_n17/full/closed_loop/paired_probe"
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
MAX_CONDITION_ATTEMPTS = 20
MINIMUM_GPU_FREE_MIB = int(os.environ.get("GROOT_MINIMUM_GPU_FREE_MIB", "10000"))
GPU_CAPACITY_POLL_SECONDS = 30
GPU_CAPACITY_STABLE_POLLS = 3
ARTIFACT_HASHES = {
    CHECKPOINT / "model-00001-of-00002.safetensors": "39b7beaadd9a06c87502f2b741a36f939b84941f37fef757c6177e54e34a5eff",
    CHECKPOINT / "model-00002-of-00002.safetensors": "8c4fa56f1b4f25a1a842c811e8a12d2bf6d77d73942f9a04f5b876a11bd6d703",
    CHECKPOINT / "model.safetensors.index.json": "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
    CHECKPOINT / "config.json": "ce763f74ad1cda2ca9e67491229cbc5810d83cae675cb4dad839a69a9731fd97",
    CHECKPOINT / "processor_config.json": "545bb44c7d8caaf5ac82df55b90831271a1469a4e48adcc9a5f469f064f4e121",
    CHECKPOINT / "statistics.json": "cbd3765acaec31fb65109304fa0c2ebd637732ddbe1b3cab0cc11bf244a498e1",
    METADATA / "config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    METADATA / "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    METADATA / "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    METADATA / "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    METADATA / "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    METADATA / "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    METADATA / "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
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
    """Find exact live full-campaign GR00T client or server processes."""
    if condition not in CONDITIONS or kind not in ("client", "server"):
        raise ValueError("invalid GR00T worker identity")
    script = f"run_groot_paired_{kind}.py"
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
                f"multiple GR00T {condition} clients are writing one cache: {clients}"
            )
        if not clients:
            break
        if clients != announced:
            print(
                f"Joining external GR00T {condition} client {clients[0]}; "
                "a duplicate writer will not be launched",
                flush=True,
            )
            announced = clients
        time.sleep(20)
    servers = condition_worker_pids(condition, "server")
    if len(servers) > 1:
        raise RuntimeError(f"multiple GR00T {condition} servers remain: {servers}")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def gpu_free_mib() -> int:
    """Return currently free memory on the single campaign GPU."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected one campaign GPU, observed free-memory values {values}")
    return values[0]


def wait_for_gpu_capacity(condition: str) -> None:
    """Avoid destructive retry churn when another evaluator owns GR00T's headroom."""
    last_announcement = 0.0
    stable_observations = 0
    while True:
        try:
            available = gpu_free_mib()
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
            available = -1
            detail = f"GPU capacity query failed: {error}"
        else:
            detail = (
                f"GPU free memory {available} MiB is below the "
                f"{MINIMUM_GPU_FREE_MIB} MiB GR00T admission floor"
            )
        if available >= MINIMUM_GPU_FREE_MIB:
            stable_observations += 1
        else:
            stable_observations = 0
        if stable_observations >= GPU_CAPACITY_STABLE_POLLS:
            print(
                f"GR00T {condition} admitted with {available} MiB GPU memory free "
                f"for {stable_observations} consecutive polls",
                flush=True,
            )
            return
        now = time.monotonic()
        if now - last_announcement >= 300 or last_announcement == 0.0:
            if stable_observations:
                detail = (
                    f"GPU free memory {available} MiB passed the admission floor for "
                    f"only {stable_observations}/{GPU_CAPACITY_STABLE_POLLS} stable polls"
                )
            print(
                f"Waiting to start GR00T {condition}: {detail}; preserving caches",
                flush=True,
            )
            last_announcement = now
        time.sleep(GPU_CAPACITY_POLL_SECONDS)


def wait_server(process: subprocess.Popen, port: int, log: Path, timeout: int = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"GR00T server exited {process.returncode}; see {log}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"GR00T server startup timed out; see {log}")


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


def make_task_manifest() -> Path:
    records = []
    for path, expected in ARTIFACT_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"GR00T exact artifact changed: {path}")
        records.append({"path": str(path.resolve()), "sha256": expected})
    value = {
        "schema_version": 1,
        "task_suite": "libero_10",
        "task_ids": list(range(10)),
        "task_descriptions": TASK_DESCRIPTIONS,
        "initializations_per_task": 50,
        "artifact_files": records,
        "checkpoint_files": records[:6],
    }
    path = PAIRED / "evaluator_task_manifest.json"
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.is_file() and path.read_text() != text:
        raise RuntimeError("GR00T task/checkpoint manifest changed")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def validate_condition(folder: Path, condition: str, expected: int) -> bool:
    manifest_path = folder / "condition_manifest.json"
    info_path = folder / "official/eval_info.json"
    if not manifest_path.is_file() or not info_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    required_batch = 1 if condition.endswith("singleton") else 4
    task_count = 10 if expected == 500 else 1
    episodes_per_task = 50 if expected == 500 else expected
    if (
        manifest.get("model") != "groot_n17"
        or manifest.get("condition") != condition
        or manifest.get("batch_size") != required_batch
        or manifest.get("invariant_ops") != condition.startswith("patched")
        or manifest.get("task_suite") != "libero_10"
        or manifest.get("task_count") != task_count
        or manifest.get("episodes_per_task") != episodes_per_task
        or manifest.get("request_local_noise") is not True
        or manifest.get("genuine_batched_policy_call") is not True
    ):
        return False
    rows = {}
    paths = sorted((folder / "rollouts").glob("*.json"))
    paths += sorted((folder / "rollouts").glob("*.json.gz"))
    for path in paths:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as stream:
                payload = json.load(stream)
        else:
            payload = json.loads(path.read_text())
        for episode in payload.get("episodes", []):
            key = (episode.get("task"), episode.get("initialization_id"))
            if key in rows:
                return False
            rows[key] = episode
    expected_keys = {
        (task, initialization)
        for task in range(task_count)
        for initialization in range(episodes_per_task)
    }
    info = json.loads(info_path.read_text())
    return set(rows) == expected_keys and info.get("overall", {}).get("n_episodes") == expected


def run_condition(
    condition: str,
    task_manifest: Path,
    *,
    probe: bool = False,
) -> None:
    folder = (PROBE if probe else PAIRED) / condition
    folder.mkdir(parents=True, exist_ok=True)
    expected = (1 if condition.endswith("singleton") else 4) if probe else 500
    if validate_condition(folder, condition, expected):
        return
    port = free_port()
    server_log, client_log = folder / "server.log", folder / "client.log"
    server_command = [
        str(SERVER_PYTHON), "-u", "scripts/run_groot_paired_server.py",
        "--condition", condition,
        "--checkpoint", str(CHECKPOINT),
        "--metadata", str(METADATA),
        "--task-manifest", str(task_manifest),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    client_command = [
        str(CLIENT_PYTHON), "-u", "scripts/run_groot_paired_client.py",
        "--condition", condition,
        "--host", "127.0.0.1", "--port", str(port),
        "--output", str(folder),
        "--task-manifest", str(task_manifest),
        "--seed", "20261005", "--num-steps-wait", "10",
    ]
    if probe:
        client_command += [
            "--task-limit", "1", "--episodes-per-task", str(expected),
        ]
    server_env = os.environ.copy()
    server_env.update({
        "PYTHONPATH": f"{ROOT}:{MODEL_ROOT}",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    client_env = os.environ.copy()
    client_env.update({
        "PYTHONPATH": f"{ROOT}:{MODEL_ROOT}",
        "LIBERO_CONFIG_PATH": "/workspace/assets/uva/libero_config",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    })
    print(f"Starting GR00T {condition}; logs={server_log},{client_log}", flush=True)
    server = None
    try:
        with server_log.open("a") as stream:
            server = subprocess.Popen(
                server_command,
                cwd=ROOT,
                env=server_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_server(server, port, server_log)
            with client_log.open("a") as stream:
                result = subprocess.run(
                    client_command,
                    cwd=ROOT,
                    env=client_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        if result.returncode != 0:
            raise RuntimeError(f"GR00T {condition} exited {result.returncode}; see {client_log}")
    finally:
        if server is not None:
            stop_group(server)
    if not validate_condition(folder, condition, expected):
        raise RuntimeError(f"GR00T {condition} lacks {expected} verified episodes")


def run_condition_resumable(condition: str, task_manifest: Path) -> None:
    failures = []
    for attempt in range(1, MAX_CONDITION_ATTEMPTS + 1):
        try:
            wait_for_gpu_capacity(condition)
            run_condition(condition, task_manifest)
            return
        except Exception as error:
            folder = PAIRED / condition
            if validate_condition(folder, condition, 500):
                return
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            (folder / "supervisor_retries.json").write_text(json.dumps({
                "schema_version": 1,
                "condition": condition,
                "maximum_attempts": MAX_CONDITION_ATTEMPTS,
                "attempts_failed": len(failures),
                "failures": failures,
                "append_resume_safe": True,
            }, indent=2) + "\n")
            if attempt == MAX_CONDITION_ATTEMPTS:
                raise
            delay = min(60, 5 * attempt)
            print(
                f"GR00T {condition} attempt {attempt} failed; preserving caches and "
                f"retrying in {delay}s: {error}",
                flush=True,
            )
            time.sleep(delay)


def finalize(task_manifest: Path) -> None:
    pairing = PAIRED / "reset_pairing_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_groot_reset_pairing.py",
        "--paired-root", str(PAIRED), "--require-complete", "--output", str(pairing),
    ], cwd=ROOT, check=True)
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError("GR00T physical reset/noise pairing failed")
    subprocess.run([
        sys.executable, "scripts/compress_preserved_rollouts.py",
        "--root", str(PAIRED), "--allow-fully-paired", "--expected-files", "1260",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(PAIRED), "--gzip-episodes",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/finalize_lerobot_paired_record.py",
        "--model-id", "groot_n17", "--paired-root", str(PAIRED),
        "--attempt-record", str(PAIRED.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(CHECKPOINT / "model.safetensors.index.json"),
        "--checkpoint-sha256", ARTIFACT_HASHES[CHECKPOINT / "model.safetensors.index.json"],
        "--source-evidence", str(MODEL_ROOT / "README.md"),
        "--official-evaluator-source", str(MODEL_ROOT / "gr00t/eval/rollout_policy.py"),
        "--evaluator-task-manifest", str(task_manifest),
    ], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-manifest-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()
    task_manifest = make_task_manifest()
    if args.make_manifest_only:
        print(task_manifest)
        return
    if not args.skip_probe:
        for condition in CONDITIONS:
            run_condition(condition, task_manifest, probe=True)
    if args.probe_only:
        print("GR00T paired probe complete", flush=True)
        return
    with supervisor_lock():
        for condition in CONDITIONS:
            if validate_condition(PAIRED / condition, condition, 500):
                print(f"Already verified GR00T {condition}; preserving evidence", flush=True)
                continue
            join_external_condition(condition)
            if not validate_condition(PAIRED / condition, condition, 500):
                run_condition_resumable(condition, task_manifest)
    finalize(task_manifest)
    print("GR00T paired campaign independently verified", flush=True)


if __name__ == "__main__":
    main()

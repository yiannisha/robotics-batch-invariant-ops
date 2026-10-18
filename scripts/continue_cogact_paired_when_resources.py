"""Run the resumable CogACT paired campaign once safe resources are available."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = (
    "native_singleton",
    "native_dynamic",
    "patched_singleton",
    "patched_dynamic",
)
EXPECTED_EPISODES = 496
ROOT = Path("/workspace/robotics-batch-invariant-ops")
OUTPUT = ROOT / "results/cross_model/cogact_small/full/closed_loop/paired"
SERVER_PYTHON = Path("/workspace/envs/openvla_oft/bin/python")
CLIENT_PYTHON = Path("/workspace/envs/simpler_eval/bin/python")
WSA_SUPERVISOR_LOCK = (
    ROOT / "results/cross_model/wsa_base/full/closed_loop/paired/.supervisor.lock"
)
INTERNVLA_OPTIMIZED_LOCK = (
    ROOT / "results/cross_model/internvla_a15_optimized/full/closed_loop/paired/.supervisor.lock"
)
UVA_JOINT_LOCK = (
    ROOT / "results/cross_model/uva_joint/full/closed_loop/paired/.supervisor.lock"
)
HIGHER_PRIORITY_LOCKS = (
    ("WSA", WSA_SUPERVISOR_LOCK),
    ("InternVLA optimized", INTERNVLA_OPTIMIZED_LOCK),
    ("UVA joint", UVA_JOINT_LOCK),
)


def gpu_free_mib() -> int:
    text = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(text.strip().splitlines()[0])


def cgroup_memory_bytes() -> int:
    return int(Path("/sys/fs/cgroup/memory.current").read_text().strip())


def lock_active(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def episode_count(condition: str) -> int:
    seen = set()
    directory = OUTPUT / condition / "rollouts"
    paths = sorted(directory.glob("*.json")) + sorted(directory.glob("*.json.gz"))
    for path in paths:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as stream:
                payload = json.load(stream)
        else:
            payload = json.loads(path.read_text())
        for row in payload.get("episodes", []):
            key = (row.get("task"), row.get("initialization_id"))
            if None in key or key in seen:
                raise RuntimeError(f"invalid or duplicate CogACT episode key {key} in {path}")
            seen.add(key)
    return len(seen)


def wait_for_resources(minimum_gpu_mib: int, maximum_cgroup_bytes: int) -> None:
    while True:
        active = next(
            (name for name, path in HIGHER_PRIORITY_LOCKS if lock_active(path)),
            None,
        )
        if active is not None:
            print(f"CogACT waiting for higher-priority {active} campaign", flush=True)
            time.sleep(30)
            continue
        free = gpu_free_mib(); memory = cgroup_memory_bytes()
        if free >= minimum_gpu_mib and memory <= maximum_cgroup_bytes:
            print(
                f"CogACT resource gate passed: gpu_free_mib={free} "
                f"cgroup_memory_bytes={memory}",
                flush=True,
            )
            return
        print(
            f"CogACT waiting: gpu_free_mib={free}/{minimum_gpu_mib} "
            f"cgroup_memory_bytes={memory}/{maximum_cgroup_bytes}",
            flush=True,
        )
        time.sleep(30)


def wait_for_server(process: subprocess.Popen, port: int, timeout: float = 1_200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"CogACT server exited during startup with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError("CogACT server did not listen within 1,200 seconds")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()


def write_retry(condition: str, attempt: int, returncode: int) -> None:
    path = OUTPUT / condition / "supervisor_retries.json"
    prior = json.loads(path.read_text()) if path.is_file() else {"retries": []}
    prior["retries"].append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "client_returncode": returncode,
        "episodes_preserved": episode_count(condition),
    })
    temporary = path.with_name(path.name + ".staging")
    temporary.write_text(json.dumps(prior, indent=2) + "\n")
    os.replace(temporary, path)


def run_condition(
    condition: str, port: int, minimum_gpu_mib: int, maximum_cgroup_bytes: int,
) -> None:
    destination = OUTPUT / condition
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "official/eval_info.json").is_file() and episode_count(condition) == EXPECTED_EPISODES:
        print(f"CogACT {condition}: already complete", flush=True)
        return

    server_env = os.environ.copy()
    server_env["PYTHONPATH"] = ":".join((
        "/dev/shm/cogact/python",
        str(ROOT), str(ROOT / "batch_invariant_ops"),
        "/workspace/models/internvla-a-series",
    ))
    client_env = os.environ.copy()
    client_env.update({
        "PYTHONPATH": ":".join((
            str(ROOT), "/workspace/models/simplerenv-openvla",
            "/workspace/models/internvla-a-series",
        )),
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    })

    for attempt in range(1, 21):
        wait_for_resources(minimum_gpu_mib, maximum_cgroup_bytes)
        client_returncode = -1
        with (destination / "server.stdout.log").open("a") as server_stdout, \
                (destination / "server.stderr.log").open("a") as server_stderr:
            server = subprocess.Popen(
                [
                    str(SERVER_PYTHON), "-u", "scripts/run_cogact_paired_server.py",
                    "--condition", condition, "--checkpoint", "/workspace/assets/cogact_small",
                    "--host", "127.0.0.1", "--port", str(port),
                ],
                cwd=ROOT, env=server_env, stdout=server_stdout, stderr=server_stderr,
            )
            try:
                wait_for_server(server, port)
                with (destination / "client.stdout.log").open("a") as client_stdout, \
                        (destination / "client.stderr.log").open("a") as client_stderr:
                    client = subprocess.run(
                        [
                            str(CLIENT_PYTHON), "-u", "scripts/run_cogact_paired_client.py",
                            "--condition", condition, "--host", "127.0.0.1",
                            "--port", str(port), "--output", str(destination),
                        ],
                        cwd=ROOT, env=client_env,
                        stdout=client_stdout, stderr=client_stderr,
                    )
                    client_returncode = client.returncode
            except Exception as error:
                with (destination / "supervisor_errors.log").open("a") as stream:
                    stream.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"attempt={attempt} {type(error).__name__}: {error}\n"
                    )
            finally:
                stop_server(server)
        if client_returncode == 0 and episode_count(condition) == EXPECTED_EPISODES:
            print(f"CogACT {condition}: complete", flush=True)
            return
        write_retry(condition, attempt, client_returncode)
        print(
            f"CogACT {condition}: retry {attempt}/20 after returncode={client_returncode}",
            flush=True,
        )
    raise RuntimeError(f"CogACT {condition} exhausted 20 retries")


def finalize_pairing() -> None:
    pairing = OUTPUT / "reset_pairing_audit.json"
    task_manifest = OUTPUT / "evaluator_task_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_cogact_reset_pairing.py",
            "--root",
            str(OUTPUT),
            "--task-manifest",
            str(task_manifest),
            "--output",
            str(pairing),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compress_preserved_rollouts.py",
            "--root",
            str(OUTPUT),
            "--allow-fully-paired",
            "--expected-files",
            "1248",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_cogact_paired.py",
            "--root",
            str(OUTPUT),
            "--task-manifest",
            str(task_manifest),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/finalize_cogact_paired_record.py",
            "--paired-root",
            str(OUTPUT),
            "--attempt-record",
            str(OUTPUT.parent / "evaluator_attempt.json"),
            "--checkpoint-weight",
            "/workspace/assets/cogact_small/checkpoints/CogACT-Small.pt",
            "--source-evidence",
            "/workspace/models/cogact/README.md",
            "--official-evaluator-source",
            "/workspace/models/cogact/sim_cogact/cogact_policy.py",
            "--evaluator-task-manifest",
            str(task_manifest),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=49817)
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=28_000)
    parser.add_argument("--maximum-cgroup-bytes", type=int, default=105_000_000_000)
    args = parser.parse_args()
    for condition in CONDITIONS:
        run_condition(
            condition, args.port,
            args.minimum_gpu_free_mib, args.maximum_cgroup_bytes,
        )
    finalize_pairing()
    print("CogACT four-condition campaign independently paired and reduced", flush=True)


if __name__ == "__main__":
    main()

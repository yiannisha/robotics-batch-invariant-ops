"""Probe both official InternVLA-A1.5 LIBERO inference backends after SmolVLA.

These are deliberately one-episode smoke probes.  They establish that the
checkpoint-matched official evaluator can execute, but they never claim the
four-condition paired success-rate campaign is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/workspace/models/internvla-a-series")
SERVER_PYTHON = Path("/workspace/envs/internvla_a15/bin/python")
CLIENT_PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
SERVER_SOURCE = MODEL_ROOT / "evaluation/LIBERO/policy_server/server_policy.py"
CLIENT_SOURCE = MODEL_ROOT / "evaluation/LIBERO/eval_libero_server_client.py"
README_SOURCE = MODEL_ROOT / "evaluation/LIBERO/README.md"
BACKEND_SOURCE = MODEL_ROOT / "evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py"
STANDARD_SOURCE = MODEL_ROOT / "src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py"
OPTIMIZED_SOURCE = MODEL_ROOT / "src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5_optimized.py"
CHECKPOINT = Path("/workspace/assets/internvla_a15/checkpoint")
CHECKPOINT_WEIGHT = CHECKPOINT / "model.safetensors"
VLM = Path("/workspace/assets/internvla_a15/qwen")
VLM_WEIGHT = VLM / "model.safetensors-00001-of-00001.safetensors"

EXPECTED_HASHES = {
    CHECKPOINT_WEIGHT: "222b6ed5dbd3224f441ad2f9165838a890aad5934282209fb635ca4daf073de0",
    CHECKPOINT / "config.json": "e9c5f1465fd8b933f5e0804fa3083410d147b94ac83c374fcadc1b5000c01ec5",
    CHECKPOINT / "stats.json": "80bd0eeb32f27d40fcdec1da6ca718c1ffa9d70792f1578d7ddcb028b8508c51",
    VLM_WEIGHT: "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
    README_SOURCE: "efa34479aca3e695981144b4977cda02b58c8f6c9b575ba29d838800c766fff5",
    SERVER_SOURCE: "b296a3332ec8dc32d228d8b2f85a48d2e40f3c07f689ce7fe384f3a06a198dc1",
    CLIENT_SOURCE: "8090d19b19202a6a2241c8b521c8f6ce98a5557907cd7bbd17db68a75c883874",
    BACKEND_SOURCE: "c78e11cf9b0c9ed6d74e260718168e347fd96e388c405f6a65270f246b0dcffe",
    STANDARD_SOURCE: "3c499b6979691b148bce55db85b30760bd40b3b8ed418203a66609ff34c0dd2c",
    OPTIMIZED_SOURCE: "e7bb04dc4a1cb83e5c032f8a02623da044b1c2621e9603f9e74d7a5bedee3572",
}
BACKENDS = {
    "internvla_a15_standard": "standard",
    "internvla_a15_optimized": "optimized",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_pinned_inputs() -> list[dict]:
    pinned = []
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned InternVLA input: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned InternVLA input changed: {path}: {actual} != {expected}")
        pinned.append({"path": str(path), "sha256": actual})
    for executable in (SERVER_PYTHON, CLIENT_PYTHON):
        if not executable.is_file():
            raise FileNotFoundError(f"missing evaluator Python: {executable}")
    return pinned


def wait_for_smol_worker(pid: int) -> dict:
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
        if "continue_smolvla_after_pi0fast.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies the SmolVLA worker")
        time.sleep(20)
    path = ROOT / "results/cross_model/smolvla/full/closed_loop/evaluator_attempt.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


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


def wait_for_server(process: subprocess.Popen, port: int, log_path: Path, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"InternVLA server exited {return_code}; see {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"InternVLA server did not listen within {timeout}s; see {log_path}")


def valid_completed_probe(identifier: str, backend: str) -> bool:
    attempt = ROOT / f"results/cross_model/{identifier}/full/closed_loop/evaluator_attempt.json"
    if not attempt.is_file():
        return False
    record = json.loads(attempt.read_text())
    if (record.get("model_id") != identifier
            or record.get("inference_backend") != backend
            or record.get("status") != "probe_succeeded_requires_paired_campaign"
            or record.get("exit_code") != 0
            or record.get("probe_tasks_completed") != 1
            or record.get("probe_episodes_completed") != 1):
        return False
    for path_key, hash_key in (
        ("probe_eval_info_path", "probe_eval_info_sha256"),
        ("official_probe_result_path", "official_probe_result_sha256"),
    ):
        path = Path(record.get(path_key, ""))
        if not path.is_file() or sha256(path) != record.get(hash_key):
            return False
    return record.get("checkpoint_weight_sha256") == EXPECTED_HASHES[CHECKPOINT_WEIGHT]


def normalize_official_metrics(official: dict) -> dict:
    per_task = official.get("per_task")
    if (official.get("task_suite") != "libero_10" or official.get("n_tasks") != 1
            or official.get("total_episodes") != 1 or not isinstance(per_task, dict)
            or len(per_task) != 1):
        raise RuntimeError(f"unexpected official InternVLA probe metrics: {official}")
    name, row = next(iter(per_task.items()))
    if row.get("n_episodes") != 1 or row.get("successes") not in (0, 1):
        raise RuntimeError(f"invalid official InternVLA per-task probe metrics: {row}")
    if official.get("total_successes") != row["successes"]:
        raise RuntimeError("official InternVLA aggregate/per-task success mismatch")
    return {
        "task_suite": "libero_10",
        "per_task": [{
            "task_id": 0,
            "task": name,
            "n_episodes": 1,
            "n_success": row["successes"],
            "success_rate": row["success_rate"],
        }],
        "overall": {
            "n_episodes": 1,
            "n_success": official["total_successes"],
            "success_rate": official["overall_success_rate"],
        },
    }


def run_probe(identifier: str, backend: str, pinned_sources: list[dict]) -> None:
    if valid_completed_probe(identifier, backend):
        print(f"Already verified InternVLA {backend} official probe; preserving evidence", flush=True)
        return

    closed_loop = ROOT / f"results/cross_model/{identifier}/full/closed_loop"
    output = closed_loop / "probe_output"
    output.mkdir(parents=True, exist_ok=True)
    attempt_path = closed_loop / "evaluator_attempt.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    server_log = output / f"server_{stamp}.log"
    client_log = output / f"client_{stamp}.log"
    port = free_local_port()

    server_command = [
        str(SERVER_PYTHON), "-u", str(SERVER_SOURCE),
        "--ckpt_path", str(CHECKPOINT), "--host", "127.0.0.1", "--port", str(port),
        "--device", "cuda", "--resize_size", "224", "--stats_key", "libero_10",
        "--robot_type", "panda", "--vlm_model_path", str(VLM), "--action_loss_only",
        "--inference_backend", backend, "--idle_timeout", "-1",
    ]
    client_command = [
        str(CLIENT_PYTHON), "-u", str(CLIENT_SOURCE),
        "--host", "127.0.0.1", "--port", str(port), "--task_suite_name", "libero_10",
        "--num_trials_per_task", "1", "--max_tasks", "1", "--num_steps_wait", "10",
        "--seed", "7", "--replan_steps", "8", "--eval_log_dir", str(output),
        "--no-save_videos",
    ]
    server_environment = os.environ.copy()
    server_environment.update({
        "PYTHONPATH": f"{MODEL_ROOT}:{MODEL_ROOT / 'src'}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    client_environment = os.environ.copy()
    client_environment.update({
        "PYTHONPATH": f"{MODEL_ROOT}:{MODEL_ROOT / 'src'}",
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "TOKENIZERS_PARALLELISM": "false",
    })
    base_record = {
        "model_id": identifier,
        "evaluator": "official InternVLA-A1.5 websocket LIBERO evaluator",
        "inference_backend": backend,
        "attempt_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "server_command": server_command,
        "attempted_command": client_command,
        "server_log_path": str(server_log),
        "stdout_path": str(server_log),
        "stderr_path": str(client_log),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_weight_path": str(CHECKPOINT_WEIGHT),
        "checkpoint_weight_sha256": EXPECTED_HASHES[CHECKPOINT_WEIGHT],
        "vlm_weight_path": str(VLM_WEIGHT),
        "vlm_weight_sha256": EXPECTED_HASHES[VLM_WEIGHT],
        "source_evidence_path": str(README_SOURCE),
        "source_evidence_sha256": EXPECTED_HASHES[README_SOURCE],
        "official_evaluator_source_path": str(CLIENT_SOURCE),
        "official_evaluator_source_sha256": EXPECTED_HASHES[CLIENT_SOURCE],
        "pinned_sources": pinned_sources,
        "probe_task_suite": "libero_10",
        "probe_seed": 7,
        "probe_num_steps_wait": 10,
        "probe_replan_steps": 8,
        "paired_episodes_completed": 0,
    }

    server = None
    exit_code = 1
    failure = None
    try:
        with server_log.open("w") as server_stream:
            server = subprocess.Popen(
                server_command, cwd=MODEL_ROOT, env=server_environment,
                stdout=server_stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
            wait_for_server(server, port, server_log)
            with client_log.open("w") as client_stream:
                result = subprocess.run(
                    client_command, cwd=MODEL_ROOT, env=client_environment,
                    stdout=client_stream, stderr=subprocess.STDOUT, check=False,
                    timeout=3600,
                )
            exit_code = result.returncode
            if exit_code != 0:
                raise RuntimeError(f"InternVLA {backend} client exited {exit_code}; see {client_log}")

        official_path = output / "results_libero_10.json"
        if not official_path.is_file():
            raise RuntimeError(f"official InternVLA result missing: {official_path}")
        official = json.loads(official_path.read_text())
        normalized = normalize_official_metrics(official)
        normalized_path = output / "probe_eval_info.json"
        write_json_atomic(normalized_path, normalized)
        record = {
            **base_record,
            "status": "probe_succeeded_requires_paired_campaign",
            "exit_code": 0,
            "probe_tasks_completed": 1,
            "probe_episodes_completed": 1,
            "probe_successes": normalized["overall"]["n_success"],
            "official_probe_result_path": str(official_path),
            "official_probe_result_sha256": sha256(official_path),
            "probe_eval_info_path": str(normalized_path),
            "probe_eval_info_sha256": sha256(normalized_path),
        }
        write_json_atomic(attempt_path, record)
        print(
            f"Verified InternVLA {backend} official probe: "
            f"{record['probe_successes']}/1; paired campaign remains pending",
            flush=True,
        )
    except subprocess.TimeoutExpired as error:
        exit_code = 124
        failure = f"client timeout: {error}"
    except Exception as error:  # preserve exact attempt evidence before continuing
        failure = f"{type(error).__name__}: {error}"
    finally:
        if server is not None:
            stop_process_group(server)

    if failure is not None:
        if not server_log.exists():
            server_log.write_text("")
        if not client_log.exists():
            client_log.write_text(failure + "\n")
        record = {
            **base_record,
            "status": "blocked",
            "exit_code": exit_code,
            "failure_class": "official_evaluator_probe_failed",
            "failure": failure,
            "probe_tasks_completed": 0,
            "probe_episodes_completed": 0,
            "probe_successes": 0,
            "blocker_investigation": {
                "reason_permanent": False,
                "resolution_attempts": ["Pinned exact assets and both environments; executed official server/client probe."],
                "evidence_files": [
                    {"path": str(server_log), "sha256": sha256(server_log)},
                    {"path": str(client_log), "sha256": sha256(client_log)},
                ],
            },
        }
        write_json_atomic(attempt_path, record)
        print(f"InternVLA {backend} probe failed independently: {failure}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smol-worker-pid", type=int, required=True)
    args = parser.parse_args()
    if args.smol_worker_pid <= 0:
        raise ValueError("invalid SmolVLA worker PID")
    pinned_sources = verify_pinned_inputs()
    predecessor = wait_for_smol_worker(args.smol_worker_pid)
    print(
        "SmolVLA predecessor ended; "
        f"status={predecessor.get('status', 'missing')} (InternVLA probes proceed independently)",
        flush=True,
    )
    for identifier, backend in BACKENDS.items():
        run_probe(identifier, backend, pinned_sources)


if __name__ == "__main__":
    main()

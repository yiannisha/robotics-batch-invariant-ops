"""Run the full paired RDT-1B ManiSkill campaign after WSA."""

from __future__ import annotations

import argparse
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
MODEL_ROOT = Path("/workspace/models/rdt")
TRANSPORT_ROOT = Path("/workspace/models/internvla-a-series")
CHECKPOINT = Path("/workspace/assets/rdt/checkpoint")
SIGLIP = Path("/workspace/assets/rdt/siglip")
PYTHON = Path("/workspace/envs/rdt/bin/python")
PAIRED = ROOT / "results/cross_model/rdt_1b/full/closed_loop/paired"
CONDITIONS = ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
MAX_CONDITION_ATTEMPTS = 20
TASKS = (
    "PegInsertionSide-v1", "PickCube-v1", "StackCube-v1", "PlugCharger-v1", "PushCube-v1",
)
CHECKPOINT_SHA256 = "3fa89cbf7bbb1b2559b29f9ae153b431541cdaa0b3f1ee44b83b9934599b9fb8"
MODEL_REPOSITORY_REVISION = "9622afab2b7ce2312a6cf1febc526928589b77eb"
ARTIFACT_HASHES = {
    CHECKPOINT / "rdt/mp_rank_00_model_states.pt": CHECKPOINT_SHA256,
    CHECKPOINT / "lang_embeds/text_embed_PegInsertionSide-v1.pt": "da798aab9df67a230259095c0d087772d10ff2dd089bd71762a1b6d1696dbbf3",
    CHECKPOINT / "lang_embeds/text_embed_PickCube-v1.pt": "a9ac30bc88ef12b171e9323b8a2c64f0b0ae9a54715deb061c021deaadb37f7a",
    CHECKPOINT / "lang_embeds/text_embed_PlugCharger-v1.pt": "c38f80c549ec1d48f74ffdeafff8dd9f02ca64de8c3bddd7fe3cca173925922d",
    CHECKPOINT / "lang_embeds/text_embed_PushCube-v1.pt": "536272882840dd1d2aef759b7551da1515b9c0cbe45e491a025169efadf51471",
    CHECKPOINT / "lang_embeds/text_embed_StackCube-v1.pt": "ca0a1f758d183e4c541b412efd2ffafacae7bdc7220d03f7e9dca3e93bb28985",
    SIGLIP / "model.safetensors": "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a",
    SIGLIP / "config.json": "adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b",
    SIGLIP / "preprocessor_config.json": "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_wsa(pid: int) -> None:
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
        if "continue_wsa_paired_after_openvla.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies WSA paired worker")
        time.sleep(20)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_server(process: subprocess.Popen, port: int, log: Path, timeout: int = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"RDT server exited {process.returncode}; see {log}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"RDT server startup timed out; see {log}")


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
    if not manifest_path.is_file() or not info_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    required_batch = 1 if condition.endswith("singleton") else 4
    if (manifest.get("model") != "rdt_1b" or manifest.get("condition") != condition
            or manifest.get("batch_size") != required_batch
            or manifest.get("invariant_ops") != condition.startswith("patched")
            or manifest.get("task_suite") != "maniskill_five"
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
    expected = {(task, initialization) for task in range(5) for initialization in range(50)}
    if set(rows) != expected:
        return False
    info = json.loads(info_path.read_text())
    per_task = {row.get("task_id"): row for row in info.get("per_task", [])}
    return (set(per_task) == set(range(5)) and info.get("overall", {}).get("n_episodes") == 250
            and all(per_task[task]["metrics"]["successes"] == [rows[task, i]["success"] for i in range(50)]
                    for task in range(5)))


def make_task_manifest() -> Path:
    records = []
    for path, expected in ARTIFACT_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"RDT exact artifact changed: {path}")
        records.append({"path": str(path.resolve()), "sha256": expected})
    value = {
        "schema_version": 1, "task_suite": "maniskill_five",
        "task_ids": list(range(5)), "task_names": list(TASKS),
        "initializations_per_task": 50, "base_seed": 20241201,
        "checkpoint_repository": "robotics-diffusion-transformer/maniskill-model",
        "checkpoint_repository_revision": MODEL_REPOSITORY_REVISION,
        "artifact_files": records, "checkpoint_files": records,
    }
    path = PAIRED / "evaluator_task_manifest.json"
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.is_file() and path.read_text() != text:
        raise RuntimeError("RDT task/checkpoint manifest changed")
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
        str(PYTHON), "-u", "scripts/run_rdt_paired_server.py",
        "--condition", condition, "--checkpoint", str(CHECKPOINT), "--siglip", str(SIGLIP),
        "--task-manifest", str(task_manifest), "--host", "127.0.0.1", "--port", str(port),
    ]
    client_command = [
        str(PYTHON), "-u", "scripts/run_rdt_paired_client.py",
        "--condition", condition, "--host", "127.0.0.1", "--port", str(port),
        "--output", str(folder), "--task-manifest", str(task_manifest),
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": f"{ROOT}:{MODEL_ROOT}:{TRANSPORT_ROOT}:{TRANSPORT_ROOT / 'src'}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    print(f"Starting RDT {condition}; logs={server_log},{client_log}", flush=True)
    server = None
    try:
        with server_log.open("a") as stream:
            server = subprocess.Popen(
                server_command, cwd=ROOT, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            wait_server(server, port, server_log)
            with client_log.open("a") as stream:
                result = subprocess.run(
                    client_command, cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, check=False,
                )
        if result.returncode != 0:
            raise RuntimeError(f"RDT {condition} exited {result.returncode}; see {client_log}")
    finally:
        if server is not None:
            stop_group(server)
    if not validate_condition(condition):
        raise RuntimeError(f"RDT {condition} lacks 250 verified episodes")


def run_condition_resumable(condition: str, task_manifest: Path) -> None:
    """Retry transport/runtime exits while preserving validated rollout caches."""
    failures = []
    for attempt in range(1, MAX_CONDITION_ATTEMPTS + 1):
        try:
            run_condition(condition, task_manifest)
            return
        except Exception as error:
            if validate_condition(condition):
                return
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            retry_path = PAIRED / condition / "supervisor_retries.json"
            retry_path.write_text(json.dumps({
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
                f"RDT {condition} attempt {attempt} failed; "
                f"preserving audited caches and retrying in {delay}s: {error}",
                flush=True,
            )
            time.sleep(delay)


def finalize(task_manifest: Path) -> None:
    pairing = PAIRED / "reset_pairing_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_lerobot_reset_pairing.py", "--root", str(PAIRED),
        "--task-count", "5", "--output", str(pairing),
    ], cwd=ROOT, check=True)
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError("RDT physical reset/noise pairing failed")
    subprocess.run([
        sys.executable, "scripts/compress_preserved_rollouts.py", "--root", str(PAIRED),
        "--allow-fully-paired", "--expected-files", "630",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(PAIRED), "--task-count", "5", "--gzip-episodes",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/finalize_lerobot_paired_record.py",
        "--model-id", "rdt_1b", "--paired-root", str(PAIRED),
        "--attempt-record", str(PAIRED.parent / "evaluator_attempt.json"),
        "--checkpoint-weight", str(CHECKPOINT / "rdt/mp_rank_00_model_states.pt"),
        "--checkpoint-sha256", CHECKPOINT_SHA256,
        "--source-evidence", str(MODEL_ROOT / "README.md"),
        "--official-evaluator-source", str(MODEL_ROOT / "eval_sim/eval_rdt_maniskill.py"),
        "--evaluator-task-manifest", str(task_manifest),
    ], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsa-paired-pid", type=int, required=True)
    args = parser.parse_args()
    wait_for_wsa(args.wsa_paired_pid)
    task_manifest = make_task_manifest()
    for condition in CONDITIONS:
        if validate_condition(condition):
            print(f"Already verified RDT {condition}; preserving evidence", flush=True)
            continue
        run_condition_resumable(condition, task_manifest)
    finalize(task_manifest)
    print("RDT paired five-task ManiSkill campaign independently verified", flush=True)


if __name__ == "__main__":
    main()

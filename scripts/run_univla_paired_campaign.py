"""Run, resume, audit, and summarize all four UniVLA paired LIBERO conditions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lerobot_paired_libero import CONDITIONS, sha256, write_json_atomic
from scripts.run_univla_paired_libero import (
    LEROBOT_REVISION, POLICY_REVISION, SEED, UNIVLA_REVISION, VISION_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/workspace/envs/dexvla/bin/python")
RUNNER = ROOT / "scripts/run_univla_paired_libero.py"
FULL = ROOT / "results/cross_model/univla_baai/full"
PAIRED = FULL / "closed_loop/paired"
CLOSED = PAIRED.parent
RECEIPT = FULL / "checkpoint_cache_receipt.json"
TASK_MANIFEST = PAIRED / "evaluator_task_manifest.json"
SOURCE_TASK_MANIFEST = ROOT / "results/cross_model/molmoact2/full/closed_loop/paired/evaluator_task_manifest.json"
PREFIX_AUDITOR = ROOT / "scripts/audit_exact_lerobot_paired_prefix.py"


def wait_for_predecessor(pid: int, command_fragment: str) -> None:
    while True:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            return
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if command_fragment not in command:
            raise RuntimeError(f"PID {pid} no longer identifies {command_fragment!r}")
        print(f"waiting for predecessor PID {pid}: {command_fragment}", flush=True)
        time.sleep(30)


def wait_for_checkpoint() -> dict:
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if RECEIPT.is_file():
            receipt = json.loads(RECEIPT.read_text())
            pid = receipt.get("daemon_pid")
            command = Path(f"/proc/{pid}/cmdline") if isinstance(pid, int) else None
            if (
                receipt.get("all_pinned_files_verified") is True
                and receipt.get("storage") == "sealed_memfd"
                and len(receipt.get("files") or {}) == 5
                and command is not None and command.is_file()
                and b"serve_univla_checkpoint_memfd.py" in command.read_bytes()
            ):
                return receipt
        time.sleep(5)
    raise TimeoutError("UniVLA exact sealed checkpoint did not become ready")


def make_task_manifest(receipt: dict) -> None:
    source = json.loads(SOURCE_TASK_MANIFEST.read_text())
    if (
        source.get("task_suite") != "libero_10"
        or source.get("task_ids") != list(range(10))
        or source.get("initializations_per_task") != 50
        or len(source.get("tasks", [])) != 10
    ):
        raise ValueError("pinned LIBERO-10 evaluator task evidence differs")
    for task in source["tasks"]:
        for field, digest_field in (
            ("bddl_path", "bddl_sha256"),
            ("init_states_path", "init_states_sha256"),
        ):
            path = Path(task[field])
            if not path.is_file() or sha256(path) != task[digest_field]:
                raise ValueError(f"UniVLA LIBERO task asset differs: {path}")
    value = {
        "schema_version": 1, "model_id": "univla_baai",
        "task_suite": "libero_10", "task_ids": list(range(10)),
        "task_descriptions": source["task_descriptions"],
        "initializations_per_task": 50, "tasks": source["tasks"],
        "official_univla_revision": UNIVLA_REVISION,
        "official_lerobot_revision": LEROBOT_REVISION,
        "policy_checkpoint_revision": POLICY_REVISION,
        "vision_checkpoint_revision": VISION_REVISION,
        "checkpoint_receipt_path": str(RECEIPT.resolve()),
        "checkpoint_receipt_sha256": sha256(RECEIPT),
        "checkpoint_weight_path": str(
            (FULL / "runtime_checkpoint/policy/UNIVLA_LIBERO_IMG_BS192_8K/model.safetensors.index.json").resolve()
        ),
        "checkpoint_weight_sha256": sha256(
            FULL / "runtime_checkpoint/policy/UNIVLA_LIBERO_IMG_BS192_8K/model.safetensors.index.json"
        ),
        "source_evidence_path": str(
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/model_wrapper_emu.py")
        ),
        "source_evidence_sha256": sha256(
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/model_wrapper_emu.py")
        ),
        "official_evaluator_source_path": str(
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/evaluate_libero_emu.py")
        ),
        "official_evaluator_source_sha256": sha256(
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/evaluate_libero_emu.py")
        ),
        "checkpoint_model_source_path": str(
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/model_wrapper_emu.py")
        ),
        "checkpoint_model_source_sha256": sha256(
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/model_wrapper_emu.py")
        ),
        "checkpoint_files": [
            {
                "key": key, "path": row["path"], "bytes": row["bytes"],
                "sha256": row["sha256"], "repository": row["repository"],
                "revision": row["revision"],
            }
            for key, row in sorted(receipt["files"].items())
        ],
    }
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if TASK_MANIFEST.is_file() and TASK_MANIFEST.read_text() != encoded:
        rollout_evidence = list(PAIRED.glob("*/rollouts/sequence_*"))
        if rollout_evidence:
            raise RuntimeError(
                "UniVLA evaluator task/checkpoint manifest changed after rollouts started"
            )
    TASK_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(TASK_MANIFEST, value)


def condition_complete(condition: str) -> bool:
    directory = PAIRED / condition
    info_path = directory / "official/eval_info.json"
    manifest_path = directory / "condition_manifest.json"
    if not info_path.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    info = json.loads(info_path.read_text())
    metrics = {row.get("task_id"): row for row in info.get("per_task", [])}
    rollout_files = list((directory / "rollouts").glob("*.json")) + list(
        (directory / "rollouts").glob("*.json.gz")
    )
    return (
        manifest.get("model") == "univla_baai"
        and manifest.get("condition") == condition
        and manifest.get("episodes_per_task") == 50
        and manifest.get("seed") == SEED
        and set(metrics) == set(range(10))
        and all(row.get("n_episodes") == 50 for row in metrics.values())
        and info.get("overall", {}).get("n_episodes") == 500
        and len(rollout_files) == 130
    )


def audit_condition(condition: str) -> None:
    directory = PAIRED / condition
    output = directory / "prefix_audit.json"
    run([
        str(PYTHON), str(PREFIX_AUDITOR), "--condition-root", str(directory),
        "--output", str(output),
    ])
    report = json.loads(output.read_text())
    if (
        report.get("integrity_passed") is not True
        or report.get("provenance_integrity_passed") is not True
        or report.get("planned_episodes_cached") != 500
        or report.get("all_ten_tasks_have_50_resets") is not True
    ):
        raise RuntimeError(f"UniVLA {condition} failed its independent prefix audit")
def run(command: list[str], *, log: Path | None = None) -> None:
    environment = os.environ.copy()
    environment.update({
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    print("RUN", " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        subprocess.run(
            command, cwd=ROOT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--wait-for-command", default="run_openhelix_paired_campaign.py")
    args = parser.parse_args()
    if args.wait_for_pid is not None:
        if args.wait_for_pid <= 0 or not args.wait_for_command:
            raise ValueError("invalid UniVLA paired predecessor")
        wait_for_predecessor(args.wait_for_pid, args.wait_for_command)
    run([
        str(PYTHON), str(PREFIX_AUDITOR),
        "--readiness-model", "univla_baai",
    ])
    receipt = wait_for_checkpoint()
    make_task_manifest(receipt)
    write_json_atomic(CLOSED / "univla_paired_supervisor.json", {
        "schema_version": 1, "pid": os.getpid(),
        "waiting_for_pid": args.wait_for_pid,
        "waiting_for_command": args.wait_for_command,
        "conditions": list(CONDITIONS), "episodes_per_condition": 500,
        "total_paired_episode_records": 2000,
    })
    for condition in CONDITIONS:
        if condition_complete(condition):
            audit_condition(condition)
            print(f"preserving verified completed UniVLA {condition}", flush=True)
            continue
        directory = PAIRED / condition
        run([
            str(PYTHON), "-u", str(RUNNER), "--condition", condition,
            "--output", str(directory), "--episodes-per-task", "50",
            "--seed", str(SEED),
        ], log=directory / "campaign.log")
        if not condition_complete(condition):
            raise RuntimeError(f"UniVLA {condition} did not produce 500 official episodes")
        audit_condition(condition)

    pairing = PAIRED / "reset_pairing_audit.json"
    run([
        str(PYTHON), "scripts/audit_lerobot_reset_pairing.py",
        "--root", str(PAIRED), "--output", str(pairing),
    ])
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError("UniVLA physical reset/deterministic pairing failed")
    run([
        str(PYTHON), "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(PAIRED), "--gzip-episodes",
    ])
    summary_path = PAIRED / "paired_summary.json"
    episodes_path = PAIRED / "episodes.jsonl.gz"
    summary = json.loads(summary_path.read_text())
    attempt = {
        "schema_version": 2, "model_id": "univla_baai",
        "evaluator": "official UniVLA image-policy LIBERO-10 evaluator",
        "status": "completed", "exit_code": 0,
        "paired_episodes_completed": 2000, "paired_start_seed": SEED,
        "episodes_path": str(episodes_path.resolve()),
        "episodes_sha256": sha256(episodes_path),
        "paired_summary_path": str(summary_path.resolve()),
        "paired_summary_sha256": sha256(summary_path),
        "condition_eval_info": summary["condition_eval_info"],
        "checkpoint_receipt_path": str(RECEIPT.resolve()),
        "checkpoint_receipt_sha256": sha256(RECEIPT),
        "official_model_repository_path": str(Path("/workspace/models/univla")),
        "official_model_repository_revision": UNIVLA_REVISION,
        "official_lerobot_revision": LEROBOT_REVISION,
        "policy_checkpoint_revision": POLICY_REVISION,
        "vision_checkpoint_revision": VISION_REVISION,
        "paired_wrapper_path": str(RUNNER.resolve()),
        "paired_wrapper_sha256": sha256(RUNNER),
        "reset_pairing_audit_path": str(pairing.resolve()),
        "reset_pairing_audit_sha256": sha256(pairing),
        "evaluator_task_manifest_path": str(TASK_MANIFEST.resolve()),
        "evaluator_task_manifest_sha256": sha256(TASK_MANIFEST),
    }
    write_json_atomic(CLOSED / "evaluator_attempt.json", attempt)
    print(json.dumps({"status": "completed", "paired_episodes": 2000}, sort_keys=True))


if __name__ == "__main__":
    main()

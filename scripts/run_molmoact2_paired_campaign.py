"""Run, resume, audit, and summarize all four MolmoAct2 paired LIBERO conditions."""

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


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/workspace/envs/smolvla_eval/bin/python")
RUNNER = ROOT / "scripts/run_molmoact2_paired_libero.py"
PAIRED = ROOT / "results/cross_model/molmoact2/full/closed_loop/paired"
CLOSED = PAIRED.parent
CHECKPOINT = Path("/workspace/assets/molmoact2/checkpoint")
TASK_MANIFEST = PAIRED / "evaluator_task_manifest.json"
EVALUATOR = Path("/workspace/models/lerobot/src/lerobot/scripts/lerobot_eval.py")
SOURCE_EVIDENCE = Path(
    "/workspace/models/molmoact2-official/experiments/lerobot/docs/source/molmoact2.mdx"
)
SEED = 20261015
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
            raise RuntimeError(f"PID {pid} no longer contains {command_fragment!r}")
        print(f"waiting for predecessor PID {pid}: {command_fragment}", flush=True)
        time.sleep(30)


def run(command: list[str], *, log: Path | None = None) -> None:
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": f"/workspace/models/lerobot/src:{ROOT}",
        "LIBERO_CONFIG_PATH": "/workspace/assets/lerobot_libero_eval_config",
        "MUJOCO_GL": "egl",
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
        manifest.get("model") == "molmoact2"
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
        raise RuntimeError(f"MolmoAct2 {condition} failed its independent prefix audit")
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--wait-for-command", default="run_openhelix_paired_campaign.py")
    args = parser.parse_args()
    if args.wait_for_pid is not None:
        if args.wait_for_pid <= 0 or not args.wait_for_command:
            raise ValueError("invalid paired predecessor")
        wait_for_predecessor(args.wait_for_pid, args.wait_for_command)
    run([
        str(PYTHON), str(PREFIX_AUDITOR),
        "--readiness-model", "molmoact2",
    ])
    write_json_atomic(CLOSED / "molmoact2_paired_supervisor.json", {
        "schema_version": 1, "pid": os.getpid(),
        "waiting_for_pid": args.wait_for_pid,
        "conditions": list(CONDITIONS), "episodes_per_condition": 500,
        "total_paired_episode_records": 2000,
    })
    for condition in CONDITIONS:
        if condition_complete(condition):
            audit_condition(condition)
            print(f"preserving verified completed MolmoAct2 {condition}", flush=True)
            continue
        directory = PAIRED / condition
        run([
            str(PYTHON), "-u", str(RUNNER), "--condition", condition,
            "--output", str(directory), "--episodes-per-task", "50", "--seed", str(SEED),
        ], log=directory / "campaign.log")
        if not condition_complete(condition):
            raise RuntimeError(f"MolmoAct2 {condition} did not produce 500 official episodes")
        audit_condition(condition)

    pairing = PAIRED / "reset_pairing_audit.json"
    run([
        str(PYTHON), "scripts/audit_lerobot_reset_pairing.py",
        "--root", str(PAIRED), "--output", str(pairing),
    ])
    if json.loads(pairing.read_text()).get("fully_paired") is not True:
        raise RuntimeError("MolmoAct2 physical reset/noise pairing failed")
    run([
        str(PYTHON), "scripts/summarize_lerobot_paired_libero.py",
        "--root", str(PAIRED), "--gzip-episodes",
    ])
    summary_path = PAIRED / "paired_summary.json"
    episodes_path = PAIRED / "episodes.jsonl.gz"
    summary = json.loads(summary_path.read_text())
    index = CHECKPOINT / "model.safetensors.index.json"
    receipt = ROOT / "results/cross_model/molmoact2/full/checkpoint_stream_receipt.json"
    if not receipt.is_file():
        raise FileNotFoundError("MolmoAct2 strict checkpoint stream receipt is missing")
    attempt = {
        "schema_version": 2, "model_id": "molmoact2",
        "evaluator": "official LeRobot LIBERO-10 evaluator with public MolmoAct2 policy",
        "status": "completed", "exit_code": 0,
        "paired_episodes_completed": 2000, "paired_start_seed": SEED,
        "episodes_path": str(episodes_path.resolve()),
        "episodes_sha256": sha256(episodes_path),
        "paired_summary_path": str(summary_path.resolve()),
        "paired_summary_sha256": sha256(summary_path),
        "condition_eval_info": summary["condition_eval_info"],
        "checkpoint_weight_path": str(index.resolve()),
        "checkpoint_weight_sha256": sha256(index),
        "checkpoint_stream_receipt_path": str(receipt.resolve()),
        "checkpoint_stream_receipt_sha256": sha256(receipt),
        "checkpoint_model_source_path": str((CHECKPOINT / "modeling_molmoact2.py").resolve()),
        "checkpoint_model_source_sha256": sha256(CHECKPOINT / "modeling_molmoact2.py"),
        "official_model_repository_path": "/workspace/models/molmoact2-official",
        "official_model_repository_revision": "66b87e64efd99dfd103241418113955cf64dfa9c",
        "official_lerobot_revision": "e595b7902714ba51f91e47523f66f89c5181b649",
        "checkpoint_repository_revision": "0d24a92bd1faf321ef497c3bbd5681af97c65aa2",
        "source_evidence_path": str(SOURCE_EVIDENCE.resolve()),
        "source_evidence_sha256": sha256(SOURCE_EVIDENCE),
        "official_evaluator_source_path": str(EVALUATOR.resolve()),
        "official_evaluator_source_sha256": sha256(EVALUATOR),
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

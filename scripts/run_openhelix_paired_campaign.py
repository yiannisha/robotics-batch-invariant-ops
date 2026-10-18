"""Wait for OpenHelix numerics, then run and summarize all paired CALVIN conditions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_openhelix_paired_calvin import CONDITIONS, sha256


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/workspace/envs/openhelix/bin/python")
FULL = ROOT / "results/cross_model/openhelix/full"
CLOSED = FULL / "closed_loop"
PAIRED = CLOSED / "paired"
RUNNER = ROOT / "scripts/run_openhelix_paired_calvin.py"
SUMMARIZER = ROOT / "scripts/summarize_openhelix_paired_calvin.py"
PREFIX_AUDITOR = ROOT / "scripts/audit_openhelix_paired_prefix.py"
POLICY = Path("/workspace/assets/openhelix/checkpoint/prompt_tuning_aux/policy.pth")
EVALUATOR = Path(
    "/workspace/models/openhelix/online_evaluation_calvin/"
    "evaluate_policy_lcb_pt_act_simple_asy10.py"
)
CLIP_OVERLAY = Path("/workspace/assets/openhelix/python_overlay")


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
            raise RuntimeError(
                f"PID {pid} no longer identifies predecessor {command_fragment!r}"
            )
        print(f"waiting for predecessor PID {pid}: {command_fragment}", flush=True)
        time.sleep(30)


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{CLIP_OVERLAY}:{ROOT}" + (
        f":{existing_pythonpath}" if existing_pythonpath else ""
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int, required=True)
    parser.add_argument(
        "--wait-for-command", default="run_openhelix_full_campaign.py"
    )
    args = parser.parse_args()
    if args.wait_for_pid <= 0 or not args.wait_for_command:
        raise ValueError("wait PID must be positive")
    write_json_atomic(CLOSED / "paired_supervisor.json", {
        "schema_version": 1,
        "pid": os.getpid(),
        "waiting_for_pid": args.wait_for_pid,
        "waiting_for_command": args.wait_for_command,
        "command": "run_openhelix_paired_campaign.py",
        "conditions": list(CONDITIONS),
        "sequences_per_condition": 1000,
    })
    wait_for_predecessor(args.wait_for_pid, args.wait_for_command)
    for condition in CONDITIONS:
        run([
            str(PYTHON), "-u", str(RUNNER), "--condition", condition,
            "--output", str(PAIRED / condition),
        ])
        prefix_audit = PAIRED / condition / "prefix_audit.json"
        run([
            str(PYTHON), str(PREFIX_AUDITOR), "--root", str(PAIRED / condition),
            "--output", str(prefix_audit),
        ])
        condition_audit = json.loads(prefix_audit.read_text())
        if (
            condition_audit.get("integrity_passed") is not True
            or condition_audit.get("complete") is not True
            or condition_audit.get("sequences_validated") != 1000
        ):
            raise RuntimeError(f"OpenHelix {condition} failed its independent audit")
    episodes = CLOSED / "episodes.jsonl.gz"
    summary = CLOSED / "paired_summary.json"
    run([
        str(PYTHON), str(SUMMARIZER), "--root", str(PAIRED),
        "--episodes", str(episodes), "--output", str(summary),
    ])
    summary_record = json.loads(summary.read_text())
    attempt = {
        "schema_version": 2,
        "model_id": "openhelix",
        "evaluator": "official OpenHelix CALVIN ABC-D 1000-sequence evaluator",
        "status": "completed",
        "conditions": list(CONDITIONS),
        "official_sequence_count": 1000,
        "paired_condition_sequence_records": 4000,
        "episodes_path": str(episodes.resolve()),
        "episodes_sha256": sha256(episodes),
        "paired_summary_path": str(summary.resolve()),
        "paired_summary_sha256": sha256(summary),
        "checkpoint_weight_path": str(POLICY.resolve()),
        "checkpoint_weight_sha256": sha256(POLICY),
        "source_evidence_path": str(EVALUATOR.resolve()),
        "source_evidence_sha256": sha256(EVALUATOR),
        "official_evaluator_source_path": str(EVALUATOR.resolve()),
        "official_evaluator_source_sha256": sha256(EVALUATOR),
        "paired_wrapper_path": str(RUNNER.resolve()),
        "paired_wrapper_sha256": sha256(RUNNER),
        "condition_metrics": summary_record["condition_metrics"],
        "paired_confidence_intervals": summary_record["paired_confidence_intervals"],
        "disagreement_counts": summary_record["disagreement_counts"],
        "action_divergence_counts": summary_record["action_divergence_counts"],
        "trajectory_divergence_counts": summary_record["trajectory_divergence_counts"],
    }
    write_json_atomic(CLOSED / "evaluator_attempt.json", attempt)
    print(json.dumps({
        "status": "completed", "paired_records": 4000,
        "episodes_sha256": attempt["episodes_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

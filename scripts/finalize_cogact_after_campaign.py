"""Wait for the live CogACT campaign, then independently pair and reduce it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from audit_cogact_paired_prefix import CONDITIONS, audit


ROOT = Path("/workspace/robotics-batch-invariant-ops")
PAIRED = ROOT / "results/cross_model/cogact_small/full/closed_loop/paired"
TASK_MANIFEST = PAIRED / "evaluator_task_manifest.json"


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.campaign_pid <= 0 or args.poll_seconds < 1:
        raise ValueError("invalid CogACT campaign PID or poll interval")
    while process_alive(args.campaign_pid):
        time.sleep(args.poll_seconds)
    for condition in CONDITIONS:
        result = audit(PAIRED / condition)
        if result.get("complete") is not True or result.get("episodes_cached") != 496:
            raise RuntimeError(
                f"CogACT campaign exited without complete {condition}: "
                f"{json.dumps(result, sort_keys=True)}"
            )
    pairing = PAIRED / "reset_pairing_audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_cogact_reset_pairing.py",
            "--root",
            str(PAIRED),
            "--task-manifest",
            str(TASK_MANIFEST),
            "--output",
            str(pairing),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_cogact_paired.py",
            "--root",
            str(PAIRED),
            "--task-manifest",
            str(TASK_MANIFEST),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/finalize_cogact_paired_record.py",
            "--paired-root",
            str(PAIRED),
            "--attempt-record",
            str(PAIRED.parent / "evaluator_attempt.json"),
            "--checkpoint-weight",
            "/workspace/assets/cogact_small/checkpoints/CogACT-Small.pt",
            "--source-evidence",
            "/workspace/models/cogact/README.md",
            "--official-evaluator-source",
            "/workspace/models/cogact/sim_cogact/cogact_policy.py",
            "--evaluator-task-manifest",
            str(TASK_MANIFEST),
        ],
        cwd=ROOT,
        check=True,
    )
    print("CogACT live campaign independently paired, reduced, and promoted", flush=True)


if __name__ == "__main__":
    main()

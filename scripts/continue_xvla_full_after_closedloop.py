"""Run X-VLA's remaining full-coverage stages after paired closed loop passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "results/cross_model/xvla/full/closed_loop/evaluator_attempt.json"
AUDIT = ROOT / "results/cross_model/full_coverage_audit.json"


def wait_for_closed_loop_supervisor(pid: int) -> None:
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
        if "continue_xvla_paired_libero.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies the X-VLA closed-loop supervisor")
        time.sleep(20)
    record = json.loads(ATTEMPT.read_text()) if ATTEMPT.is_file() else {}
    if record.get("model_id") != "xvla" or record.get("status") != "completed":
        raise RuntimeError("X-VLA closed-loop supervisor ended without audited completion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-loop-supervisor-pid", type=int, required=True)
    args = parser.parse_args()
    wait_for_closed_loop_supervisor(args.closed_loop_supervisor_pid)
    log_path = ROOT / "results/cross_model/xvla/full/remaining_campaign.log"
    with log_path.open("a") as log:
        result = subprocess.run(
            [sys.executable, "-u", "scripts/run_xvla_remaining_full_campaign.py"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"X-VLA remaining campaign exited {result.returncode}; see {log_path}")
    audit_result = subprocess.run([
        sys.executable, "scripts/audit_full_cross_model_campaign.py",
        "--results", "results/cross_model", "--output", str(AUDIT),
    ], cwd=ROOT, check=False)
    report = json.loads(AUDIT.read_text()) if AUDIT.is_file() else {}
    xvla = next((item for item in report.get("configurations", []) if item.get("id") == "xvla"), {})
    if not xvla.get("complete"):
        raise RuntimeError(f"X-VLA runner returned but independent full audit still fails: {xvla.get('issues')}")
    if audit_result.returncode not in (0, 1):
        raise RuntimeError(f"independent campaign audit crashed with {audit_result.returncode}")
    print("X-VLA full π0-equivalent coverage independently verified", flush=True)


if __name__ == "__main__":
    main()

"""Resume X-VLA full coverage after queued success-rate campaigns and probes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/cross_model/full_coverage_audit.json"


def wait_for_rdt_worker(pid: int) -> dict:
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
        if "continue_rdt_paired_after_wsa.py" not in cmdline:
            raise RuntimeError(f"PID {pid} no longer identifies the RDT paired worker")
        time.sleep(20)
    attempt = ROOT / "results/cross_model/rdt_1b/full/closed_loop/evaluator_attempt.json"
    return {"rdt_1b": json.loads(attempt.read_text()) if attempt.is_file() else {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdt-paired-pid", type=int, required=True)
    args = parser.parse_args()
    if args.rdt_paired_pid <= 0:
        raise ValueError("invalid RDT paired worker PID")
    predecessor = wait_for_rdt_worker(args.rdt_paired_pid)
    statuses = ", ".join(
        f"{key}: {value.get('status', 'missing')}" for key, value in predecessor.items()
    )
    print(
        "RDT paired predecessor ended; "
        f"statuses={{{statuses}}} "
        "(resuming X-VLA full coverage)",
        flush=True,
    )
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
        raise RuntimeError(f"X-VLA independent full audit still fails: {xvla.get('issues')}")
    if audit_result.returncode not in (0, 1):
        raise RuntimeError(f"independent campaign audit crashed with {audit_result.returncode}")
    print("X-VLA full pi0-equivalent coverage independently verified", flush=True)


if __name__ == "__main__":
    main()

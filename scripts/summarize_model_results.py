"""Print verified model results from research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def status(path: Path) -> tuple[str, str, str, str]:
    baseline_path = path / "baseline.json"
    fixed_path = path / "fixed.json"
    if not baseline_path.exists() and not fixed_path.exists():
        return path.name, "-", "NOT RUN", "NOT RUN"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    fixed = json.loads(fixed_path.read_text()) if fixed_path.exists() else None
    batches = "-"
    for report in (fixed, baseline):
        if report:
            batches = ",".join(str(value) for value in report["requested_batch_sizes"])
            break
    baseline_status = "PASS" if baseline and baseline["end_to_end_exact"] else "FAIL"
    fixed_status = "PASS" if fixed and fixed["end_to_end_exact"] else "FAIL"
    return path.name, batches, baseline_status, fixed_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-dir", type=Path, default=Path("research"))
    args = parser.parse_args()
    rows = [status(path) for path in sorted(args.research_dir.iterdir()) if path.is_dir()]
    headers = ("Model", "B tested", "Baseline", "batch_invariant_ops")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(4)]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(4)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(4)))


if __name__ == "__main__":
    main()

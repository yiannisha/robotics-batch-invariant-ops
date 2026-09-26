"""Print verified model results from research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def status(name: str, baseline_path: Path, fixed_path: Path) -> tuple[str, str, str, str]:
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    fixed = json.loads(fixed_path.read_text()) if fixed_path.exists() else None
    batches = "-"
    for report in (fixed, baseline):
        if report:
            batches = ",".join(str(value) for value in report["requested_batch_sizes"])
            break

    def result(report: dict | None) -> str:
        if report is None:
            return "NOT RUN"
        return "PASS" if report["end_to_end_exact"] else "FAIL"

    baseline_status = result(baseline)
    fixed_status = result(fixed)
    return name, batches, baseline_status, fixed_status


def statuses(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    baseline_path = path / "baseline.json"
    fixed_path = path / "fixed.json"
    if baseline_path.exists() or fixed_path.exists():
        rows.append(status(path.name, baseline_path, fixed_path))
    for variant_baseline in sorted(path.glob("*_baseline.json")):
        variant = variant_baseline.name.removesuffix("_baseline.json")
        rows.append(
            status(
                f"{path.name}/{variant}",
                variant_baseline,
                path / f"{variant}_fixed.json",
            )
        )
    if not rows:
        rows.append((path.name, "-", "NOT RUN", "NOT RUN"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-dir", type=Path, default=Path("research"))
    args = parser.parse_args()
    rows = [
        row
        for path in sorted(args.research_dir.iterdir())
        if path.is_dir()
        for row in statuses(path)
    ]
    headers = ("Model", "B tested", "Baseline", "batch_invariant_ops")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(4)]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(4)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(4)))


if __name__ == "__main__":
    main()

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_applicability_report_is_timestamped_and_partitions_all_paths(
    tmp_path: Path,
) -> None:
    completed = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/audit_closed_loop_applicability.py"),
        "--repository", str(ROOT),
        "--results", str(tmp_path),
    ], cwd=ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "closed_loop/applicability.json").read_text())
    generated = datetime.fromisoformat(report["generated_at_utc"])
    assert generated.tzinfo is not None
    assert report["configuration_count"] == 29
    assert report["applicable_public_simulator"] == 26
    assert report["not_applicable"] == 3
    assert report["applicable_public_simulator"] + report["not_applicable"] == 29
    assert report["missing_evidence"] == []

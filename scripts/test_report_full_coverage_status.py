"""Focused checks that progress reports cannot promote partial evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from report_full_coverage_status import main


class CoverageStatusTests(unittest.TestCase):
    def test_partial_and_probe_are_not_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_path = root / "full_coverage_audit.json"
            rows = []
            for index in range(29):
                rows.append({
                    "id": f"model_{index}", "complete": False,
                    "issues": ["incomplete"],
                    "components": {
                        "policy_timing": {"complete": False, "timed": 200 if index == 0 else 0},
                        "closed_loop": {
                            "complete": False,
                            "status": (
                                "probe_succeeded_requires_paired_campaign" if index == 0 else
                                "completed" if index == 1 else
                                "not_applicable" if index == 2 else None
                            ),
                        },
                    },
                })
            audit_path.write_text(json.dumps({
                "schema_version": 1, "configuration_count": 29,
                "full_complete": 0, "passed": False, "configurations": rows,
            }))
            paired = root / "model_0" / "full" / "closed_loop" / "paired" / "native_singleton" / "rollouts"
            paired.mkdir(parents=True)
            (paired / "task_0.json").write_text("{}")
            with mock.patch.object(sys, "argv", [
                "report_full_coverage_status.py", "--audit", str(audit_path),
                "--results", str(root),
            ]):
                main()
            aggregate = json.loads((root / "full_coverage_status.json").read_text())
            self.assertEqual(aggregate["full_audit_complete"], 0)
            self.assertEqual(aggregate["component_status_counts"]["policy_timing"], {
                "partial": 1, "missing": 28,
            })
            first = json.loads((root / "model_0" / "full" / "status.json").read_text())
            self.assertEqual(first["component_status"]["closed_loop"], "partial")
            self.assertEqual(first["active_paired_rollouts_by_condition"]["native_singleton"], 1)
            self.assertFalse(first["full_audit_complete"])
            self.assertEqual(aggregate["component_status_counts"]["closed_loop"], {
                "partial": 2, "not_applicable_unaccepted": 1, "missing": 26,
            })


if __name__ == "__main__":
    unittest.main()

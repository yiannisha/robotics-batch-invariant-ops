"""Tests for CogACT cross-condition pairing and reduction tools."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audit_cogact_reset_pairing import audit
from summarize_cogact_paired import first_divergence, summarize


TASK_MANIFEST = Path(
    "/workspace/robotics-batch-invariant-ops/results/cross_model/cogact_small/"
    "full/closed_loop/paired/evaluator_task_manifest.json"
)


class CogACTPairingToolsTest(unittest.TestCase):
    def test_first_divergence_is_step_exact(self) -> None:
        self.assertIsNone(first_divergence([[1], [2]], [[1], [2]]))
        self.assertEqual(first_divergence([[1], [2]], [[1], [3]]), 1)
        self.assertEqual(first_divergence([[1]], [[1], [2]]), 1)

    def test_empty_campaign_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = audit(Path(temporary), TASK_MANIFEST)
        self.assertFalse(report["fully_paired"])
        self.assertEqual(report["expected_resets_per_condition"], 496)
        self.assertTrue(any("prefix audit failed" in issue for issue in report["issues"]))

    def test_reducer_refuses_unpaired_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "before raw pairing passes"):
                summarize(Path(temporary), TASK_MANIFEST)


if __name__ == "__main__":
    unittest.main()

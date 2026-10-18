"""Tests for the independent CogACT paired-evidence auditor."""

from __future__ import annotations

import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from audit_cogact_paired_prefix import (
    EXPECTED_POSE_KEYS,
    audit,
    successes_in_initialization_order,
)


PROBE = Path(
    "/workspace/robotics-batch-invariant-ops/results/cross_model/cogact_small/"
    "full/closed_loop/probe/native_singleton"
)


class CogACTPrefixAuditTest(unittest.TestCase):
    def test_pose_contract_covers_each_pinned_environment(self) -> None:
        self.assertEqual(
            EXPECTED_POSE_KEYS,
            ("tcp_pose",) * 4 + ("base_pose",) * 6 + ("tcp_pose",) * 2,
        )

    def test_successes_are_ordered_by_numeric_initialization(self) -> None:
        self.assertEqual(
            successes_in_initialization_order({10: True, 2: False, 1: True}),
            [True, False, True],
        )

    def copy_probe(self, destination: Path) -> Path:
        shutil.copytree(PROBE, destination)
        manifest_path = destination / "condition_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        metadata = destination / "server_metadata.json"
        manifest["server_metadata_path"] = str(metadata.resolve())
        manifest_path.write_text(json.dumps(manifest))
        return destination

    def test_exact_probe_passes(self) -> None:
        report = audit(PROBE)
        self.assertTrue(report["integrity_passed"])
        self.assertTrue(report["complete"])
        self.assertEqual(report["episodes_cached"], 1)
        self.assertEqual(report["successes_cached"], 0)

    def test_wrong_request_local_noise_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copy_probe(Path(temporary) / "probe")
            rollout = next((root / "rollouts").glob("*.json"))
            payload = json.loads(rollout.read_text())
            payload["episodes"][0]["noise_sha256"][0] = "0" * 64
            rollout.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
                audit(root)

    def test_duplicate_episode_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copy_probe(Path(temporary) / "probe")
            source = next((root / "rollouts").glob("*.json"))
            shutil.copy2(source, root / "rollouts/task_0_initializations_0_0-copy.json")
            with self.assertRaises(ValueError):
                audit(root)

    def test_losslessly_compressed_probe_remains_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copy_probe(Path(temporary) / "probe")
            source = next((root / "rollouts").glob("*.json"))
            compressed = source.with_suffix(".json.gz")
            compressed.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
            source.unlink()
            report = audit(root)
            self.assertTrue(report["integrity_passed"])
            self.assertEqual(report["episodes_cached"], 1)


if __name__ == "__main__":
    unittest.main()

"""Fail-closed tests for the fixed-environment LeRobot pairing protocol."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_lerobot_reset_pairing import CONDITIONS, audit


class LeRobotResetPairingTests(unittest.TestCase):
    def _write_fixture(
        self, root: Path, *, protocol_version: int = 2, model: str = "xvla"
    ) -> None:
        for condition in CONDITIONS:
            folder = root / condition
            (folder / "rollouts").mkdir(parents=True)
            (folder / "official").mkdir()
            policy_batch = 1 if condition.endswith("singleton") else 4
            manifest = {
                "model": model, "condition": condition,
                "batch_size": policy_batch,
            }
            if protocol_version == 2:
                manifest.update({
                    "paired_protocol_version": 2,
                    "policy_batch_size": policy_batch,
                    "environment_batch_size": 4,
                })
            (folder / "condition_manifest.json").write_text(json.dumps(manifest))
            (folder / "official/eval_info.json").write_text(json.dumps({"overall": {}}))
            episodes = [{
                "task": 0,
                "initialization_id": initialization,
                "initial_raw_observation_sha256": f"{initialization + 1:064x}",
                "initial_observation_sha256": f"{initialization + 101:064x}",
                "first_noise_sha256": f"{initialization + 201:064x}",
            } for initialization in range(50)]
            (folder / "rollouts/task_0.json").write_text(json.dumps({"episodes": episodes}))

    def test_protocol_v2_accepts_fixed_raw_resets_and_within_mode_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            # A repaired preprocessor may legitimately differ from native, as
            # long as singleton/dynamic match within each implementation.
            for condition in ("patched_singleton", "patched_dynamic"):
                path = root / condition / "rollouts/task_0.json"
                payload = json.loads(path.read_text())
                for row in payload["episodes"]:
                    row["initial_observation_sha256"] = "f" * 64
                path.write_text(json.dumps(payload))
            report = audit(root, task_count=1)
            self.assertTrue(report["protocol_isolation_valid"])
            self.assertTrue(report["fully_paired"])
            self.assertEqual(report["raw_initial_observation_mismatch_count"], 0)
            self.assertEqual(report["within_implementation_initial_observation_mismatch_count"], 0)
            self.assertEqual(report["initial_observation_mismatch_count"], 50)

    def test_legacy_lerobot_environment_batch_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root, protocol_version=1)
            report = audit(root, task_count=1)
            self.assertFalse(report["protocol_isolation_valid"])
            self.assertFalse(report["fully_paired"])

    def test_molmoact2_also_requires_fixed_b4_physical_worlds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root, protocol_version=1, model="molmoact2")
            report = audit(root, task_count=1)
            self.assertFalse(report["protocol_isolation_valid"])
            self.assertFalse(report["fully_paired"])

    def test_within_native_processed_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            path = root / "native_dynamic/rollouts/task_0.json"
            payload = json.loads(path.read_text())
            payload["episodes"][7]["initial_observation_sha256"] = "e" * 64
            path.write_text(json.dumps(payload))
            report = audit(root, task_count=1)
            self.assertFalse(report["fully_paired"])
            self.assertEqual(report["within_implementation_initial_observation_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()

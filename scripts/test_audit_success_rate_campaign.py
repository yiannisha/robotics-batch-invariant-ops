from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import audit_success_rate_campaign as module


class SuccessRateCampaignAuditTests(TestCase):
    def test_all_configurations_must_pass_the_evaluator_audit(self) -> None:
        with TemporaryDirectory() as directory:
            results = Path(directory)

            def fake_audit(path: Path, identifier: str, issues: list[str]) -> dict:
                self.assertEqual(
                    path,
                    results / identifier / "full" / "closed_loop" / "evaluator_attempt.json",
                )
                if identifier == "pi_fast":
                    issues.append("paired campaign pending")
                    return {"complete": False, "status": "probe_succeeded_requires_paired_campaign"}
                return {"complete": True, "status": "completed"}

            with mock.patch.object(module, "evaluator_audit", side_effect=fake_audit):
                report = module.audit_campaign(results)

        self.assertEqual(report["configuration_count"], 29)
        self.assertEqual(report["complete_count"], 28)
        self.assertEqual(report["pending_count"], 1)
        self.assertFalse(report["passed"])
        pi_fast = next(row for row in report["configurations"] if row["id"] == "pi_fast")
        self.assertEqual(pi_fast["issues"], ["paired campaign pending"])
        self.assertIsNone(pi_fast["evaluator_attempt_sha256"])

    def test_report_passes_only_when_every_configuration_is_satisfied(self) -> None:
        with TemporaryDirectory() as directory:
            with mock.patch.object(
                module,
                "evaluator_audit",
                return_value={"complete": True, "status": "completed"},
            ):
                report = module.audit_campaign(Path(directory))

        self.assertTrue(report["passed"])
        self.assertEqual(report["complete_count"], 29)
        self.assertEqual(report["pending_count"], 0)
        self.assertEqual(report["status_counts"], {"completed": 29})

    def test_nonpermanent_blocker_requires_pinned_safe_probe_evidence(self) -> None:
        import hashlib
        import json

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "README.md"
            source.write_text("official evaluator requirements")
            manifest = root / "model.safetensors.index.json"
            manifest.write_text("checkpoint manifest")
            stdout = root / "probe.stdout.log"
            stdout.write_text("required checkpoint shard is absent\n")
            stderr = root / "probe.stderr.log"
            stderr.write_text("")
            capacity = root / "capacity.txt"
            capacity.write_text("available_bytes=1000 required_bytes=2000\n")

            def pinned(path: Path) -> dict:
                return {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "molmoact2",
                "status": "blocked",
                "blocker_type": "missing_exact_checkpoint",
                "evaluator": "official evaluator",
                "attempt_timestamp_utc": "2026-10-06T00:00:00Z",
                "attempted_command": ["python", "preflight.py"],
                "exit_code": 1,
                "stdout_path": str(stdout),
                "stderr_path": str(stderr),
                "failure_class": "checkpoint_capacity_preflight_failed",
                "probe_episodes_completed": 0,
                "paired_episodes_completed": 0,
                "source_evidence_path": str(source),
                "source_evidence_sha256": pinned(source)["sha256"],
                "checkpoint_availability": "missing",
                "checkpoint_manifest": pinned(manifest),
                "missing_checkpoint_artifacts": [{
                    "name": "model-00001-of-00001.safetensors",
                    "sha256": "a" * 64,
                    "size_bytes": 2000,
                }],
                "blocker_investigation": {
                    "safe_probe_exhausted": True,
                    "reason_permanent": False,
                    "resolution_attempts": ["Inspected the pinned manifest and filesystem capacity."],
                    "required_resolution": "Provide additional persistent storage.",
                    "evidence_files": [pinned(capacity)],
                },
            }))

            issues: list[str] = []
            result = module.evidence_backed_blocker_audit(report, "molmoact2", issues)
            self.assertTrue(result["complete"], issues)

            data = json.loads(report.read_text())
            data["blocker_investigation"]["safe_probe_exhausted"] = False
            report.write_text(json.dumps(data))
            issues = []
            self.assertFalse(module.evidence_backed_blocker_audit(report, "molmoact2", issues)["complete"])
            self.assertTrue(any("safe probing" in problem for problem in issues))

    def test_new_verified_checkpoint_receipt_invalidates_old_missing_blocker(self) -> None:
        import hashlib
        import json
        import os

        with TemporaryDirectory() as directory:
            full = Path(directory) / "model/full"
            closed = full / "closed_loop"
            closed.mkdir(parents=True)
            source = full / "source.md"
            source.write_text("official")
            manifest = full / "checkpoint.txt"
            manifest.write_text("manifest")
            stdout = full / "stdout.log"
            stdout.write_text("missing\n")
            stderr = full / "stderr.log"
            stderr.write_text("")
            evidence = full / "capacity.txt"
            evidence.write_text("too small")

            def pinned(candidate: Path) -> dict:
                return {
                    "path": str(candidate),
                    "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                }

            attempt = closed / "evaluator_attempt.json"
            attempt.write_text(json.dumps({
                "model_id": "model", "status": "blocked",
                "blocker_type": "missing_exact_checkpoint", "evaluator": "official",
                "attempt_timestamp_utc": "2026-10-06T00:00:00Z",
                "attempted_command": ["python", "probe.py"], "exit_code": 1,
                "stdout_path": str(stdout), "stderr_path": str(stderr),
                "failure_class": "missing", "probe_episodes_completed": 0,
                "paired_episodes_completed": 0,
                "source_evidence_path": str(source),
                "source_evidence_sha256": pinned(source)["sha256"],
                "checkpoint_availability": "missing",
                "checkpoint_manifest": pinned(manifest),
                "missing_checkpoint_artifacts": [{
                    "name": "model.safetensors", "sha256": "a" * 64,
                    "size_bytes": 1,
                }],
                "blocker_investigation": {
                    "safe_probe_exhausted": True, "reason_permanent": False,
                    "resolution_attempts": ["checked"],
                    "required_resolution": "fetch it", "evidence_files": [pinned(evidence)],
                },
            }))
            receipt = full / "checkpoint_cache_receipt.json"
            receipt.write_text(json.dumps({"all_pinned_files_verified": True}))
            newer = attempt.stat().st_mtime + 2
            os.utime(receipt, (newer, newer))

            issues: list[str] = []
            result = module.evidence_backed_blocker_audit(attempt, "model", issues)

            self.assertFalse(result["complete"])
            self.assertTrue(any("newer verified receipt" in issue for issue in issues))

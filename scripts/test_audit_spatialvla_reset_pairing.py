"""Fail-closed tests for SpatialVLA raw pairing reconstruction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_spatialvla_reset_pairing import (
    CHECKPOINT, CHECKPOINT_HASHES, CONDITIONS, audit, sha256,
)
from summarize_spatialvla_paired import summarize
from audit_full_cross_model_campaign import evaluator_audit


SOURCE_MANIFEST = Path(
    "results/cross_model/spatialvla/full/closed_loop/paired/evaluator_task_manifest.json"
)


class SpatialVLAPairingAuditTests(unittest.TestCase):
    def test_complete_schedule_passes_and_mutations_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "paired"; task_manifest = Path(temporary) / "tasks.json"
            task_manifest.write_bytes(SOURCE_MANIFEST.read_bytes())
            tasks = json.loads(task_manifest.read_text())["tasks"]
            task_hash = sha256(task_manifest)
            client_hash = sha256(Path("scripts/run_spatialvla_paired_client.py"))
            server_hash = sha256(Path("scripts/run_spatialvla_paired_server.py"))
            all_paths = []
            for condition in CONDITIONS:
                batch = 1 if condition.endswith("singleton") else 4
                folder = root / condition; (folder / "rollouts").mkdir(parents=True)
                metadata_path = folder / "server_metadata.json"
                metadata_path.write_text(json.dumps({
                    "protocol_version": "2.1", "policy_type": "spatialvla",
                    "checkpoint": str(CHECKPOINT.resolve()),
                    "task_suite": "simplerenv_spatialvla_semantic_12_variant",
                    "paired_condition": condition, "genuine_batched_policy_call": True,
                    "maximum_request_batch": batch,
                    "batch_invariant_ops": condition.startswith("patched"),
                    "deterministic_policy": True,
                    "unnorm_keys": {
                        "google_robot": "fractal20220817_data/0.1.0",
                        "widowx_bridge": "bridge_orig/1.0.0",
                    },
                    "image_size": [224, 224], "action_chunk_size": 4, "action_dim": 7,
                    "action_ensemble_temperature": -0.8, "sticky_gripper_repeat": 10,
                    "artifact_files": [
                        {"path": str((CHECKPOINT / name).resolve()), "sha256": digest}
                        for name, digest in CHECKPOINT_HASHES.items()
                    ],
                }))
                (folder / "condition_manifest.json").write_text(json.dumps({
                    "schema_version": 1, "model": "spatialvla", "condition": condition,
                    "batch_size": batch, "invariant_ops": condition.startswith("patched"),
                    "seed": 20261005,
                    "task_suite": "simplerenv_spatialvla_semantic_12_variant",
                    "task_manifest_sha256": task_hash, "task_count": 12,
                    "planned_initializations": 496, "deterministic_policy": True,
                    "explicit_physical_configuration": True,
                    "client_source_sha256": client_hash, "server_source_sha256": server_hash,
                    "server_metadata_path": str(metadata_path.resolve()),
                    "server_metadata_sha256": sha256(metadata_path),
                }))
                for task in tasks:
                    for start in range(0, task["initializations"], batch):
                        ids = list(range(start, min(start + batch, task["initializations"])))
                        episodes = []
                        for initialization in ids:
                            episodes.append({
                                "task": task["task_id"], "initialization_id": initialization,
                                "reset_seed": 20261005 + initialization,
                                "deterministic_policy": True,
                                "initial_observation_sha256": f"{task['task_id']:02x}{initialization:04x}".ljust(64, "0"),
                                "success": False, "trajectory": [{}, {}],
                                "action_sequence": [[0.0] * 7],
                                "raw_action_sequence": [[0.0] * 7],
                                "raw_action_chunks": [[[0.0] * 7 for _ in range(4)]],
                                "rewards": [0.0], "terminated": [False], "truncated": [False],
                                "policy_batch_sizes": [len(ids)],
                            })
                        path = folder / "rollouts" / f"task_{task['task_id']}_{start}.json"
                        path.write_text(json.dumps({
                            "task_id": task["task_id"], "explicit_initialization_ids": ids,
                            "seeds": [20261005 + value for value in ids], "episodes": episodes,
                        }))
                        all_paths.append(path)
                (folder / "official").mkdir()
                (folder / "official/eval_info.json").write_text(json.dumps({
                    "condition": condition,
                    "task_suite": "simplerenv_spatialvla_semantic_12_variant",
                    "overall": {"n_episodes": 496, "n_success": 0},
                }))
            complete = audit(root, task_manifest)
            self.assertTrue(complete["fully_paired"])
            (root / "reset_pairing_audit.json").write_text(json.dumps(complete))
            reduced = summarize(root, task_manifest)
            self.assertEqual(reduced["paired_episode_rows"], 1_984)
            summary = json.loads((root / "paired_summary.json").read_text())
            self.assertEqual(summary["success_counts"], dict.fromkeys(CONDITIONS, 0))
            self.assertEqual(len(summary["paired_confidence_intervals"]), 6)
            self.assertTrue(all(
                item["ci95"] == [0.0, 0.0]
                for item in summary["paired_confidence_intervals"].values()
            ))
            checkpoint_index = CHECKPOINT / "model.safetensors.index.json"
            source_evidence = Path("/workspace/models/spatialvla/README.md")
            evaluator_source = Path(
                "/workspace/models/simplerenv-openvla/simpler_env/evaluation/maniskill2_evaluator.py"
            )
            record = root / "evaluator_attempt.json"
            record.write_text(json.dumps({
                "schema_version": 1, "model_id": "spatialvla", "status": "completed",
                "checkpoint_weight_path": str(checkpoint_index.resolve()),
                "checkpoint_weight_sha256": CHECKPOINT_HASHES[checkpoint_index.name],
                "exact_checkpoint_files": [
                    {"path": str((CHECKPOINT / name).resolve()), "sha256": digest}
                    for name, digest in CHECKPOINT_HASHES.items()
                ],
                "source_evidence_path": str(source_evidence.resolve()),
                "source_evidence_sha256": sha256(source_evidence),
                "official_evaluator_source_path": str(evaluator_source.resolve()),
                "official_evaluator_source_sha256": sha256(evaluator_source),
                "paired_start_seed": 20261005, "paired_episodes_completed": 1984,
                "episodes_path": summary["episodes_path"],
                "episodes_sha256": summary["episodes_sha256"],
                "paired_summary_path": str((root / "paired_summary.json").resolve()),
                "paired_summary_sha256": sha256(root / "paired_summary.json"),
                "reset_pairing_audit_path": str((root / "reset_pairing_audit.json").resolve()),
                "reset_pairing_audit_sha256": sha256(root / "reset_pairing_audit.json"),
                "condition_eval_info": summary["condition_eval_info"],
                "evaluator_task_manifest_path": str(task_manifest.resolve()),
                "evaluator_task_manifest_sha256": sha256(task_manifest),
            }))
            issues = []
            self.assertTrue(evaluator_audit(record, "spatialvla", issues)["complete"], issues)

            missing = all_paths[-1]; payload = missing.read_bytes(); missing.unlink()
            report = audit(root, task_manifest)
            self.assertFalse(report["fully_paired"])
            self.assertTrue(any("physical group schedule differs" in issue for issue in report["issues"]))
            missing.write_bytes(payload)

            target = root / "patched_dynamic/rollouts/task_0_0.json"
            value = json.loads(target.read_text())
            value["episodes"][0]["initial_observation_sha256"] = "f" * 64
            target.write_text(json.dumps(value))
            report = audit(root, task_manifest)
            self.assertFalse(report["fully_paired"])
            self.assertTrue(any("differs physically" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()

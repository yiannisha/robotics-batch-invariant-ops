"""Focused fail-closed checks for the independent full-campaign timing gate."""

from __future__ import annotations

import json
import hashlib
import gzip
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audit_full_cross_model_campaign import (
    ARRANGEMENTS_PER_IMPLEMENTATION,
    COGACT_POSE_KEYS,
    EXPECTED_EXECUTION_CONFIG,
    EXPECTED_FIXED_LANGUAGE_SHAPE,
    SERVING_MODES,
    diagnostic_audit,
    evaluator_audit,
    openhelix_evaluator_evidence_audit,
    flow_summary_audit,
    operator_audit,
    pi_fast_flow_length,
    queue_audit,
    required_arrangements,
    request_multiplicity,
    spatialvla_flow_length,
    timing_audit,
    timing_input_selection_audit,
    univla_flow_length,
)
from audit_lerobot_reset_pairing import audit as audit_reset_pairing
from run_model_replay_invariance import unrelated_companion_rows
from summarize_model_replay_invariance import main as summarize_model_replay
from summarize_lerobot_paired_libero import (
    CONDITIONS,
    INTERNVLA_CURRENT_SERVER_SHA256,
    INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256,
    main as summarize_paired_libero,
    validate_official_successes,
    validate_stable_manifests,
)


class FullAuditTests(unittest.TestCase):
    def test_multi_series_flow_summary_preserves_action_and_video_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl.gz"
            fidelity = root / "fidelity.jsonl.gz"
            output = root / "summary.json"
            flow = root / "flow.json"
            metric = {
                "exact": True, "max_absolute_error": 0.0,
                "mean_absolute_error": 0.0, "rmse": 0.0,
                "relative_l2_error": 0.0,
            }
            with gzip.open(raw, "wt") as stream:
                for mode in ("native", "batch_invariant"):
                    stream.write(json.dumps({
                        "request_id": "r0", "batch_size": 2,
                        "implementation": mode, "output": metric,
                        "flow_steps": {
                            **{f"action.diffusion.step_{step:03d}": metric for step in range(2)},
                            **{f"video.diffusion.step_{step:03d}": metric for step in range(2)},
                        },
                    }) + "\n")
            with gzip.open(fidelity, "wt") as stream:
                stream.write(json.dumps({"request_id": "r0", "output": metric}) + "\n")
            with mock.patch.object(sys, "argv", [
                "summarize_model_replay_invariance.py",
                "--input", str(raw), "--fidelity", str(fidelity),
                "--output", str(output), "--flow-output", str(flow),
                "--flow-steps", "2",
                "--flow-series", "action=action.diffusion.step_{step:03d}",
                "--flow-series", "video=video.diffusion.step_{step:03d}",
            ]):
                summarize_model_replay()
            report = json.loads(flow.read_text())
            self.assertEqual(set(report["series"]), {"action", "video"})
            for series in ("action", "video"):
                self.assertEqual(
                    set(report["series"][series]["implementations"]),
                    {"native", "batch_invariant"},
                )
                self.assertEqual(
                    set(report["series"][series]["implementations"]["native"]),
                    {"0", "1"},
                )

    def test_noniterative_summary_writes_numerics_without_fake_flow_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl.gz"
            fidelity = root / "fidelity.jsonl.gz"
            output = root / "summary.json"
            flow = root / "flow.json"
            metric = {
                "exact": True, "max_absolute_error": 0.0,
                "mean_absolute_error": 0.0, "rmse": 0.0,
                "relative_l2_error": 0.0,
            }
            with gzip.open(raw, "wt") as stream:
                for mode in ("native", "batch_invariant"):
                    stream.write(json.dumps({
                        "request_id": "r0", "batch_size": 2,
                        "implementation": mode, "output": metric,
                        "flow_steps": {"actions.hidden": metric},
                    }) + "\n")
            with gzip.open(fidelity, "wt") as stream:
                stream.write(json.dumps({"request_id": "r0", "output": metric}) + "\n")
            with mock.patch.object(sys, "argv", [
                "summarize_model_replay_invariance.py",
                "--input", str(raw), "--fidelity", str(fidelity),
                "--output", str(output), "--flow-output", str(flow),
                "--noniterative",
            ]):
                summarize_model_replay()
            self.assertTrue(output.is_file())
            self.assertFalse(flow.exists())
            summary = json.loads(output.read_text())
            self.assertEqual(set(summary["implementations"]), {"native", "batch_invariant"})

    def test_noniterative_flow_na_is_reconstructed_from_raw_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            heldout = Path(directory) / "heldout"
            heldout.mkdir()
            evidence = Path(directory) / "adapter.py"
            evidence.write_text("def run_model(): pass\n")
            raw = heldout / "invariance.jsonl.gz"
            metric = {
                "exact": True, "unequal_element_count": 0,
                "max_absolute_error": 0.0, "mean_absolute_error": 0.0,
                "rmse": 0.0, "relative_l2_error": 0.0, "nonfinite_count": 0,
            }
            with gzip.open(raw, "wt") as stream:
                for mode in ("native", "batch_invariant"):
                    stream.write(json.dumps({
                        "batch_size": 2, "implementation": mode,
                        "flow_steps": {"actions.hidden": metric},
                    }) + "\n")
            report = heldout / "flow_summary.json"
            report.write_text(json.dumps({
                "schema_version": 1, "model_id": "one_shot", "status": "not_applicable",
                "reason": "The policy executes one feed-forward action prediction.",
                "evidence_path": str(evidence.resolve()),
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "source": str(raw.resolve()),
                "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "nonsingleton_records": 2,
                "implementations": ["native", "batch_invariant"],
                "observed_boundaries": ["actions.hidden"],
            }))
            issues = []
            self.assertTrue(flow_summary_audit(report, "one_shot", issues)["complete"])
            self.assertFalse(issues)
            with gzip.open(raw, "wt") as stream:
                stream.write(json.dumps({
                    "batch_size": 2, "implementation": "native",
                    "flow_steps": {"flow.step_00.state": metric},
                }) + "\n")
            report_data = json.loads(report.read_text())
            report_data["source_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            report.write_text(json.dumps(report_data))
            issues = []
            self.assertFalse(flow_summary_audit(report, "one_shot", issues)["complete"])
            self.assertTrue(any("not a noniterative boundary trace" in problem for problem in issues))

    def test_openvla_oft_canonical_padding_contract_is_fail_closed(self) -> None:
        self.assertEqual(EXPECTED_EXECUTION_CONFIG["openvla_oft"], {
            "native_language_padding": "longest_in_batch_right",
            "invariant_language_padding": "canonical_right",
            "language_max_length": 37,
            "language_pad_token_id": 32000,
        })
        self.assertEqual(EXPECTED_FIXED_LANGUAGE_SHAPE["openvla_oft"], {
            "language_padding": "canonical_right",
            "language_tokens": 37,
            "canonical_length": 37,
        })

    def test_univla_canonical_padding_contract_is_fail_closed(self) -> None:
        self.assertEqual(EXPECTED_EXECUTION_CONFIG["univla_baai"], {
            "native_language_padding": "longest_in_batch_right",
            "invariant_language_padding": "canonical_right",
            "language_max_length": 1293,
            "language_pad_token_id": 151643,
            "maximum_generated_action_tokens": 80,
            "decoding": "greedy_constrained_fast",
            "camera_resize_hw": [200, 200],
        })
        self.assertEqual(EXPECTED_FIXED_LANGUAGE_SHAPE["univla_baai"], {
            "language_padding": "canonical_right",
            "language_tokens": 1293,
            "canonical_length": 1293,
        })

    def test_pi_fast_flow_trace_requires_every_contiguous_token_step(self) -> None:
        metric = {
            "exact": True, "unequal_element_count": 0,
            "max_absolute_error": 0.0, "mean_absolute_error": 0.0,
            "rmse": 0.0, "relative_l2_error": 0.0, "nonfinite_count": 0,
        }
        complete = {
            f"flow.step_{step:03d}.prediction": dict(metric)
            for step in range(3)
        }
        self.assertEqual(pi_fast_flow_length(complete), 3)
        del complete["flow.step_001.prediction"]
        self.assertIsNone(pi_fast_flow_length(complete))

    def test_spatialvla_flow_trace_allows_boundaries_but_requires_contiguous_scores(self) -> None:
        metric = {
            "exact": True, "unequal_element_count": 0,
            "max_absolute_error": 0.0, "mean_absolute_error": 0.0,
            "rmse": 0.0, "relative_l2_error": 0.0, "nonfinite_count": 0,
        }
        complete = {
            "input.input_ids": dict(metric),
            **{
                f"generation.step_{step:03d}.scores": dict(metric)
                for step in range(3)
            },
            "actions.unnormalized": dict(metric),
        }
        self.assertEqual(spatialvla_flow_length(complete), 3)
        del complete["generation.step_001.scores"]
        self.assertIsNone(spatialvla_flow_length(complete))
        complete["flow.step_001.prediction"] = dict(metric)
        complete["flow.step_003.wrong"] = dict(metric)
        self.assertIsNone(pi_fast_flow_length(complete))

    def test_univla_flow_trace_requires_contiguous_constrained_scores(self) -> None:
        metric = {
            "exact": True, "unequal_element_count": 0,
            "max_absolute_error": 0.0, "mean_absolute_error": 0.0,
            "rmse": 0.0, "relative_l2_error": 0.0, "nonfinite_count": 0,
        }
        complete = {
            f"flow.step_{step:03d}.scores": metric for step in range(3)
        }
        self.assertEqual(univla_flow_length(complete), 3)
        complete.pop("flow.step_001.scores")
        self.assertIsNone(univla_flow_length(complete))

    def test_cogact_pose_contract_covers_each_pinned_environment(self) -> None:
        self.assertEqual(
            COGACT_POSE_KEYS,
            ("tcp_pose",) * 4 + ("base_pose",) * 6 + ("tcp_pose",) * 2,
        )

    def test_paired_reducer_accepts_exact_aggregate_only_official_successes(self) -> None:
        metric = {"n_episodes": 4, "n_success": 3, "pc_success": 75.0}
        self.assertEqual(
            validate_official_successes(
                metric, [True, False, True, True], condition="native_singleton", task=0,
            ),
            "aggregate_only",
        )
        for bad_metric in (
            {"n_episodes": 3, "n_success": 3},
            {"n_episodes": 4, "n_success": 2},
            {"n_episodes": 4, "n_success": True},
        ):
            with self.assertRaises(ValueError):
                validate_official_successes(
                    bad_metric, [True, False, True, True],
                    condition="native_singleton", task=0,
                )

    def test_internvla_standard_accepts_only_exact_proven_b1_wrapper_delta(self) -> None:
        manifests = {
            condition: {
                "model": "internvla_a15_standard",
                "seed": 20261005,
                "shared_rollout_wrapper_sha256": (
                    INTERNVLA_LEGACY_SINGLETON_SERVER_SHA256
                    if condition == "native_singleton"
                    else INTERNVLA_CURRENT_SERVER_SHA256
                ),
            }
            for condition in CONDITIONS
        }
        result = validate_stable_manifests(manifests)
        self.assertEqual(result["status"], "proven_b1_noop_source_patch")
        manifests["native_dynamic"]["shared_rollout_wrapper_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_stable_manifests(manifests)

    def test_short_diagnostic_cannot_pass_with_self_declared_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "test_model" / "full"
            (directory / "diagnostic").mkdir(parents=True)
            (directory / "replay_manifest.json").write_text(json.dumps({
                "observations": [{
                    "split": "diagnostic", "task": "task", "episode_id": 0,
                    "observation_index": 0, "replay_index": 0,
                    "request_ids": ["r0", "r1", "r2"],
                    "noise": [{"sha256": str(index)} for index in range(3)],
                    "companion_replay_indices": [],
                }],
            }))
            with gzip.open(directory / "diagnostic" / "invariance.jsonl.gz", "wt") as stream:
                stream.write("")
            with gzip.open(directory / "diagnostic" / "singleton_fidelity.jsonl.gz", "wt") as stream:
                stream.write("")
            issues = []
            result = diagnostic_audit(directory, issues)
            self.assertFalse(result["complete"])
            self.assertTrue(any("600" in problem for problem in issues))

    def test_diverse_companions_are_other_episodes_and_tasks(self) -> None:
        rows = [
            {"task": f"task-{task}", "episode_id": episode, "replay_index": 2 * (task * 4 + episode) + frame}
            for task in range(4) for episode in range(4) for frame in range(2)
        ]
        target = rows[0]
        companions = unrelated_companion_rows(rows, target, 0, 7)
        self.assertEqual(len(companions), 7)
        self.assertEqual(len({(row["task"], row["episode_id"]) for row in companions}), 7)
        self.assertTrue(all(row["task"] != target["task"] for row in companions))

    def test_required_arrangements_have_independent_permutation(self) -> None:
        arrangements = required_arrangements()
        self.assertEqual(len(arrangements), ARRANGEMENTS_PER_IMPLEMENTATION)
        self.assertIn(("permutation", 2, "last", "diverse"), arrangements)
        self.assertIn(("partition", 8, "first", "diverse"), arrangements)
        self.assertEqual(len(required_arrangements((1, 2, 4, 8, 16, 32))), 39)

    def test_deterministic_model_needs_pinned_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "decoder.py"
            evidence.write_text("temperature = 0.0\n")
            manifest = {
                "observations": [{"request_ids": ["request-1"]}],
                "deterministic_decoding_contract": {
                    "reason": "Greedy decoding has no stochastic flow noise.",
                    "evidence_path": str(evidence),
                    "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                },
            }
            issues = []
            self.assertEqual(request_multiplicity(manifest, issues), 1)
            self.assertFalse(issues)
            manifest["deterministic_decoding_contract"]["evidence_sha256"] = "wrong"
            self.assertEqual(request_multiplicity(manifest, issues), 0)
            self.assertTrue(issues)

    def test_timing_replay_selection_is_bound_to_frozen_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay_manifest.json"
            path.write_text(json.dumps({"observations": [
                {"replay_index": index, "split": "diagnostic", "episode_id": index,
                 "task_index": index, "request_ids": [f"request-{index}"]}
                for index in range(2)
            ]}))
            selection = {
                "replay_manifest_path": str(path),
                "replay_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "replay_indices": [0, 1], "episode_ids": [0, 1],
                "task_indices": [0, 1], "request_ids": ["request-0", "request-1"],
            }
            issues = []
            timing_input_selection_audit(selection, 2, issues)
            self.assertFalse(issues)
            selection["task_indices"] = [0, 0]
            timing_input_selection_audit(selection, 2, issues)
            self.assertTrue(any("task_indices" in problem for problem in issues))

    def test_timing_replay_selection_must_cover_available_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay_manifest.json"
            rows = [
                {"replay_index": 0, "split": "diagnostic", "episode_id": 0,
                 "task_index": 0, "request_ids": ["r0"]},
                {"replay_index": 1, "split": "diagnostic", "episode_id": 1,
                 "task_index": 0, "request_ids": ["r1"]},
                {"replay_index": 2, "split": "diagnostic", "episode_id": 2,
                 "task_index": 1, "request_ids": ["r2"]},
            ]
            path.write_text(json.dumps({"observations": rows}))
            selection = {
                "replay_manifest_path": str(path),
                "replay_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "replay_indices": [0, 1], "episode_ids": [0, 1],
                "task_indices": [0, 0], "request_ids": ["r0", "r1"],
            }
            issues = []
            timing_input_selection_audit(selection, 2, issues)
            self.assertTrue(any("distinct diagnostic tasks" in problem for problem in issues))

    def test_successful_evaluator_probe_is_not_paired_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info_path = root / "eval_info.json"
            info_path.write_text(json.dumps({
                "per_task": [{"task_id": 0}],
                "overall": {"n_episodes": 1, "n_success": 0},
            }))
            report_path = root / "evaluator_attempt.json"
            report_path.write_text(json.dumps({
                "model_id": "smolvla", "status": "probe_succeeded_requires_paired_campaign",
                "exit_code": 0, "probe_eval_info_path": str(info_path),
                "probe_eval_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
                "probe_tasks_completed": 1, "probe_episodes_completed": 1,
                "probe_successes": 0, "paired_episodes_completed": 0,
            }))
            issues = []
            self.assertFalse(evaluator_audit(report_path, "smolvla", issues)["complete"])
            self.assertTrue(any("paired four-condition campaign is pending" in problem for problem in issues))

    def test_openhelix_runnable_preflight_is_not_a_model_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition_manifest = root / "condition_manifest.json"
            condition_manifest.write_text(json.dumps({
                "model": "openhelix", "condition": "native_dynamic",
                "official_sequence_count": 1000, "sequence_count": 1000,
                "dynamic_batch_schedule": [1, 2, 4, 8],
                "target_position_schedule": ["first", "middle", "last"],
                "request_local_noise": True,
                "policy_calls_per_subtask": 60, "execute_len": 20,
            }))
            condition_hash = hashlib.sha256(condition_manifest.read_bytes()).hexdigest()
            preflight = root / "preflight.json"
            preflight.write_text(json.dumps({
                "passed": True, "sequence_generator_count": 1000,
                "condition_manifest_sha256": condition_hash,
            }))
            evidence = {}
            for name in ("runtime", "assets", "checkpoint", "source", "wrapper"):
                path = root / name
                path.write_text(name)
                evidence[name] = path
            report = root / "evaluator_attempt.json"
            payload = {
                "model_id": "openhelix", "status": "ready_pending_model_campaign",
                "paired_preflight_path": str(preflight),
                "paired_preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
                "paired_condition_manifest_path": str(condition_manifest),
                "paired_condition_manifest_sha256": condition_hash,
                "probe_episodes_completed": 0, "paired_episodes_completed": 0,
            }
            for name, path in evidence.items():
                key = {
                    "runtime": "runtime_probe", "assets": "evaluator_assets",
                    "checkpoint": "checkpoint_weight", "source": "source_evidence",
                    "wrapper": "paired_wrapper",
                }[name]
                payload[f"{key}_path"] = str(path)
                payload[f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report.write_text(json.dumps(payload))
            issues = []
            self.assertFalse(evaluator_audit(report, "openhelix", issues)["complete"])
            self.assertEqual(issues, [
                "closed loop: openhelix evaluator is runnable, but no model-backed paired episode has completed"
            ])

    def test_molmoact2_stream_preflight_is_not_a_model_episode(self) -> None:
        report = Path(
            "/workspace/robotics-batch-invariant-ops/results/cross_model/"
            "molmoact2/full/closed_loop/evaluator_attempt.json"
        )
        issues = []
        self.assertFalse(evaluator_audit(report, "molmoact2", issues)["complete"])
        self.assertIn(
            "closed loop: MolmoAct2 exact model call succeeded, but no model-backed "
            "evaluator episode has completed",
            issues,
        )

    def test_univla_exact_checkpoint_preflight_is_not_a_model_episode(self) -> None:
        report = Path(
            "/workspace/robotics-batch-invariant-ops/results/cross_model/"
            "univla_baai/full/closed_loop/evaluator_attempt.json"
        )
        issues = []
        self.assertFalse(evaluator_audit(report, "univla_baai", issues)["complete"])
        self.assertEqual(issues, [
            "closed loop: UniVLA exact checkpoint/evaluator preflight passes, but no "
            "model-backed episode has completed"
        ])

    def test_univla_completed_claim_requires_sealed_exact_checkpoint_and_raw_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.jsonl"
            episodes.write_text("")
            checkpoint = root / "model.safetensors.index.json"
            checkpoint.write_text("{}")
            source = root / "model_wrapper_emu.py"
            source.write_text("official policy")
            evaluator = root / "evaluate_libero_emu.py"
            evaluator.write_text("official evaluator")
            receipt = root / "checkpoint_cache_receipt.json"
            receipt.write_text(json.dumps({
                "all_pinned_files_verified": True,
                "storage": "sealed_memfd", "files": {},
            }))
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps({
                "model_id": "univla_baai", "task_suite": "libero_10",
                "task_ids": list(range(10)), "tasks": [],
            }))
            summary = root / "paired_summary.json"
            summary.write_text(json.dumps({"condition_manifests": {}}))
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "univla_baai", "status": "completed",
                "episodes_path": str(episodes), "paired_start_seed": 20261015,
                "checkpoint_weight_path": str(checkpoint),
                "checkpoint_weight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "checkpoint_receipt_path": str(receipt),
                "checkpoint_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "source_evidence_path": str(source),
                "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "checkpoint_model_source_path": str(source),
                "checkpoint_model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "official_evaluator_source_path": str(evaluator),
                "official_evaluator_source_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
                "evaluator_task_manifest_path": str(tasks),
                "evaluator_task_manifest_sha256": hashlib.sha256(tasks.read_bytes()).hexdigest(),
                "paired_summary_path": str(summary),
                "official_model_repository_revision": "invented",
                "official_lerobot_revision": "invented",
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "univla_baai", issues)["complete"])
            self.assertTrue(any(
                "exact sealed checkpoint load differs" in problem for problem in issues
            ))
            self.assertTrue(any(
                "lacks four pinned evaluator condition manifests" in problem
                for problem in issues
            ))

    def test_openhelix_completed_claim_cannot_use_empty_episode_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.jsonl.gz"
            with gzip.open(episodes, "wt"):
                pass
            record = {
                "episodes_sha256": hashlib.sha256(episodes.read_bytes()).hexdigest(),
            }
            issues = []
            openhelix_evaluator_evidence_audit(record, episodes, issues)
            self.assertTrue(any("0/4000 unique paired sequence records" in item for item in issues))

    def test_internvla_completed_claim_requires_raw_condition_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.jsonl"
            episodes.write_text("")
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            source = root / "eval_libero_server_client.py"
            source.write_text("official evaluator")
            task_manifest = root / "tasks.json"
            task_manifest.write_text(json.dumps({"task_ids": list(range(10))}))
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "internvla_a15_standard", "status": "completed",
                "episodes_path": str(episodes), "paired_start_seed": 20261005,
                "checkpoint_weight_path": str(checkpoint),
                "checkpoint_weight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_evidence_path": str(source),
                "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "official_evaluator_source_path": str(source),
                "official_evaluator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "evaluator_task_manifest_path": str(task_manifest),
                "evaluator_task_manifest_sha256": hashlib.sha256(task_manifest.read_bytes()).hexdigest(),
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "internvla_a15_standard", issues)["complete"])
            self.assertTrue(any(
                "lacks four pinned evaluator condition manifests" in problem for problem in issues
            ))

    def test_openvla_completed_claim_requires_exact_checkpoint_and_raw_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.jsonl"
            episodes.write_text("")
            checkpoint = root / "model.safetensors.index.json"
            checkpoint.write_bytes(b"not-the-exact-index")
            source = root / "run_libero_eval.py"
            source.write_text("official evaluator")
            task_manifest = root / "tasks.json"
            task_manifest.write_text(json.dumps({
                "schema_version": 1,
                "task_suite": "libero_spatial",
                "task_ids": list(range(10)),
                "task_descriptions": ["invented"] * 10,
                "initializations_per_task": 50,
                "checkpoint_files": [],
            }))
            summary = root / "paired_summary.json"
            summary.write_text(json.dumps({"condition_manifests": {}}))
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "openvla_oft", "status": "completed",
                "episodes_path": str(episodes), "paired_start_seed": 20261005,
                "checkpoint_weight_path": str(checkpoint),
                "checkpoint_weight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_evidence_path": str(source),
                "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "official_evaluator_source_path": str(source),
                "official_evaluator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "evaluator_task_manifest_path": str(task_manifest),
                "evaluator_task_manifest_sha256": hashlib.sha256(task_manifest.read_bytes()).hexdigest(),
                "paired_summary_path": str(summary),
                "paired_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "exact_checkpoint_files": [],
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "openvla_oft", issues)["complete"])
            self.assertTrue(any("exact task/checkpoint manifest differs" in problem
                                for problem in issues))
            self.assertTrue(any("lacks four pinned evaluator condition manifests" in problem
                                for problem in issues))

    def test_wsa_completed_claim_requires_exact_artifacts_and_raw_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.jsonl"
            episodes.write_text("")
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"not-wsa")
            source = root / "inference.py"
            source.write_text("official evaluator")
            task_manifest = root / "tasks.json"
            task_manifest.write_text(json.dumps({
                "schema_version": 1, "task_suite": "libero_10",
                "task_ids": list(range(10)), "task_descriptions": ["invented"] * 10,
                "initializations_per_task": 50, "artifact_files": [], "checkpoint_files": [],
            }))
            summary = root / "paired_summary.json"
            summary.write_text(json.dumps({"condition_manifests": {}}))
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "wsa_base", "status": "completed",
                "episodes_path": str(episodes), "paired_start_seed": 20261005,
                "checkpoint_weight_path": str(checkpoint),
                "checkpoint_weight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_evidence_path": str(source),
                "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "official_evaluator_source_path": str(source),
                "official_evaluator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "evaluator_task_manifest_path": str(task_manifest),
                "evaluator_task_manifest_sha256": hashlib.sha256(task_manifest.read_bytes()).hexdigest(),
                "paired_summary_path": str(summary),
                "paired_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "exact_checkpoint_files": [],
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "wsa_base", issues)["complete"])
            self.assertTrue(any("exact task/checkpoint manifest differs" in problem
                                for problem in issues))
            self.assertTrue(any("lacks four pinned evaluator condition manifests" in problem
                                for problem in issues))

    def test_rdt_completed_claim_requires_five_task_exact_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.jsonl"
            episodes.write_text("")
            checkpoint = root / "mp_rank_00_model_states.pt"
            checkpoint.write_bytes(b"not-rdt")
            source = root / "eval_rdt_maniskill.py"
            source.write_text("official evaluator")
            task_manifest = root / "tasks.json"
            task_manifest.write_text(json.dumps({
                "schema_version": 1, "task_suite": "maniskill_five",
                "task_ids": list(range(5)), "task_names": ["invented"] * 5,
                "initializations_per_task": 50, "artifact_files": [], "checkpoint_files": [],
            }))
            summary = root / "paired_summary.json"
            summary.write_text(json.dumps({"condition_manifests": {}}))
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "rdt_1b", "status": "completed",
                "episodes_path": str(episodes), "paired_start_seed": 20241201,
                "checkpoint_weight_path": str(checkpoint),
                "checkpoint_weight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_evidence_path": str(source),
                "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "official_evaluator_source_path": str(source),
                "official_evaluator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "evaluator_task_manifest_path": str(task_manifest),
                "evaluator_task_manifest_sha256": hashlib.sha256(task_manifest.read_bytes()).hexdigest(),
                "paired_summary_path": str(summary),
                "paired_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "exact_checkpoint_files": [],
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "rdt_1b", issues)["complete"])
            self.assertTrue(any("exact task/checkpoint manifest differs" in problem
                                for problem in issues))
            self.assertTrue(any("lacks four pinned evaluator condition manifests" in problem
                                for problem in issues))

    def test_missing_dependency_is_not_an_evaluator_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = root / "stderr.log"
            stdout = root / "stdout.log"
            checkpoint = root / "model.safetensors"
            source = root / "eval.py"
            stderr.write_text("ModuleNotFoundError: missing package\n")
            stdout.write_text("")
            checkpoint.write_bytes(b"checkpoint")
            source.write_text("official evaluator")
            report_path = root / "evaluator_attempt.json"
            report_path.write_text(json.dumps({
                "model_id": "smolvla", "status": "blocked",
                "attempted_command": ["python", "eval.py"],
                "attempt_timestamp_utc": "2026-10-05T00:00:00Z", "exit_code": 1,
                "failure_class": "missing_evaluator_dependency", "evaluator": "official",
                "stderr_path": str(stderr), "stdout_path": str(stdout),
                "checkpoint_weight_path": str(checkpoint),
                "checkpoint_weight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_evidence_path": str(source),
                "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }))
            issues = []
            self.assertFalse(evaluator_audit(report_path, "smolvla", issues)["complete"])
            self.assertTrue(any("remediable setup/timeout failure" in problem for problem in issues))

    def test_paired_evaluator_cannot_pass_one_task_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_path = root / "episodes.jsonl"
            conditions = (
                "native_singleton", "native_dynamic",
                "patched_singleton", "patched_dynamic",
            )
            with episodes_path.open("w") as stream:
                for initialization in range(50):
                    for condition in conditions:
                        stream.write(json.dumps({
                            "task": 0, "initialization_id": initialization,
                            "condition": condition, "success": False,
                            "reset_seed": 1000 + initialization,
                            "initial_observation_sha256": "a" * 64,
                            "trajectory": [[0.0]], "action_sequence": [[0.0]],
                        }) + "\n")
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "xvla", "status": "completed",
                "episodes_path": str(episodes_path), "paired_start_seed": 1000,
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "xvla", issues)["complete"])
            self.assertTrue(any("every task and 50 resets" in problem for problem in issues))
            report.write_text(json.dumps({
                "model_id": "pi_fast", "status": "completed",
                "episodes_path": str(episodes_path), "paired_start_seed": 1000,
            }))
            pi_fast_issues = []
            self.assertFalse(evaluator_audit(report, "pi_fast", pi_fast_issues)["complete"])
            self.assertTrue(any("pinned byte-identical official tokenizer provenance" in problem
                                for problem in pi_fast_issues))

    def test_completed_lerobot_evaluator_requires_raw_physical_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition_root = root / "native_singleton"
            condition_root.mkdir()
            manifest = condition_root / "condition_manifest.json"
            manifest.write_text(json.dumps({
                "model": "xvla", "condition": "native_singleton",
                "batch_size": 1, "invariant_ops": False,
                "explicit_init_state_by_initialization_id": True,
                "episodes_per_task": 50, "seed": 1000,
            }))
            summary = root / "paired_summary.json"
            summary.write_text(json.dumps({
                "condition_manifests": {
                    "native_singleton": {
                        "path": str(manifest),
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    },
                },
            }))
            episodes = root / "episodes.jsonl"
            episodes.write_text("")
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "xvla", "status": "completed",
                "episodes_path": str(episodes), "paired_start_seed": 1000,
                "paired_summary_path": str(summary),
                "paired_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "xvla", issues)["complete"])
            self.assertTrue(any("0/500 physical rollout batches" in problem for problem in issues))
            self.assertTrue(any("raw physical reset coverage differs (0/500)" in problem
                                for problem in issues))

    def test_lerobot_protocol_v2_audits_fixed_b4_physical_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition_root = root / "native_singleton"
            condition_root.mkdir()
            manifest = condition_root / "condition_manifest.json"
            manifest.write_text(json.dumps({
                "model": "xvla", "condition": "native_singleton",
                "batch_size": 1, "policy_batch_size": 1,
                "environment_batch_size": 4,
                "execution_mode": "serial_singleton",
                "paired_protocol_version": 2, "smoke_test": False,
                "invariant_ops": False,
                "explicit_init_state_by_initialization_id": True,
                "episodes_per_task": 50, "seed": 1000,
            }))
            summary = root / "paired_summary.json"
            summary.write_text(json.dumps({
                "condition_manifests": {
                    "native_singleton": {
                        "path": str(manifest),
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    },
                },
            }))
            episodes = root / "episodes.jsonl"
            episodes.write_text("")
            report = root / "evaluator_attempt.json"
            report.write_text(json.dumps({
                "model_id": "xvla", "status": "completed",
                "episodes_path": str(episodes), "paired_start_seed": 1000,
                "paired_summary_path": str(summary),
                "paired_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            }))
            issues = []
            self.assertFalse(evaluator_audit(report, "xvla", issues)["complete"])
            self.assertTrue(any("0/130 physical rollout batches" in problem for problem in issues))
            self.assertTrue(any("raw physical reset coverage differs (0/520)" in problem
                                for problem in issues))

    def test_paired_reducer_keeps_reduced_run_in_pilot_tier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "paired_pilot"
            conditions = {
                "native_singleton": 1, "native_dynamic": 4,
                "patched_singleton": 1, "patched_dynamic": 4,
            }
            for condition, batch_size in conditions.items():
                condition_root = root / condition
                (condition_root / "official").mkdir(parents=True)
                (condition_root / "rollouts").mkdir()
                (condition_root / "condition_manifest.json").write_text(json.dumps({
                    "model": "xvla", "condition": condition,
                    "batch_size": batch_size,
                    "invariant_ops": condition.startswith("patched"),
                    "explicit_init_state_by_initialization_id": True,
                    "episodes_per_task": 1, "seed": 1000,
                    "checkpoint": "same", "checkpoint_weight": "same",
                    "checkpoint_sha256": "same", "official_evaluator_sha256": "same",
                    "official_policy_sha256": "same", "paired_wrapper_sha256": "same",
                }))
                per_task = []
                for task in range(10):
                    per_task.append({
                        "task_id": task, "n_success": 0, "pc_success_ci95": [0.0, 79.0],
                        "metrics": {"successes": [False]},
                    })
                    (condition_root / "rollouts" / f"task_{task}.json").write_text(json.dumps({
                        "seeds": [1000], "explicit_init_state_ids": [0],
                        "episodes": [{
                            "task": task, "initialization_id": 0, "reset_seed": 1000,
                            "initial_observation_sha256": "a" * 64,
                            "first_noise_sha256": "b" * 64,
                            "success": False, "trajectory": [[0.0]],
                            "action_sequence": [[0.0]],
                        }],
                    }))
                (condition_root / "official" / "eval_info.json").write_text(json.dumps({"per_task": per_task}))
            with mock.patch.object(sys, "argv", [
                "summarize_lerobot_paired_libero.py", "--root", str(root),
                "--initializations-per-task", "1",
            ]):
                summarize_paired_libero()
            report = json.loads((root / "paired_summary.json").read_text())
            self.assertEqual(report["evidence_tier"], "pilot")
            self.assertEqual(report["paired_episodes"], 40)
            self.assertEqual(len(report["paired_confidence_intervals"]), 6)
            self.assertEqual(set(report["condition_manifests"]), set(CONDITIONS))

    def test_paired_reducer_supports_pinned_five_task_full_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rdt_full"
            conditions = {
                "native_singleton": 1, "native_dynamic": 4,
                "patched_singleton": 1, "patched_dynamic": 4,
            }
            for condition, batch_size in conditions.items():
                condition_root = root / condition
                (condition_root / "official").mkdir(parents=True)
                (condition_root / "rollouts").mkdir()
                (condition_root / "condition_manifest.json").write_text(json.dumps({
                    "model": "rdt_1b", "condition": condition, "batch_size": batch_size,
                    "invariant_ops": condition.startswith("patched"),
                    "explicit_init_state_by_initialization_id": True,
                    "episodes_per_task": 50, "seed": 1000,
                }))
                per_task = []
                for task in range(5):
                    successes = [False] * 50
                    per_task.append({
                        "task_id": task, "n_success": 0, "n_episodes": 50,
                        "pc_success_ci95": [0.0, 7.1], "metrics": {"successes": successes},
                    })
                    episodes = [{
                        "task": task, "initialization_id": initialization,
                        "reset_seed": 1000 + initialization,
                        "initial_observation_sha256": (
                            "f" * 64
                            if condition.startswith("patched") and task == 0 and initialization == 0
                            else f"{task:02x}{initialization:02x}".ljust(64, "a")
                        ),
                        "first_noise_sha256": f"{task:02x}{initialization:02x}".ljust(64, "b"),
                        "success": False, "trajectory": [[0.0]],
                        "action_sequence": [[0.0]],
                    } for initialization in range(50)]
                    (condition_root / "rollouts" / f"task_{task}.json").write_text(json.dumps({
                        "seeds": list(range(1000, 1050)),
                        "explicit_init_state_ids": list(range(50)),
                        "episodes": episodes,
                    }))
                (condition_root / "official" / "eval_info.json").write_text(json.dumps({
                    "per_task": per_task, "overall": {"n_episodes": 250},
                }))
            with mock.patch.object(sys, "argv", [
                "summarize_lerobot_paired_libero.py", "--root", str(root),
                "--task-count", "5",
            ]):
                summarize_paired_libero()
            report = json.loads((root / "paired_summary.json").read_text())
            self.assertEqual(report["evidence_tier"], "full")
            self.assertEqual(report["tasks"], 5)
            self.assertEqual(report["paired_episodes"], 1000)
            self.assertEqual(
                report["paired_sample_counts"]["native_singleton|native_dynamic"], 250
            )
            self.assertEqual(
                report["paired_sample_counts"]["native_singleton|patched_singleton"], 249
            )
            self.assertEqual(
                report["excluded_initial_observation_mismatch_counts"][
                    "native_singleton|patched_singleton"
                ],
                1,
            )
            pairing = audit_reset_pairing(root, task_count=5)
            self.assertTrue(pairing["fully_paired"])
            self.assertEqual(pairing["requested_resets_per_condition"], 250)

    def _write_timing(self, path: Path, *, missing_warmup=False, pilot_schedule=False):
        rows = [
            {"replay_index": index, "split": "diagnostic", "episode_id": index,
             "task_index": index, "request_ids": [f"request-{index}"]}
            for index in range(8)
        ]
        manifest_path = path.parent / "replay_manifest.json"
        manifest_path.write_text(json.dumps({"observations": rows}))
        environment = {
            "record_type": "environment", "campaign_protocol": "full_v1",
            "sessions": 2 if pilot_schedule else 5,
            "warmup": 2 if pilot_schedule else 20,
            "repetitions": 5 if pilot_schedule else 200,
            "logical_batch_size": 4,
            "fixed_batch_size": 8,
            "input_selection": {
                "replay_manifest_path": str(manifest_path),
                "replay_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "replay_indices": list(range(8)),
                "episode_ids": list(range(8)), "task_indices": list(range(8)),
                "request_ids": [f"request-{index}" for index in range(8)],
            },
        }
        with path.open("w") as stream:
            stream.write(json.dumps(environment) + "\n")
            stream.write(json.dumps({
                "record_type": "fixed_shape", "metadata": {"physical_batch_size": 8},
            }) + "\n")
            for mode in SERVING_MODES:
                for session in range(5):
                    for index in range(20):
                        if missing_warmup and mode == SERVING_MODES[0] and session == 0 and index == 19:
                            continue
                        stream.write(json.dumps({
                            "record_type": "warmup", "configuration": mode,
                            "session": session, "attempt_id": f"{mode}-{session}",
                            "warmup_index": index,
                        }) + "\n")
                    for repetition in range(200):
                        stream.write(json.dumps({
                            "record_type": "timing", "configuration": mode,
                            "session": session, "repetition": repetition,
                            "latency_seconds": 0.01,
                            "logical_batch_size": 4,
                            "physical_batch_size": 8 if mode == "fixed_size_padded" else 1 if mode == "serial_singleton" else 4,
                            "requests_per_second": 400.0,
                            "peak_incremental_bytes": 0,
                        }) + "\n")

    def test_full_schedule_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.jsonl"
            self._write_timing(path)
            issues = []
            result = timing_audit(path, "serving", issues)
            self.assertTrue(result["complete"])
            self.assertFalse(issues)
            self.assertEqual(result["timed"], 4000)
            self.assertEqual(result["warmups"], 400)

    def test_missing_actual_warmup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.jsonl"
            self._write_timing(path, missing_warmup=True)
            issues = []
            self.assertFalse(timing_audit(path, "serving", issues)["complete"])
            self.assertTrue(any("preceding warmups" in problem for problem in issues))

    def test_self_declared_pilot_schedule_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.jsonl"
            self._write_timing(path, pilot_schedule=True)
            issues = []
            self.assertFalse(timing_audit(path, "serving", issues)["complete"])
            self.assertTrue(any("5x20x200" in problem for problem in issues))

    def test_unpinned_timing_inputs_fail_even_with_full_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.jsonl"
            self._write_timing(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0].pop("input_selection")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            issues = []
            self.assertFalse(timing_audit(path, "serving", issues)["complete"])
            self.assertTrue(any("missing frozen diagnostic input selection" in problem for problem in issues))

    def test_missing_fixed_padded_shape_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.jsonl"
            self._write_timing(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows = [row for row in rows if row.get("record_type") != "fixed_shape"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            issues = []
            self.assertFalse(timing_audit(path, "serving", issues)["complete"])
            self.assertTrue(any("expected one fixed-padded input-shape record" in problem for problem in issues))

    def test_malformed_queue_record_fails_without_summary_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text(json.dumps({
                "configuration": "native_dynamic", "offered_load": "underload",
                "request_id": "request-0", "arrival_time": 0.0,
                "offered_rate_requests_per_second": 1.0,
                "reference_capacity_requests_per_second": 4.0,
            }) + "\n")
            issues = []
            self.assertFalse(queue_audit(path, issues)["complete"])
            self.assertTrue(any("missing request field" in problem for problem in issues))

    def test_operator_manifest_cannot_omit_traced_sdpa(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "operator"
            root.mkdir()
            (root / "raw.jsonl").write_text("")
            (root / "configurations.json").write_text(json.dumps({
                "configurations": [{
                    "id": "matrix", "source_operation": "observed matrix",
                    "operation": "mm", "m_per_request": 1, "k": 1, "n": 1,
                    "layout": "contiguous",
                }],
            }))
            (root / "aten_trace_B8.json").write_text(json.dumps({
                "operations": [{
                    "aten_operation": "aten::scaled_dot_product_attention",
                    "arguments": [
                        {"shape": [8, 1, 1, 1], "stride": [1, 1, 1, 1]}
                        for _ in range(3)
                    ],
                }],
            }))
            issues = []
            self.assertFalse(operator_audit(root / "raw.jsonl", issues)["complete"])
            self.assertTrue(any("traced SDPA shapes/strides" in problem for problem in issues))

    def test_operator_manifest_preserves_grouped_query_heads_and_strides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "operator"
            root.mkdir()
            trace = root / "aten_trace_B8.json"
            arguments = [
                {"shape": [8, 8, 1, 256], "stride": [2048, 256, 256, 1]},
                {"shape": [8, 1, 970, 256], "stride": [248320, 248320, 256, 1]},
                {"shape": [8, 1, 970, 256], "stride": [248320, 248320, 256, 1]},
            ]
            trace.write_text(json.dumps({"operations": [{
                "aten_operation": "aten::scaled_dot_product_attention",
                "arguments": arguments,
            }]}))
            (root / "configurations.json").write_text(json.dumps({
                "configurations": [{
                    "id": "gqa", "source_operation": "traced grouped-query attention",
                    "operation": "sdpa", "m_per_request": 1,
                    "query_heads": 8, "key_heads": 1, "value_heads": 1,
                    "enable_gqa": True, "query_length": 1, "key_length": 970,
                    "head_dim": 256, "value_dim": 256, "scale": 0.0625,
                    "layout": "qkv_source_strided", "source_trace_path": str(trace),
                    "source_q_stride": arguments[0]["stride"],
                    "source_k_stride": arguments[1]["stride"],
                    "source_v_stride": arguments[2]["stride"],
                }],
            }))
            (root / "raw.jsonl").write_text("")
            issues = []
            self.assertFalse(operator_audit(root / "raw.jsonl", issues)["complete"])
            self.assertFalse(any("shape/stride is not in" in problem for problem in issues))
            self.assertFalse(any("traced SDPA shapes/strides" in problem for problem in issues))
            self.assertTrue(any("100 unique seeds" in problem for problem in issues))

    def test_operator_manifest_accepts_real_convolution_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "operator"
            root.mkdir()
            (root / "configurations.json").write_text(json.dumps({
                "configurations": [
                    {
                        "id": "causal_conv", "source_operation": "traced depthwise causal conv",
                        "operation": "conv1d", "m_per_request": 1,
                        "input_per_unit": [6144, 650], "weight_shape": [6144, 1, 4],
                        "stride": [1], "padding": [3], "dilation": [1],
                        "groups": 6144, "layout": "contiguous",
                    },
                    {
                        "id": "patch_conv", "source_operation": "traced visual patch conv",
                        "operation": "conv3d", "m_per_request": 512,
                        "input_per_unit": [3, 2, 16, 16],
                        "weight_shape": [1024, 3, 2, 16, 16],
                        "stride": [2, 16, 16], "padding": [0, 0, 0],
                        "dilation": [1, 1, 1], "groups": 1, "layout": "contiguous",
                    },
                    {
                        "id": "temporal_upsampler",
                        "source_operation": "traced temporal ConvTranspose3d",
                        "operation": "conv_transpose3d", "m_per_request": 1,
                        "input_per_unit": [1024, 4, 16, 16],
                        "weight_shape": [1024, 1024, 4, 1, 1],
                        "stride": [4, 1, 1], "padding": [0, 0, 0],
                        "output_padding": [0, 0, 0], "dilation": [1, 1, 1],
                        "groups": 1, "layout": "contiguous",
                    },
                    {
                        "id": "spatial_upsampler",
                        "source_operation": "traced spatial ConvTranspose2d",
                        "operation": "conv_transpose2d", "m_per_request": 1,
                        "input_per_unit": [256, 24, 24],
                        "weight_shape": [256, 256, 4, 4],
                        "stride": [4, 4], "padding": [0, 0],
                        "output_padding": [0, 0], "dilation": [1, 1],
                        "groups": 1, "layout": "contiguous",
                    },
                ],
            }))
            (root / "raw.jsonl").write_text("")
            issues = []
            self.assertFalse(operator_audit(root / "raw.jsonl", issues)["complete"])
            self.assertFalse(any("incomplete source-shape" in problem for problem in issues))
            self.assertTrue(any("100 unique seeds" in problem for problem in issues))

    def test_operator_manifest_verifies_sharded_checkpoint_weight(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operator = root / "operator"
            operator.mkdir()
            replay = root / "replay_manifest.json"
            replay.write_text('{"observations": []}\n')
            replay_hash = hashlib.sha256(replay.read_bytes()).hexdigest()
            first = root / "model-00001-of-00002.safetensors"
            second = root / "model-00002-of-00002.safetensors"
            save_file({"unrelated": torch.zeros(1)}, first)
            save_file({"layer.weight": torch.zeros(3, 2)}, second)
            for batch_size in (1, 8):
                (operator / f"aten_trace_B{batch_size}.json").write_text(json.dumps({
                    "batch_size": batch_size,
                    "request_ids": [f"request-{index}" for index in range(batch_size)],
                    "replay_manifest_sha256": replay_hash,
                    "operations": [],
                }))
            (operator / "configurations.json").write_text(json.dumps({
                "checkpoint_weight_shards": [str(first), str(second)],
                "checkpoint_weight_shards_sha256": [
                    hashlib.sha256(first.read_bytes()).hexdigest(),
                    hashlib.sha256(second.read_bytes()).hexdigest(),
                ],
                "configurations": [{
                    "id": "linear", "source_operation": "checkpoint-backed linear",
                    "source_weight_key": "layer.weight", "weight_shape": [3, 2],
                    "operation": "mm", "m_per_request": 1, "k": 2, "n": 3,
                    "layout": "right_transposed",
                }],
            }))
            (operator / "raw.jsonl").write_text("")
            issues = []
            self.assertFalse(operator_audit(operator / "raw.jsonl", issues)["complete"])
            self.assertFalse(any("checkpoint shard" in problem for problem in issues))
            self.assertTrue(any("100 unique seeds" in problem for problem in issues))

    def test_operator_manifest_accepts_request_preserving_mean_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operator = root / "operator"
            operator.mkdir()
            replay = root / "replay_manifest.json"
            replay.write_text('{"observations": []}\n')
            replay_hash = hashlib.sha256(replay.read_bytes()).hexdigest()
            for batch_size in (1, 8):
                (operator / f"aten_trace_B{batch_size}.json").write_text(json.dumps({
                    "batch_size": batch_size,
                    "request_ids": [f"request-{index}" for index in range(batch_size)],
                    "replay_manifest_sha256": replay_hash,
                    "operations": [],
                }))
            (operator / "configurations.json").write_text(json.dumps({
                "configurations": [{
                    "id": "rms_mean", "source_operation": "RMSNorm mean reduction",
                    "operation": "mean", "m_per_request": 10,
                    "input_per_unit": [1024], "dim": -1, "keepdim": True,
                    "layout": "contiguous",
                }],
            }))
            (operator / "raw.jsonl").write_text("")
            issues = []
            self.assertFalse(operator_audit(operator / "raw.jsonl", issues)["complete"])
            self.assertFalse(any("incomplete source-shape" in problem for problem in issues))
            self.assertTrue(any("100 unique seeds" in problem for problem in issues))

    def test_single_mode_large_batch_timing_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy_B16.jsonl"
            rows = [
                {"replay_index": index, "split": "diagnostic", "episode_id": index,
                 "task_index": index, "request_ids": [f"request-{index}"]}
                for index in range(16)
            ]
            manifest_path = path.parent / "replay_manifest.json"
            manifest_path.write_text(json.dumps({"observations": rows}))
            with path.open("w") as stream:
                stream.write(json.dumps({
                    "record_type": "environment", "campaign_protocol": "full_v1",
                    "sessions": 5, "warmup": 20, "repetitions": 200,
                    "batch_sizes": [16], "implementations": ["native"],
                    "input_selection": {
                        "replay_manifest_path": str(manifest_path),
                        "replay_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        "replay_indices": list(range(16)),
                        "episode_ids": list(range(16)), "task_indices": list(range(16)),
                        "request_ids": [f"request-{index}" for index in range(16)],
                    },
                }) + "\n")
                for session in range(5):
                    for index in range(20):
                        stream.write(json.dumps({
                            "record_type": "warmup", "implementation": "native",
                            "batch_size": 16, "session": session,
                            "attempt_id": f"native-{session}", "warmup_index": index,
                        }) + "\n")
                    for repetition in range(200):
                        stream.write(json.dumps({
                            "record_type": "timing", "implementation": "native",
                            "batch_size": 16, "session": session,
                            "repetition": repetition, "latency_seconds": 0.1,
                            "requests_per_second": 160.0,
                            "peak_incremental_bytes": 0,
                        }) + "\n")
            issues = []
            result = timing_audit(
                path, "policy", issues,
                required_policy_batches=(16,), expected_policy_modes=("native",),
            )
            self.assertTrue(result["complete"])
            self.assertFalse(issues)


if __name__ == "__main__":
    unittest.main()

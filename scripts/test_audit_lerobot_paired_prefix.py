"""Focused checks for live paired-rollout integrity without full promotion."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

import torch

from audit_lerobot_paired_prefix import audit
from run_lerobot_paired_libero import (
    attach_policy_observation_recorder,
    attach_serial_singleton_execution,
    set_explicit_init_state_ids,
)


class PairedPrefixTests(unittest.TestCase):
    def test_serial_singleton_keeps_four_independent_action_queues(self) -> None:
        class FakePolicy:
            def __init__(self):
                self.forward_calls = []
                self.reset()

            def reset(self):
                self._action_queue = deque(maxlen=2)

            def select_action(self, batch):
                if not self._action_queue:
                    value = batch["observation.state"][:, :1]
                    self.forward_calls.append(
                        (self._paired_active_request_index, float(value.item()))
                    )
                    self._action_queue.extend((value, value + 100.0))
                return self._action_queue.popleft()

        policy = FakePolicy()
        attach_serial_singleton_execution(policy, 4)
        attach_policy_observation_recorder(policy, 4)
        policy._paired_first_sensor_hashes = []
        policy._paired_state_trajectory = []
        batch = {
            "observation.state": torch.tensor([[0.0], [10.0], [20.0], [30.0]]),
            "observation.images.image": torch.zeros(4, 3, 2, 2),
            "task": ["a", "b", "c", "d"],
        }
        self.assertEqual(policy.select_action(batch).flatten().tolist(), [0.0, 10.0, 20.0, 30.0])
        self.assertEqual(policy.select_action(batch).flatten().tolist(), [100.0, 110.0, 120.0, 130.0])
        self.assertEqual(policy.forward_calls, [(0, 0.0), (1, 10.0), (2, 20.0), (3, 30.0)])
        self.assertEqual(len(policy._paired_state_trajectory), 2)
        self.assertEqual(len(policy._paired_first_sensor_hashes), 4)

    def test_protocol_v2_singleton_requires_fixed_b4_and_raw_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "rollouts").mkdir()
            manifest = {
                "schema_version": 2, "paired_protocol_version": 2,
                "model": "smolvla", "condition": "native_singleton",
                "batch_size": 1, "policy_batch_size": 1,
                "environment_batch_size": 4, "execution_mode": "serial_singleton",
                "smoke_test": False,
                "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "relative",
                "episode_length": 520,
                "explicit_init_state_by_initialization_id": True,
            }
            (root / "condition_manifest.json").write_text(json.dumps(manifest))
            path = root / "rollouts/task_0_seed_1000.json"
            path.with_suffix(".pt").write_bytes(b"cached official rollout")
            path.write_text(json.dumps({
                "task_group": "libero_10", "task_id": 0,
                "seeds": [1000, 1001, 1002, 1003],
                "explicit_init_state_ids": [0, 1, 2, 3],
                "episodes": [{
                    "task": 0, "task_group": "libero_10", "initialization_id": index,
                    "reset_seed": 1000 + index, "success": False,
                    "initial_raw_observation_sha256": f"{index + 1:064x}",
                    "initial_observation_sha256": f"{index + 11:064x}",
                    "first_noise_sha256": f"{index + 21:064x}",
                    "trajectory": [[0.0] * 8], "action_sequence": [[0.0] * 7],
                    "rewards": [0.0], "done": [False],
                } for index in range(4)],
            }))
            report = audit(root)
            self.assertTrue(report["causal_pairing_eligible"])
            self.assertEqual(report["policy_batch_size"], 1)
            self.assertEqual(report["environment_batch_size"], 4)
            manifest["environment_batch_size"] = 1
            (root / "condition_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "isolate policy batch"):
                audit(root)

    def test_smolvla_accepts_official_eight_dimensional_environment_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "rollouts").mkdir()
            (root / "condition_manifest.json").write_text(json.dumps({
                "model": "smolvla", "condition": "native_singleton",
                "batch_size": 1, "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "relative",
                "episode_length": 520,
                "explicit_init_state_by_initialization_id": True,
            }))
            path = root / "rollouts/task_0_seed_1000.json"
            path.with_suffix(".pt").write_bytes(b"cached official rollout")
            path.write_text(json.dumps({
                "task_group": "libero_10", "task_id": 0, "seeds": [1000],
                "explicit_init_state_ids": [0],
                "episodes": [{
                    "task": 0, "task_group": "libero_10",
                    "initialization_id": 0, "reset_seed": 1000,
                    "success": False, "initial_observation_sha256": "a" * 64,
                    "first_noise_sha256": "b" * 64,
                    "trajectory": [[0.0] * 8], "action_sequence": [[0.0] * 7],
                    "rewards": [0.0], "done": [False],
                }],
            }))
            report = audit(root)
            self.assertEqual(report["environment_state_dimension"], 8)
            value = json.loads(path.read_text())
            value["episodes"][0]["trajectory"] = [[0.0] * 20]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "incomplete or nonfinite"):
                audit(root)

    def test_explicit_init_state_ids_override_vector_reset_history(self) -> None:
        class FakeVectorEnv:
            num_envs = 4

            def __init__(self):
                self.values = [91, 92, 93, 94]

            def set_attr(self, name, values):
                self.assert_name = name
                self.values = list(values)

            def get_attr(self, name):
                self.assert_name = name
                return tuple(self.values)

        env = FakeVectorEnv()
        self.assertEqual(
            set_explicit_init_state_ids(env, [1008, 1009, 1010, 1011], 1000),
            [8, 9, 10, 11],
        )
        self.assertEqual(env.values, [8, 9, 10, 11])

    def test_explicit_init_state_ids_support_lerobot_lazy_vector_env(self) -> None:
        class InnerVectorEnv:
            def __init__(self):
                self.values = [91, 92, 93, 94]

            def set_attr(self, name, values):
                self.assert_name = name
                self.values = list(values)

            def get_attr(self, name):
                self.assert_name = name
                return tuple(self.values)

        class FakeLazyVectorEnv:
            num_envs = 4

            def __init__(self):
                self._env = None
                self.ensure_calls = 0

            def _ensure(self):
                self.ensure_calls += 1
                if self._env is None:
                    self._env = InnerVectorEnv()

            def get_attr(self, name):
                self._ensure()
                return self._env.get_attr(name)

        env = FakeLazyVectorEnv()
        self.assertEqual(
            set_explicit_init_state_ids(env, [1008, 1009, 1010, 1011], 1000),
            [8, 9, 10, 11],
        )
        self.assertEqual(env._env.values, [8, 9, 10, 11])
        self.assertEqual(env.ensure_calls, 2)

    def test_noise_reuse_across_tasks_is_allowed_but_within_task_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rollouts").mkdir()
            (root / "condition_manifest.json").write_text(json.dumps({
                "model": "xvla", "condition": "native_singleton",
                "batch_size": 1, "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "absolute",
                "episode_length": 800,
                "explicit_init_state_by_initialization_id": True,
            }))
            for task, count in ((0, 50), (1, 1)):
                for initialization in range(count):
                    seed = 1000 + initialization
                    path = root / "rollouts" / f"task_{task}_seed_{seed}.json"
                    path.with_suffix(".pt").write_bytes(b"cached official rollout")
                    path.write_text(json.dumps({
                        "task_group": "libero_10", "task_id": task, "seeds": [seed],
                        "explicit_init_state_ids": [initialization],
                        "episodes": [{
                            "task": task, "task_group": "libero_10",
                            "initialization_id": initialization, "reset_seed": seed,
                            "success": False, "initial_observation_sha256": "a" * 64,
                            "first_noise_sha256": f"{initialization:064x}",
                            "trajectory": [[0.0] * 20],
                            "action_sequence": [[0.0] * 7],
                            "rewards": [0.0], "done": [False],
                        }],
                    }))
            report = audit(root)
            self.assertEqual(report["planned_episodes_cached"], 51)
            self.assertFalse(report["all_ten_tasks_have_50_resets"])
            duplicate_path = root / "rollouts" / "task_0_seed_1001.json"
            duplicate = json.loads(duplicate_path.read_text())
            duplicate["episodes"][0]["first_noise_sha256"] = "0" * 64
            duplicate_path.write_text(json.dumps(duplicate))
            with self.assertRaisesRegex(ValueError, "within one task"):
                audit(root)

    def test_xvla_rejects_generic_libero_control_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "condition_manifest.json").write_text(json.dumps({
                "model": "xvla", "condition": "native_singleton",
                "batch_size": 1, "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "relative",
                "episode_length": 520,
                "explicit_init_state_by_initialization_id": True,
            }))
            with self.assertRaisesRegex(ValueError, "absolute/800"):
                audit(root)

    def test_pi0fast_rejects_unpinned_libero_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "condition_manifest.json").write_text(json.dumps({
                "model": "pi_fast", "condition": "native_singleton",
                "batch_size": 1, "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "absolute",
                "episode_length": 800,
                "explicit_init_state_by_initialization_id": True,
            }))
            with self.assertRaisesRegex(ValueError, "PI0-FAST"):
                audit(root)

    def test_smolvla_rejects_unpinned_libero_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "condition_manifest.json").write_text(json.dumps({
                "model": "smolvla", "condition": "native_singleton",
                "batch_size": 1, "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "relative",
                "episode_length": 800,
                "explicit_init_state_by_initialization_id": True,
            }))
            with self.assertRaisesRegex(ValueError, "SmolVLA"):
                audit(root)

    def test_univla_accepts_only_its_pinned_deterministic_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rollouts").mkdir()
            contract = "d" * 64
            (root / "condition_manifest.json").write_text(json.dumps({
                "schema_version": 2, "paired_protocol_version": 2,
                "model": "univla_baai", "condition": "native_singleton",
                "batch_size": 1, "policy_batch_size": 1,
                "environment_batch_size": 4, "execution_mode": "serial_singleton",
                "smoke_test": False, "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "relative",
                "episode_length": 520, "explicit_init_state_by_initialization_id": True,
                "request_local_noise": False, "deterministic_equivalent": True,
                "deterministic_contract_sha256": contract,
            }))
            path = root / "rollouts/task_0_seeds_1000_1003.json"
            path.with_suffix(".pt").write_bytes(b"cached official rollout")
            value = {
                "task_group": "libero_10", "task_id": 0,
                "seeds": [1000, 1001, 1002, 1003],
                "explicit_init_state_ids": [0, 1, 2, 3],
                "episodes": [{
                    "task": 0, "task_group": "libero_10", "initialization_id": index,
                    "reset_seed": 1000 + index, "success": False,
                    "initial_raw_observation_sha256": f"{index + 1:064x}",
                    "initial_observation_sha256": f"{index + 11:064x}",
                    "first_noise_sha256": contract,
                    "trajectory": [[0.0] * 8], "action_sequence": [[0.0] * 7],
                    "rewards": [0.0], "done": [False],
                } for index in range(4)],
            }
            path.write_text(json.dumps(value))
            self.assertTrue(audit(root)["integrity_passed"])
            value["episodes"][2]["first_noise_sha256"] = "e" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "UniVLA prefix"):
                audit(root)

    def test_molmoact2_requires_unique_request_local_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rollouts").mkdir()
            (root / "condition_manifest.json").write_text(json.dumps({
                "schema_version": 2, "paired_protocol_version": 2,
                "model": "molmoact2", "condition": "native_dynamic",
                "batch_size": 4, "policy_batch_size": 4,
                "environment_batch_size": 4, "execution_mode": "dynamic_batch",
                "smoke_test": False, "episodes_per_task": 50, "seed": 1000,
                "task_suite": "libero_10", "control_mode": "relative",
                "episode_length": 520, "explicit_init_state_by_initialization_id": True,
                "request_local_noise": True,
            }))
            path = root / "rollouts/task_0_seeds_1000_1003.json"
            path.with_suffix(".pt").write_bytes(b"cached official rollout")
            value = {
                "task_group": "libero_10", "task_id": 0,
                "seeds": [1000, 1001, 1002, 1003],
                "explicit_init_state_ids": [0, 1, 2, 3],
                "episodes": [{
                    "task": 0, "task_group": "libero_10", "initialization_id": index,
                    "reset_seed": 1000 + index, "success": False,
                    "initial_raw_observation_sha256": f"{index + 1:064x}",
                    "initial_observation_sha256": f"{index + 11:064x}",
                    "first_noise_sha256": f"{index + 21:064x}",
                    "trajectory": [[0.0] * 8], "action_sequence": [[0.0] * 7],
                    "rewards": [0.0], "done": [False],
                } for index in range(4)],
            }
            path.write_text(json.dumps(value))
            self.assertTrue(audit(root)["integrity_passed"])
            value["episodes"][1]["first_noise_sha256"] = value["episodes"][0]["first_noise_sha256"]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "within one task"):
                audit(root)


if __name__ == "__main__":
    unittest.main()

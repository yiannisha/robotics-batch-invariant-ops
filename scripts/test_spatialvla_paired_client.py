"""Focused schedule checks for the SpatialVLA SimplerEnv paired client."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, "/workspace/models/internvla-a-series")
sys.path.insert(0, "/workspace/models/simplerenv-openvla")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_spatialvla_paired_client import (
    coke_config,
    drawer_config,
    physical_hash,
    state_record,
    task_configuration,
)


MANIFEST = Path(
    "results/cross_model/spatialvla/full/closed_loop/paired/evaluator_task_manifest.json"
)


def canonical(value):
    return json.dumps(value, sort_keys=True, default=lambda item: item.tolist())


class SpatialVLAPairedClientTests(unittest.TestCase):
    def test_manifest_cardinality_and_source_limit(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(len(manifest["tasks"]), 12)
        self.assertEqual(sum(task["initializations"] for task in manifest["tasks"]), 496)
        self.assertEqual([task["initializations"] for task in manifest["tasks"][:4]], [24] * 4)
        self.assertEqual(manifest["logical_episodes_all_conditions"], 1984)

    def test_drawer_first_fifty_are_unique(self) -> None:
        values = [canonical(drawer_config(index)) for index in range(50)]
        self.assertEqual(len(set(values)), 50)

    def test_coke_first_fifty_are_unique(self) -> None:
        values = [canonical(coke_config(index)) for index in range(50)]
        self.assertEqual(len(set(values)), 50)

    def test_every_manifest_initialization_resolves(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        for task in manifest["tasks"]:
            values = [canonical(task_configuration(task, index))
                      for index in range(task["initializations"])]
            self.assertEqual(len(values), len(set(values)))

    def test_state_evidence_covers_both_official_robot_schemas(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        common_agent = {"qpos": np.zeros(2), "qvel": np.zeros(2)}
        observations = (
            (
                "tcp_pose",
                {"agent": common_agent, "extra": {"tcp_pose": np.zeros(7)}},
            ),
            (
                "base_pose",
                {
                    "agent": {**common_agent, "base_pose": np.zeros(7)},
                    "extra": {},
                },
            ),
        )
        for pose_key, observation in observations:
            with self.subTest(pose_key=pose_key):
                self.assertEqual(
                    set(state_record(observation)),
                    {"qpos", "qvel", pose_key},
                )
                self.assertEqual(len(physical_hash(observation, image)), 64)


if __name__ == "__main__":
    unittest.main()

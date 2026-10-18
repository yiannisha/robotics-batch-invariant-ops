"""Focused checks for the SpatialVLA paired policy server."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, "/workspace/models/internvla-a-series")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_spatialvla_paired_server import RequestState


def official_action_ensembler():
    path = Path(
        "/workspace/models/simplerenv-openvla/"
        "simpler_env/utils/action/action_ensemble.py"
    )
    spec = importlib.util.spec_from_file_location("official_action_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ActionEnsembler


class SpatialVLAPairedServerTests(unittest.TestCase):
    def test_request_local_ensemble_is_exact_official_implementation(self) -> None:
        reference = official_action_ensembler()(4, -0.8)
        implementation = RequestState(4, -0.8)
        generator = np.random.default_rng(20261005)
        for _ in range(8):
            chunk = generator.standard_normal((4, 7))
            self.assertTrue(np.array_equal(
                implementation.ensemble(chunk), reference.ensemble_action(chunk),
            ))

    def test_request_histories_do_not_cross_contaminate(self) -> None:
        generator = np.random.default_rng(20261005)
        left = RequestState(4, -0.8)
        right = RequestState(4, -0.8)
        left_reference = official_action_ensembler()(4, -0.8)
        right_reference = official_action_ensembler()(4, -0.8)
        for _ in range(8):
            left_chunk = generator.standard_normal((4, 7))
            right_chunk = generator.standard_normal((4, 7))
            self.assertTrue(np.array_equal(
                left.ensemble(left_chunk), left_reference.ensemble_action(left_chunk),
            ))
            self.assertTrue(np.array_equal(
                right.ensemble(right_chunk), right_reference.ensemble_action(right_chunk),
            ))

    def test_widowx_gripper_uses_official_binary_mapping(self) -> None:
        state = RequestState(4, -0.8, policy_setup="widowx_bridge")
        opened = np.zeros((4, 7)); opened[:, 6] = 0.75
        closed = np.zeros((4, 7)); closed[:, 6] = 0.25
        self.assertEqual(state.process(opened)[1][-1], 1.0)
        self.assertEqual(state.process(closed)[1][-1], -1.0)


if __name__ == "__main__":
    unittest.main()

from unittest import TestCase

import numpy as np

from run_cogact_paired_server import RequestState


class _FirstActionEnsembler:
    def __init__(self, horizon: int, alpha: float) -> None:
        self.horizon = horizon
        self.alpha = alpha

    def ensemble_action(self, actions: np.ndarray) -> np.ndarray:
        return actions[0]


class CogACTPairedWrapperTests(TestCase):
    def test_widowx_postprocessing_matches_released_contract(self) -> None:
        state = RequestState("widowx_bridge", _FirstActionEnsembler)
        chunk = np.zeros((16, 7), dtype=np.float64)
        chunk[0, :3] = [0.1, -0.2, 0.3]
        chunk[0, 6] = 0.75
        selected, action = state.process(chunk)
        self.assertEqual(state.ensemble.horizon, 7)
        self.assertEqual(state.ensemble.alpha, 0.1)
        self.assertTrue(np.array_equal(selected, chunk[0]))
        np.testing.assert_allclose(action, [0.1, -0.2, 0.3, 0, 0, 0, 1])

    def test_google_first_gripper_action_is_zero(self) -> None:
        state = RequestState("google_robot", _FirstActionEnsembler)
        chunk = np.zeros((16, 7), dtype=np.float64)
        chunk[0, 6] = 0.9
        _selected, action = state.process(chunk)
        self.assertEqual(state.ensemble.horizon, 2)
        self.assertEqual(action[-1], 0)

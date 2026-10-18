from unittest import TestCase

import numpy as np

from run_cogact_paired_client import physical_hash, request_noise, state_record


class CogACTPairedClientTests(TestCase):
    def test_request_noise_is_request_local_and_reproducible(self) -> None:
        first = request_noise(20261005, 2, 7, 11)
        repeated = request_noise(20261005, 2, 7, 11)
        companion = request_noise(20261005, 2, 8, 11)
        self.assertEqual(first.shape, (16, 7))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.array_equal(first, repeated))
        self.assertFalse(np.array_equal(first, companion))

    def test_state_evidence_covers_both_official_robot_schemas(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        common_agent = {"qpos": np.zeros(2), "qvel": np.zeros(2)}
        observations = (
            ("tcp_pose", {"agent": common_agent, "extra": {"tcp_pose": np.zeros(7)}}),
            (
                "base_pose",
                {"agent": {**common_agent, "base_pose": np.zeros(7)}, "extra": {}},
            ),
        )
        for pose_key, observation in observations:
            with self.subTest(pose_key=pose_key):
                self.assertEqual(
                    set(state_record(observation)), {"qpos", "qvel", pose_key}
                )
                self.assertEqual(len(physical_hash(observation, image)), 64)

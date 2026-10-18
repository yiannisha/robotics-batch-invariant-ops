from __future__ import annotations

import fcntl
import tempfile
import unittest
from pathlib import Path

import continue_cogact_paired_when_resources as cogact
import continue_internvla_optimized_when_resources as internvla
import continue_uva_joint_when_resources as uva


class ResourcePriorityLockTest(unittest.TestCase):
    def assert_lock_detector(self, detector) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.lock"
            self.assertFalse(detector(path))
            with path.open("a+") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                self.assertTrue(detector(path))
                fcntl.flock(held, fcntl.LOCK_UN)
            self.assertFalse(detector(path))

    def test_uva_detects_held_campaign_lock(self) -> None:
        self.assert_lock_detector(uva.lock_active)

    def test_cogact_detects_held_campaign_lock(self) -> None:
        self.assert_lock_detector(cogact.lock_active)

    def test_internvla_detects_held_higher_priority_lock(self) -> None:
        self.assert_lock_detector(internvla.lock_active)

    def test_transitive_launch_order_prevents_post_wsa_race(self) -> None:
        self.assertEqual(
            [name for name, _ in internvla.HIGHER_PRIORITY_LOCKS],
            ["WSA", "PI0-FAST", "SmolVLA"],
        )
        self.assertEqual(
            [name for name, _ in uva.HIGHER_PRIORITY_LOCKS],
            ["WSA", "InternVLA optimized"],
        )
        self.assertEqual(
            [name for name, _ in cogact.HIGHER_PRIORITY_LOCKS],
            ["WSA", "InternVLA optimized", "UVA joint"],
        )


if __name__ == "__main__":
    unittest.main()

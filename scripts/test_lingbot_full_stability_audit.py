"""Independent magnitude arithmetic controls; no campaign acceptance evidence."""
import hashlib
import unittest

import numpy as np

from scripts.audit_lingbot_full_stability import arm_distances, fixed_errors


def call(values):
    values = np.asarray(values, dtype=np.float32)
    return dict(normalized_actions=values.tolist(), normalized_sha256=hashlib.sha256(values.tobytes()).hexdigest())


class IndependentMagnitudeChecks(unittest.TestCase):
    def poses(self):
        x = np.zeros((3, 2, 7), dtype=np.float64)
        x[:, :, 3] = 1
        return x

    def test_separate_arm_translation_and_known_rotation(self):
        x, y = self.poses(), self.poses()
        y[:, 0, 0] = .01
        y[:, 1, 3:5] = np.sqrt(.5)
        positions, rotations = arm_distances(x, y)
        np.testing.assert_allclose(positions[:, 0], .01)
        np.testing.assert_array_equal(positions[:, 1], 0)
        np.testing.assert_array_equal(rotations[:, 0], 0)
        np.testing.assert_allclose(rotations[:, 1], np.pi / 2)

    def test_quaternion_sign_equivalence_and_tiny_rotation(self):
        x, y = self.poses(), self.poses()
        y[:, 0, 3:] *= -1
        angle = 1e-10
        y[:, 1, 3] = np.cos(angle / 2)
        y[:, 1, 4] = np.sin(angle / 2)
        _, rotations = arm_distances(x, y)
        np.testing.assert_array_equal(rotations[:, 0], 0)
        np.testing.assert_allclose(rotations[:, 1], angle, atol=1e-20)

    def test_zero_quaternion_rejected(self):
        x, y = self.poses(), self.poses()
        y[:, 0, 3:] = 0
        with self.assertRaisesRegex(ValueError, 'Zero quaternion'):
            arm_distances(x, y)

    def test_known_normalized_error(self):
        x, y = np.zeros((50, 55), dtype=np.float32), np.full((50, 55), .125, dtype=np.float32)
        r = fixed_errors(call(x), call(y))
        self.assertFalse(r['exact'])
        self.assertEqual(r['different_elements'], 2750)
        self.assertEqual(r['rms_error'], .125)
        self.assertEqual(r['squared_error_sum'], 2750 * .125 ** 2)

    def test_signed_zero_has_zero_error_and_nonexact_bits(self):
        x, y = np.zeros((50, 55), dtype=np.float32), np.zeros((50, 55), dtype=np.float32)
        y[0, 0] = -0.
        r = fixed_errors(call(x), call(y))
        self.assertFalse(r['exact'])
        self.assertEqual(r['different_elements'], 1)
        self.assertEqual(r['rms_error'], 0)
        self.assertEqual(r['max_ulp'], 1)

    def test_normalized_hash_corruption_rejected(self):
        x = call(np.zeros((50, 55)))
        y = dict(x, normalized_sha256='f' * 64)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            fixed_errors(x, y)

    def test_nonfinite_normalized_output_rejected(self):
        x = np.zeros((50, 55), dtype=np.float32)
        y = x.copy()
        y[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid normalized'):
            fixed_errors(call(x), call(y))


if __name__ == '__main__':
    unittest.main()

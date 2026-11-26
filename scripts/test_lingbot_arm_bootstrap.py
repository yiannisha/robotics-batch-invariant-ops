import unittest
import numpy as np
from scripts.bootstrap_lingbot_arm_stability import values, estimate, resample


class ArmBootstrap(unittest.TestCase):
    def test_known_whole_start_pair_draws(self):
        data = np.array([[1., 2., .1, .2, 0.], [3., 6., .3, .6, 1.]])
        indices = np.array([[0, 0], [1, 1], [0, 1]], dtype=np.int32)
        result = resample(data, indices)
        np.testing.assert_allclose(result[:, 0], [1., 3., 2.])
        np.testing.assert_allclose(result[:, 1], [2., 6., 4.])
        np.testing.assert_allclose(result[:, 8], [0., 1., .5])
        np.testing.assert_allclose(result[2], estimate(data))

    def test_mean_and_median_are_distinct(self):
        data = np.array([[0., 0., 0., 0., 0.], [1., 2., 0., 0., 1.], [8., 4., 0., 0., 1.]])
        point = estimate(data)
        self.assertEqual(point[0], 3.)
        self.assertEqual(point[4], 1.)

    def test_zero_invariant_stays_exact_under_all_draws(self):
        data = np.zeros((3, 5))
        np.testing.assert_array_equal(resample(data, np.array([[0, 1, 2], [2, 2, 2]])), 0.)

    def test_arm_units_and_nonfinite_rejection(self):
        comparison = dict(arms={'left': dict(position_max_m=.001, rotation_max_rad=.1),
                                'right': dict(position_max_m=.002, rotation_max_rad=.2)}, action_disagreement_controls=1)
        start = dict(comparisons={'native': comparison})
        np.testing.assert_allclose(values([start], 'native'), [[1., 2., .1, .2, 1.]])
        comparison['arms']['right']['position_max_m'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid arm'):
            values([start], 'native')


if __name__ == '__main__':
    unittest.main()

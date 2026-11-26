import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('task_stability', Path(__file__).with_name('summarize_lingbot_task_stability.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TaskRows(unittest.TestCase):
    def starts(self):
        result = []
        for i in range(2):
            values = dict(batched_policy_calls=1, action_disagreement_controls=1,
                          first_action_divergence_control_index=50,
                          fixed_identical_input_nonsingleton_pairs=[dict(elements=2, squared_error_sum=.125, max_absolute_error=.25)],
                          arms={'left': dict(position_max_m=(i + 1) * .001, rotation_max_rad=.1),
                                'right': dict(position_max_m=(i + 1) * .002, rotation_max_rad=.2)})
            result.append(dict(comparisons={'native': copy.deepcopy(values), 'invariant': copy.deepcopy(values)}))
        return result

    def test_arm_units_and_pooled_fixed_input_error(self):
        rows = module.task_rows('task', self.starts())
        left, right = rows[:2]
        self.assertEqual(left['median_peak_position_drift_mm'], 1.5)
        self.assertEqual(right['median_peak_position_drift_mm'], 3.)
        self.assertEqual(left['pooled_fixed_input_mse'], .0625)
        self.assertEqual(left['pooled_fixed_input_rms'], .25)

    def test_no_numerical_exposure_is_null(self):
        starts = self.starts()
        for start in starts:
            for value in start['comparisons'].values():
                value['fixed_identical_input_nonsingleton_pairs'] = []
                value['batched_policy_calls'] = 0
        for row in module.task_rows('task', starts):
            self.assertIsNone(row['pooled_fixed_input_mse'])
            self.assertIsNone(row['pooled_fixed_input_rms'])
            self.assertIsNone(row['maximum_fixed_input_absolute_error'])
            self.assertEqual(row['starts_exposed_to_nonsingleton_calls'], 0)

    def test_nonfinite_drift_rejected(self):
        starts = self.starts()
        starts[0]['comparisons']['native']['arms']['right']['position_max_m'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid arm'):
            module.task_rows('task', starts)


if __name__ == '__main__':
    unittest.main()

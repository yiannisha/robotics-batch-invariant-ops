import copy
import csv
import importlib.util
from pathlib import Path
import unittest
import tempfile

spec = importlib.util.spec_from_file_location('task_audit', Path(__file__).with_name('audit_lingbot_task_stability.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TaskAudit(unittest.TestCase):
    def values(self):
        return [dict(comparisons={'native': dict(
            batched_policy_calls=1, action_disagreement_controls=1,
            first_action_divergence_control_index=50 + i,
            fixed_identical_input_nonsingleton_pairs=[dict(elements=2, squared_error_sum=.125, max_absolute_error=.25)],
            arms={'left': dict(position_max_m=(i + 1) * .001, rotation_max_rad=.1)})}) for i in range(2)]

    def test_hand_calculated_statistics(self):
        row = audit.expected_row('task', 'native', 'left', self.values())
        self.assertEqual(row['median_peak_position_drift_mm'], 1.5)
        self.assertAlmostEqual(row['p95_peak_position_drift_mm'], 1.95)
        self.assertEqual(row['conditional_median_first_divergence_control'], 50.5)
        self.assertEqual(row['pooled_fixed_input_rms'], .25)
        audit.verify_row(row, row)

    def test_modified_unit_rejected(self):
        expected = audit.expected_row('task', 'native', 'left', self.values())
        row = dict(expected, median_peak_position_drift_mm=.0015)
        with self.assertRaisesRegex(ValueError, 'median_peak'):
            audit.verify_row(row, expected)

    def test_null_is_not_zero(self):
        values = self.values()
        for value in values:
            value['comparisons']['native']['fixed_identical_input_nonsingleton_pairs'] = []
        expected = audit.expected_row('task', 'native', 'left', values)
        self.assertIsNone(expected['pooled_fixed_input_mse'])
        with self.assertRaisesRegex(ValueError, 'pooled_fixed_input_mse'):
            audit.verify_row(dict(expected, pooled_fixed_input_mse=0.), expected)

    def test_nonfinite_report_rejected(self):
        expected = audit.expected_row('task', 'native', 'left', self.values())
        with self.assertRaisesRegex(ValueError, 'worst_position'):
            audit.verify_row(dict(expected, worst_position_drift_mm=float('nan')), expected)

    def test_csv_export_and_corruption(self):
        row = audit.expected_row('task', 'native', 'left', self.values())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'table.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            audit.verify_csv(path, [row])
            path.write_text(path.read_text().replace('1.5', '999.5'))
            with self.assertRaisesRegex(ValueError, 'CSV values differ'):
                audit.verify_csv(path, [row])


if __name__ == '__main__':
    unittest.main()

"""Synthetic-only serving figure controls; no model queue evidence is created."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_serving_tradeoff_table import audit, sha
from scripts.plot_serving_stability_tradeoff import MODES, LOADS, RATIOS, summarize


def fixture(directory):
    raw = directory / 'synthetic_queue.jsonl'
    with raw.open('w') as stream:
        for mode_index, mode in enumerate(MODES):
            for load_index, (load, ratio) in enumerate(zip(LOADS, RATIOS)):
                for i in range(250):
                    arrival = i / (8 * ratio)
                    latency = (.2 + .07 * mode_index) * (1 + load_index ** 2) + (i % 7) * .001
                    violated = mode == 'native_dynamic' and ((load_index >= 2 and i % 5 == 0) or (load_index == 1 and i == 0))
                    # The moderate-load violation deliberately has zero numeric
                    # max error; bitwise inequality must not become "exact".
                    difference = .0001 if violated and load_index >= 2 else 0.
                    row = dict(configuration=mode, offered_load=load, request_id=f'synthetic-{i}',
                               source_replay_index=i % 8, arrival_time=arrival, completion_time=arrival + latency,
                               arrival_to_completion_seconds=latency, offered_rate_requests_per_second=8 * ratio,
                               numerical_contract='violated' if violated else 'exact', numerical_max_abs_difference=difference,
                               reference_type='fixed_padded_same_position' if mode == 'fixed_size_padded' else 'mode_matched_singleton')
                    stream.write(json.dumps(row) + '\n')
    proof = directory / 'synthetic_proof.json'
    proof.write_text(json.dumps(dict(status='complete_isolated_four_model_performance_independently_verified',
                                    model_id='SYNTHETIC_ONLY', source_evidence_sha256={str(raw.resolve()): sha(raw)})))
    return raw, proof


class ServingFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-serving-figure-controls-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.raw, self.proof = fixture(self.directory)
        self.output = self.directory / 'figures'
        subprocess.run([sys.executable, '-m', 'scripts.plot_serving_stability_tradeoff',
                        '--raw', str(self.raw), '--audit', str(self.proof), '--output-dir', str(self.output)],
                       cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
        self.table_path = self.output / 'serving_tradeoff.json'
        self.table = json.loads(self.table_path.read_text())

    def mutate(self):
        self.table_path.write_text(json.dumps(self.table))

    def test_positive_full_synthetic_protocol(self):
        result = audit(self.table_path)
        self.assertEqual(result['groups_verified'], 16)
        self.assertEqual(result['requests_verified'], 4000)

    def test_zero_numeric_error_does_not_hide_violation(self):
        row = next(r for r in self.table['rows'] if r['configuration'] == 'native_dynamic' and r['offered_load'] == 'moderate')
        self.assertEqual(row['maximum_absolute_difference'], 0.)
        self.assertEqual(row['violated_reference_requests'], 1)
        self.assertFalse(row['all_requests_exact'])

    def test_wrong_p95_rejected(self):
        self.table['rows'][0]['p95_arrival_to_completion_seconds'] = 999.
        self.mutate()
        with self.assertRaises(ValueError): audit(self.table_path)

    def test_missing_group_rejected(self):
        self.table['rows'].pop()
        self.mutate()
        with self.assertRaises(ValueError): audit(self.table_path)

    def test_duplicate_group_rejected(self):
        self.table['rows'].append(copy.deepcopy(self.table['rows'][0]))
        self.mutate()
        with self.assertRaises(ValueError): audit(self.table_path)

    def test_padding_reference_conflation_rejected(self):
        row = next(r for r in self.table['rows'] if r['configuration'] == 'fixed_size_padded')
        row['reference_type'] = 'mode_matched_singleton'
        self.mutate()
        with self.assertRaises(ValueError): audit(self.table_path)

    def test_rehashed_csv_corruption_rejected(self):
        item = next(a for a in self.table['artifacts'] if a['path'].endswith('.csv'))
        path = Path(item['path'])
        path.write_text(path.read_text().replace(',250,', ',249,', 1))
        item['sha256'] = sha(path)
        self.mutate()
        with self.assertRaises(ValueError): audit(self.table_path)

    def test_duplicate_raw_request_rejected(self):
        with self.raw.open('a') as stream:
            stream.write(self.raw.read_text().splitlines()[0] + '\n')
        value = json.loads(self.proof.read_text())
        value['source_evidence_sha256'][str(self.raw.resolve())] = sha(self.raw)
        self.proof.write_text(json.dumps(value))
        with self.assertRaises(ValueError): summarize(self.raw, self.proof)


if __name__ == '__main__':
    unittest.main()

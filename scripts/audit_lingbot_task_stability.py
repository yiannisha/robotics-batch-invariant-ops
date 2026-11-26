"""Recount task tables from independently accepted full LingBot magnitudes."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo) if lo != hi else ordered[lo]


def expected_row(task, mode, arm, starts):
    values = [s['comparisons'][mode] for s in starts]
    positions = [v['arms'][arm]['position_max_m'] * 1000. for v in values]
    rotations = [v['arms'][arm]['rotation_max_rad'] for v in values]
    first = [v['first_action_divergence_control_index'] for v in values if v['first_action_divergence_control_index'] is not None]
    pairs = [pair for v in values for pair in v['fixed_identical_input_nonsingleton_pairs']]
    elements = sum(pair['elements'] for pair in pairs)
    squared = math.fsum(pair['squared_error_sum'] for pair in pairs)
    return dict(task=task, implementation=mode, arm=arm, starts=len(starts),
                starts_exposed_to_nonsingleton_calls=sum(v['batched_policy_calls'] != 0 for v in values),
                starts_with_fixed_identical_input_batch_comparison=sum(len(v['fixed_identical_input_nonsingleton_pairs']) != 0 for v in values),
                starts_with_action_divergence=sum(v['action_disagreement_controls'] != 0 for v in values),
                conditional_median_first_divergence_control=percentile(first, .5) if first else None,
                mean_peak_position_drift_mm=math.fsum(positions) / len(positions),
                median_peak_position_drift_mm=percentile(positions, .5),
                p95_peak_position_drift_mm=percentile(positions, .95),
                worst_position_drift_mm=max(positions),
                median_peak_rotation_drift_rad=percentile(rotations, .5),
                worst_rotation_drift_rad=max(rotations), fixed_identical_input_pairs=len(pairs),
                pooled_fixed_input_mse=squared / elements if elements else None,
                pooled_fixed_input_rms=math.sqrt(squared / elements) if elements else None,
                maximum_fixed_input_absolute_error=max((p['max_absolute_error'] for p in pairs), default=None))


def verify_row(actual, expected):
    check(actual.keys() == expected.keys(), 'Task table schema differs')
    for key, value in expected.items():
        if isinstance(value, (int, float)):
            check(isinstance(actual[key], (int, float)) and math.isfinite(actual[key])
                  and math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-20), 'Task statistic differs: ' + key)
        else:
            check(actual[key] == value, 'Task statistic differs: ' + key)


def verify_csv(path, rows):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == list(rows[0]), 'CSV field order/schema differs')
        actual = list(reader)
    expected = [{key: '' if value is None else str(value) for key, value in row.items()} for row in rows]
    check(actual == expected, 'CSV values differ from JSON table')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--csv-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Preserve prior audit')
    table_hash = sha(args.table)
    csv_hash = sha(args.csv_table)
    table = json.loads(args.table.read_text())
    source, audit_path = Path(table['report']), Path(table['magnitude_audit'])
    check(sha(source) == table['report_sha256'] and sha(audit_path) == table['magnitude_audit_sha256'], 'Source changed')
    acceptance = json.loads(audit_path.read_text())
    check(acceptance['status'] == 'full_lingbot_dual_arm_stability_magnitudes_independently_verified'
          and acceptance['paired_starts_verified'] == 2500 and acceptance['episode_files_verified'] == 20000
          and acceptance['report_sha256'] == sha(source), 'Complete magnitude acceptance required')
    groups = defaultdict(list)
    seen = set()
    for start in json.loads(source.read_text())['starts']:
        key = (start['task'], start['initialization'])
        check(key not in seen, 'Duplicate source start')
        seen.add(key)
        groups[start['task']].append(start)
    check(len(groups) == 50 and len(seen) == 2500
          and all({s['initialization'] for s in starts} == set(range(50)) for starts in groups.values()), 'Incomplete source cohort')
    rows = {(r['task'], r['implementation'], r['arm']): r for r in table['rows']}
    expected = {(task, mode, arm) for task in groups for mode in ('native', 'invariant') for arm in ('left', 'right')}
    check(rows.keys() == expected and len(table['rows']) == len(rows) == 200, 'Missing or duplicate task/mode/arm row')
    for task, mode, arm in sorted(expected):
        verify_row(rows[task, mode, arm], expected_row(task, mode, arm, groups[task]))
    verify_csv(args.csv_table, table['rows'])
    check(sha(args.table) == table_hash and sha(source) == table['report_sha256']
          and sha(audit_path) == table['magnitude_audit_sha256'] and sha(args.csv_table) == csv_hash,
          'Inputs changed during recount')
    result = dict(status='full_lingbot_task_stability_descriptive_tables_independently_verified',
                  table_sha256=table_hash, source_report_sha256=sha(source), magnitude_audit_sha256=sha(audit_path),
                  csv_sha256=csv_hash,
                  auditor_sha256=sha(Path(__file__)), verified_rows=200, verified_tasks=50, verified_starts=2500,
                  scope='Descriptive task/mode/arm arithmetic from independently accepted magnitudes; no new raw-frame audit or CI claim',
                  full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Independently verified200 task/mode/arm descriptive rows', flush=True)


if __name__ == '__main__':
    main()

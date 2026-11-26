"""Recount full-cohort paired-start arm intervals without reporter imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def percentiles(samples):
    ordered = np.sort(samples, axis=0)
    bounds = []
    for probability in (.025, .975):
        position = (len(ordered) - 1) * probability
        lo, hi = math.floor(position), math.ceil(position)
        fraction = position - lo
        bounds.append(ordered[lo] * (1 - fraction) + ordered[hi] * fraction)
    return np.asarray(bounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Preserve prior audit')
    report_hash = sha(args.report)
    report = json.loads(args.report.read_text())
    source, acceptance_path, draws_path = map(Path, (report['source_report'], report['magnitude_audit'], report['draws_file']))
    hashes = {source: report['source_report_sha256'], acceptance_path: report['magnitude_audit_sha256'],
              draws_path: report['draws_sha256']}
    check(all(sha(p) == h for p, h in hashes.items()), 'Source changed')
    acceptance = json.loads(acceptance_path.read_text())
    check(acceptance['status'] == 'full_lingbot_dual_arm_stability_magnitudes_independently_verified'
          and acceptance['report_sha256'] == sha(source) and acceptance['paired_starts_verified'] == 2500
          and acceptance['episode_files_verified'] == 20000, 'Full magnitude acceptance required')
    starts = {(s['task'], s['initialization']): s for s in json.loads(source.read_text())['starts']}
    check(len(starts) == 2500, 'Incomplete source start roster')
    tasks = {k[0] for k in starts}
    check(len(tasks) == 50 and starts.keys() == {(t, i) for t in tasks for i in range(50)}, 'Incomplete task roster')
    roster = {r['task']: r for r in report['rosters']}
    check(roster.keys() == tasks and len(report['rosters']) == 50, 'Missing/duplicate reported task')
    metrics = ('left_mean_peak_position_mm', 'right_mean_peak_position_mm',
               'left_mean_peak_rotation_rad', 'right_mean_peak_rotation_rad',
               'left_median_peak_position_mm', 'right_median_peak_position_mm',
               'left_median_peak_rotation_rad', 'right_median_peak_rotation_rad',
               'fraction_starts_with_action_divergence')
    rows = {(r['task'], r['comparison'], r['metric']): r for r in report['rows']}
    expected_keys = {(t, mode, metric) for t in tasks for mode in ('native', 'invariant', 'native_minus_invariant') for metric in metrics}
    check(rows.keys() == expected_keys and len(report['rows']) == 1350, 'Missing/duplicate interval')
    check(report['resamples'] >= 10000 and report['seed'] >= 0, 'Invalid resampling protocol')
    with np.load(draws_path, allow_pickle=False) as saved:
        check(set(saved.files) == {r['draw_key'] for r in roster.values()}, 'Draw inventory differs')
        for task in sorted(tasks):
            info = roster[task]
            check(info['initializations'] == list(range(50)), 'Paired start order differs')
            key = hashlib.sha256(json.dumps(['lingbot', task]).encode()).hexdigest()
            entropy = [report['seed']] + [int.from_bytes(bytes.fromhex(key)[i:i + 4], 'little') for i in range(0, 32, 4)]
            check(info['draw_key'] == key and info['seed_entropy'] == entropy, 'Task RNG identity differs')
            indices = saved[key]
            check(indices.dtype == np.int32 and indices.shape == (report['resamples'], 50)
                  and np.all((indices >= 0) & (indices < 50)), 'Invalid resampling indices')
            recreated = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy))).integers(0, 50, size=indices.shape, dtype=np.int32)
            check(np.array_equal(indices, recreated), 'Retained draws differ from seed')
            expected_exposed = sum(starts[task, i]['comparisons']['native']['batched_policy_calls'] > 0 for i in range(50))
            expected_fixed = sum(bool(starts[task, i]['comparisons']['native']['fixed_identical_input_nonsingleton_pairs']) for i in range(50))
            check(info['native_starts_exposed_to_nonsingleton_calls'] == expected_exposed
                  and info['native_starts_with_fixed_identical_input_comparison'] == expected_fixed, 'Exposure counts differ')
            points, distributions = {}, {}
            for mode in ('native', 'invariant'):
                values = []
                for initial in range(50):
                    comparison = starts[task, initial]['comparisons'][mode]
                    arm = comparison['arms']
                    values.append([1000. * arm['left']['position_max_m'], 1000. * arm['right']['position_max_m'],
                                   arm['left']['rotation_max_rad'], arm['right']['rotation_max_rad'],
                                   int(comparison['action_disagreement_controls'] != 0)])
                data = np.asarray(values, dtype=np.float64)
                check(np.isfinite(data).all() and np.all(data >= 0), 'Invalid source magnitudes')
                if mode == 'invariant':
                    check(np.all(data == 0), 'Invariant drift')
                ordered = np.sort(data[:, :4], axis=0)
                points[mode] = np.r_[np.sum(data[:, :4], axis=0) / 50., (ordered[24] + ordered[25]) / 2,
                                     np.sum(data[:, 4]) / 50.]
                selected = data[indices]
                ordered = np.sort(selected[:, :, :4], axis=1)
                distributions[mode] = np.column_stack((np.sum(selected[:, :, :4], axis=1) / 50.,
                                                       (ordered[:, 24, :] + ordered[:, 25, :]) / 2,
                                                       np.sum(selected[:, :, 4], axis=1) / 50.))
            points['native_minus_invariant'] = points['native'] - points['invariant']
            distributions['native_minus_invariant'] = distributions['native'] - distributions['invariant']
            for mode, samples in distributions.items():
                bounds = percentiles(samples)
                for column, metric in enumerate(metrics):
                    row = rows[task, mode, metric]
                    check(row['starts'] == 50, 'Interval start count differs')
                    for field, value in [('estimate', points[mode][column]), ('ci95_low', bounds[0, column]), ('ci95_high', bounds[1, column])]:
                        check(math.isfinite(row[field]) and math.isclose(row[field], value, rel_tol=1e-11, abs_tol=1e-20),
                              task + '/' + mode + '/' + metric + ': ' + field + ' differs')
    check(sha(args.report) == report_hash and all(sha(p) == h for p, h in hashes.items()), 'Sources changed during recount')
    result = dict(status='full_lingbot_paired_start_arm_bootstrap_independently_recounted',
                  report_sha256=report_hash, source_hashes={str(p): h for p, h in hashes.items()},
                  auditor_sha256=sha(Path(__file__)), verified_starts=2500, verified_tasks=50, verified_estimates=1350,
                  resamples=report['resamples'], scope='Conditional paired-start percentile intervals, separate arms; no population guarantee or numerical-MSE CI',
                  full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Independently recounted1350 paired-start arm estimates', flush=True)


if __name__ == '__main__':
    main()

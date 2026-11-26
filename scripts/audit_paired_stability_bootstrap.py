"""Rebuild paired-start bootstrap estimates from audited start records and draws."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Preserve prior audits')
    report_hash = digest(args.report)
    report = json.loads(args.report.read_text())
    source = Path(report['source_report'])
    accepted_path = Path(report['magnitude_audit'])
    draws_path = Path(report['draws_file'])
    paths = {source: report['source_report_sha256'], accepted_path: report['magnitude_audit_sha256'],
             draws_path: report['draws_sha256']}
    check(all(digest(p) == h for p, h in paths.items()), 'Changed bootstrap source')
    accepted = json.loads(accepted_path.read_text())
    check(accepted['status'] == 'reported_subset_stability_metrics_independently_verified'
          and accepted['report_sha256'] == digest(source), 'Unaccepted magnitude source')
    check(report['resamples'] >= 10000, 'Insufficient resampling')
    cohort = {}
    for study in json.loads(source.read_text())['studies']:
        for start in study['starts']:
            key = (study['model'], str(start['task']), start['initialization'])
            check(key not in cohort, 'Duplicate source start')
            cohort[key] = start
    roster = {(r['model'], str(r['task']), r['initialization']) for r in accepted['paired_starts_verified']}
    check(cohort.keys() == roster and len(roster) == len(accepted['paired_starts_verified']), 'Magnitude roster mismatch')
    tasks = {(k[0], k[1]) for k in cohort}
    groups = {(r['model'], r['task']): r for r in report['rosters']}
    check(groups.keys() == tasks and len(groups) == len(report['rosters']), 'Missing/duplicate bootstrap task')
    metrics = ('mean_per_start_fixed_input_mse', 'mean_peak_eef_drift_mm',
               'median_peak_eef_drift_mm', 'fraction_starts_with_action_divergence')
    table = {(r['model'], r['task'], r['comparison'], r['metric']): r for r in report['rows']}
    expected = {(m, t, mode, metric) for m, t in tasks for mode in
                ('native', 'invariant', 'native_minus_invariant') for metric in metrics}
    check(table.keys() == expected and len(table) == len(report['rows']), 'Missing/duplicate estimates')
    with np.load(draws_path, allow_pickle=False) as retained:
        check(set(retained.files) == {r['draw_key'] for r in groups.values()}, 'Draw inventory mismatch')
        for (model, task), info in sorted(groups.items()):
            ids = sorted(k[2] for k in cohort if k[:2] == (model, task))
            n = len(ids)
            check(info['initializations'] == ids and info['starts'] == n, 'Start order mismatch')
            key = hashlib.sha256(json.dumps([model, task]).encode()).hexdigest()
            check(info['draw_key'] == key, 'Draw task key mismatch')
            entropy = [report['seed']] + [int.from_bytes(bytes.fromhex(key)[i:i + 4], 'little') for i in range(0, 32, 4)]
            check(entropy == info['seed_entropy'], 'Task RNG seed mismatch')
            indices = retained[key]
            check(indices.dtype == np.int32 and indices.shape == (report['resamples'], n)
                  and n >= 2 and np.all((indices >= 0) & (indices < n)), 'Invalid resample indices')
            recreated = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy))).integers(
                0, n, size=indices.shape, dtype=np.int32)
            check(np.array_equal(indices, recreated), 'Retained draws differ from declared seed')
            distributions, points = {}, {}
            for mode in ('native', 'invariant'):
                data = []
                for initial in ids:
                    start = cohort[model, task, initial]
                    check(all(start['same_condition_repeat_controls_exact'].values()), 'Inexact repeat')
                    value = start['comparisons'][mode]
                    pairs = value['fixed_identical_input_nonsingleton_pairs']
                    check(pairs, 'No fixed-input pairs')
                    mse = math.fsum(p['squared_error_sum'] for p in pairs) / sum(p['elements'] for p in pairs)
                    data.append([mse, value['eef_position_max_m'] * 1000., int(value['action_disagreement_controls'] != 0)])
                data = np.asarray(data, dtype=np.float64)
                check(np.all(np.isfinite(data)) and np.all(data >= 0), 'Invalid start metrics')
                selected = data[indices]
                ordered = np.sort(selected[:, :, 1], axis=1)
                medians = (ordered[:, (n - 1) // 2] + ordered[:, n // 2]) / 2
                distributions[mode] = np.column_stack((selected[:, :, 0].sum(axis=1) / n,
                                                       selected[:, :, 1].sum(axis=1) / n,
                                                       medians, selected[:, :, 2].sum(axis=1) / n))
                points[mode] = [math.fsum(data[:, 0]) / n, math.fsum(data[:, 1]) / n,
                                statistics.median(data[:, 1]), math.fsum(data[:, 2]) / n]
            distributions['native_minus_invariant'] = distributions['native'] - distributions['invariant']
            points['native_minus_invariant'] = [a - b for a, b in zip(points['native'], points['invariant'])]
            for mode, samples in distributions.items():
                for column, metric in enumerate(metrics):
                    row = table[model, task, mode, metric]
                    percentiles = statistics.quantiles(samples[:, column], n=1000, method='inclusive')
                    check(row['starts'] == n, 'Estimate start count mismatch')
                    for field, value in [('estimate', points[mode][column]),
                                         ('ci95_low', percentiles[24]), ('ci95_high', percentiles[974])]:
                        check(math.isclose(row[field], value, rel_tol=1e-11, abs_tol=1e-20),
                              f'{model}/{task}/{mode}/{metric}: {field} differs')
    check(digest(args.report) == report_hash and all(digest(p) == h for p, h in paths.items()), 'Inputs changed')
    result = dict(status='paired_start_bootstrap_independently_recounted',
                  report_sha256=report_hash, source_hashes={str(p): h for p, h in paths.items()},
                  auditor_sha256=digest(Path(__file__)), verified_tasks=len(tasks), verified_starts=len(cohort),
                  verified_estimates=len(table), resamples=report['resamples'],
                  scope='Paired-start percentile intervals for retained audited cohorts; no population guarantee',
                  full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f'Independently verified {len(table)} bootstrap estimates / {len(cohort)} starts', flush=True)


if __name__ == '__main__':
    main()

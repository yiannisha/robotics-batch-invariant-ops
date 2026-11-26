"""Task-specific paired-start uncertainty from independently audited magnitudes.

Frames, inference calls and noise replicas are not resampling units. Percentile
intervals describe the retained start cohort; degenerate intervals at zero or
one do not establish population-wide certainty.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def start_values(comparison):
    pairs = comparison['fixed_identical_input_nonsingleton_pairs']
    require(pairs, 'Start lacks a fixed-identical-input batching comparison')
    elements = sum(p['elements'] for p in pairs)
    require(elements > 0, 'Empty fixed-input output')
    mse = sum(p['squared_error_sum'] for p in pairs) / elements
    peak = comparison['eef_position_max_m'] * 1000
    divergent = float(comparison['action_disagreement_controls'] > 0)
    values = [mse, peak, divergent]
    require(all(np.isfinite(v) and v >= 0 for v in values), 'Invalid start values')
    return values


def statistics(values):
    return np.array([values[:, 0].mean(), values[:, 1].mean(),
                     np.median(values[:, 1]), values[:, 2].mean()])


METRICS = ['mean_per_start_fixed_input_mse', 'mean_peak_eef_drift_mm',
           'median_peak_eef_drift_mm', 'fraction_starts_with_action_divergence']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'magnitude-audit', 'output', 'draws-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    require(args.resamples >= 10000, 'At least10000 resamples required')
    require(not args.output.exists() and not args.draws_output.exists(), 'Preserve existing artifacts')
    report_hash, audit_hash = sha(args.report), sha(args.magnitude_audit)
    report = json.loads(args.report.read_text())
    audit = json.loads(args.magnitude_audit.read_text())
    require(audit['status'] == 'reported_subset_stability_metrics_independently_verified'
            and audit['report_sha256'] == report_hash, 'Matching accepted magnitude audit required')
    accepted = {(r['model'], str(r['task']), r['initialization']) for r in audit['paired_starts_verified']}
    require(len(accepted) == len(audit['paired_starts_verified']), 'Duplicate audited start')
    groups = {}
    actual = set()
    for study in report['studies']:
        for start in study['starts']:
            key = (study['model'], str(start['task']), start['initialization'])
            require(key not in actual, 'Duplicate reported start')
            actual.add(key)
            require(all(start['same_condition_repeat_controls_exact'].values()), 'Inexact repeat control')
            groups.setdefault(key[:2], []).append(start)
    require(actual == accepted, 'Reported and independently accepted rosters differ')
    arrays, rows, rosters = {}, [], []
    for (model, task), starts in sorted(groups.items()):
        starts.sort(key=lambda s: s['initialization'])
        n = len(starts)
        require(n >= 2, 'At least two starts required per task')
        key = hashlib.sha256(json.dumps([model, task]).encode()).hexdigest()
        entropy = [args.seed, *np.frombuffer(bytes.fromhex(key), dtype='<u4').tolist()]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
        draws = rng.integers(0, n, size=(args.resamples, n), dtype=np.int32)
        arrays[key] = draws
        rosters.append(dict(model=model, task=task, draw_key=key, starts=n,
                            initializations=[s['initialization'] for s in starts], seed_entropy=entropy))
        values = {mode: np.asarray([start_values(s['comparisons'][mode]) for s in starts], dtype=np.float64)
                  for mode in ('native', 'invariant')}
        require(np.all(values['invariant'] == 0), 'Invariant drift in accepted report')
        points, distributions = {}, {}
        for mode, data in values.items():
            points[mode] = statistics(data)
            samples = data[draws]
            distributions[mode] = np.stack([samples[:, :, 0].mean(axis=1), samples[:, :, 1].mean(axis=1),
                                            np.median(samples[:, :, 1], axis=1), samples[:, :, 2].mean(axis=1)], axis=1)
        points['native_minus_invariant'] = points['native'] - points['invariant']
        distributions['native_minus_invariant'] = distributions['native'] - distributions['invariant']
        for mode in points:
            intervals = np.quantile(distributions[mode], [.025, .975], axis=0, method='linear')
            for index, metric in enumerate(METRICS):
                rows.append(dict(model=model, task=task, starts=n, comparison=mode, metric=metric,
                                 estimate=float(points[mode][index]),
                                 ci95_low=float(intervals[0, index]), ci95_high=float(intervals[1, index])))
    args.draws_output.parent.mkdir(parents=True, exist_ok=True)
    with args.draws_output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    require(sha(args.report) == report_hash and sha(args.magnitude_audit) == audit_hash, 'Inputs changed')
    result = dict(status='paired_start_bootstrap_extracted_independent_recount_pending',
                  source_report=str(args.report.resolve()), source_report_sha256=report_hash,
                  magnitude_audit=str(args.magnitude_audit.resolve()), magnitude_audit_sha256=audit_hash,
                  source_sha256=sha(Path(__file__)), numpy_version=np.__version__,
                  seed=args.seed, resamples=args.resamples, rng='PCG64; task-keyed SeedSequence',
                  draws_file=str(args.draws_output.resolve()), draws_sha256=sha(args.draws_output),
                  unit='One restored simulator start; paired conditions share resampled indices',
                  method='95% percentile; linear quantiles; no multiplicity adjustment',
                  estimand='Task-specific retained-start distribution; no cross-model pooling',
                  mse_scope='Equal weight per start; within-start element-weighted identical-input deviations only',
                  limitations='Conditional cohort estimates; partial tasks stay partial. Degenerate intervals are not population guarantees.',
                  full_campaign_complete=False, rosters=rosters, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f'Bootstrapped {len(groups)} tasks / {len(actual)} paired starts; independent recount pending', flush=True)


if __name__ == '__main__':
    main()

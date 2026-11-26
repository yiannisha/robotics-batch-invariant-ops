"""Paired-start percentile intervals for the accepted complete dual-arm cohort."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

METRICS = ('left_mean_peak_position_mm', 'right_mean_peak_position_mm',
           'left_mean_peak_rotation_rad', 'right_mean_peak_rotation_rad',
           'left_median_peak_position_mm', 'right_median_peak_position_mm',
           'left_median_peak_rotation_rad', 'right_median_peak_rotation_rad',
           'fraction_starts_with_action_divergence')


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values(starts, mode):
    result = []
    for start in starts:
        comparison = start['comparisons'][mode]
        arms = comparison['arms']
        result.append([arms['left']['position_max_m'] * 1000., arms['right']['position_max_m'] * 1000.,
                       arms['left']['rotation_max_rad'], arms['right']['rotation_max_rad'],
                       float(comparison['action_disagreement_controls'] > 0)])
    array = np.asarray(result, dtype=np.float64)
    check(array.shape == (len(starts), 5) and np.isfinite(array).all() and np.all(array >= 0), 'Invalid arm magnitudes')
    return array


def estimate(data):
    return np.concatenate((data[:, :4].mean(axis=0), np.median(data[:, :4], axis=0), [data[:, 4].mean()]))


def resample(data, indices):
    samples = data[indices]
    return np.concatenate((samples[:, :, :4].mean(axis=1), np.median(samples[:, :, :4], axis=1),
                           samples[:, :, 4].mean(axis=1)[:, None]), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'magnitude-audit', 'output', 'draws-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    check(args.resamples >= 10000 and args.seed >= 0, 'Invalid bootstrap protocol')
    check(not args.output.exists() and not args.draws_output.exists(), 'Preserve prior artifacts')
    report_hash, audit_hash = sha(args.report), sha(args.magnitude_audit)
    report, audit = json.loads(args.report.read_text()), json.loads(args.magnitude_audit.read_text())
    check(audit['status'] == 'full_lingbot_dual_arm_stability_magnitudes_independently_verified'
          and audit['paired_starts_verified'] == 2500 and audit['episode_files_verified'] == 20000
          and audit['report_sha256'] == report_hash, 'Full magnitude acceptance required')
    groups, seen = defaultdict(list), set()
    for start in report['starts']:
        key = (start['task'], start['initialization'])
        check(key not in seen and all(start['same_condition_repeat_controls_exact'].values()), 'Duplicate start or inexact repeat')
        seen.add(key)
        groups[start['task']].append(start)
    check(len(groups) == 50 and len(seen) == 2500
          and all({s['initialization'] for s in g} == set(range(50)) for g in groups.values()), 'Full50-by50 cohort required')
    draws, rosters, rows = {}, [], []
    for task, starts in sorted(groups.items()):
        starts.sort(key=lambda start: start['initialization'])
        key = hashlib.sha256(json.dumps(['lingbot', task]).encode()).hexdigest()
        entropy = [args.seed, *np.frombuffer(bytes.fromhex(key), dtype='<u4').tolist()]
        indices = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy))).integers(
            0, 50, size=(args.resamples, 50), dtype=np.int32)
        draws[key] = indices
        rosters.append(dict(task=task, initializations=[s['initialization'] for s in starts],
                            draw_key=key, seed_entropy=entropy,
                            native_starts_exposed_to_nonsingleton_calls=sum(s['comparisons']['native']['batched_policy_calls'] > 0 for s in starts),
                            native_starts_with_fixed_identical_input_comparison=sum(bool(s['comparisons']['native']['fixed_identical_input_nonsingleton_pairs']) for s in starts)))
        points, distributions = {}, {}
        for mode in ('native', 'invariant'):
            data = values(starts, mode)
            if mode == 'invariant':
                check(np.all(data == 0), 'Invariant trajectory drift')
            points[mode], distributions[mode] = estimate(data), resample(data, indices)
        points['native_minus_invariant'] = points['native'] - points['invariant']
        distributions['native_minus_invariant'] = distributions['native'] - distributions['invariant']
        for mode in points:
            intervals = np.quantile(distributions[mode], [.025, .975], axis=0, method='linear')
            for column, metric in enumerate(METRICS):
                rows.append(dict(task=task, starts=50, comparison=mode, metric=metric,
                                 estimate=float(points[mode][column]), ci95_low=float(intervals[0, column]),
                                 ci95_high=float(intervals[1, column])))
    args.draws_output.parent.mkdir(parents=True, exist_ok=True)
    with args.draws_output.open('xb') as stream:
        np.savez_compressed(stream, **draws)
    check(sha(args.report) == report_hash and sha(args.magnitude_audit) == audit_hash, 'Source changed')
    result = dict(status='full_lingbot_paired_start_arm_bootstrap_extracted_independent_recount_pending',
                  source_report=str(args.report.resolve()), source_report_sha256=report_hash,
                  magnitude_audit=str(args.magnitude_audit.resolve()), magnitude_audit_sha256=audit_hash,
                  reporter_sha256=sha(Path(__file__)), numpy_version=np.__version__, seed=args.seed,
                  resamples=args.resamples, rng='Task-keyed PCG64/SeedSequence',
                  draws_file=str(args.draws_output.resolve()), draws_sha256=sha(args.draws_output),
                  method='95% percentile, linear quantiles, no multiplicity adjustment',
                  unit='Whole restored start, paired native/invariant indices, separately per task',
                  limitations='Conditional on retained cohort; degenerate intervals do not establish population certainty. All50 starts included per task; exposure counts explicit. No numerical-MSE confidence interval claimed.',
                  rosters=rosters, rows=rows, full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Extracted1350 paired-start arm stability estimates; independent recount pending', flush=True)


if __name__ == '__main__':
    main()

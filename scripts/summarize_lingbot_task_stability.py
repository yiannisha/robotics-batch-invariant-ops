"""Task-specific descriptive tables from accepted full LingBot magnitudes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def task_rows(task, starts):
    rows = []
    for mode in ('native', 'invariant'):
        comparisons = [s['comparisons'][mode] for s in starts]
        pairs = [p for c in comparisons for p in c['fixed_identical_input_nonsingleton_pairs']]
        elements = sum(p['elements'] for p in pairs)
        squared = math.fsum(p['squared_error_sum'] for p in pairs)
        for arm in ('left', 'right'):
            peaks = [c['arms'][arm]['position_max_m'] * 1000 for c in comparisons]
            rotations = [c['arms'][arm]['rotation_max_rad'] for c in comparisons]
            require(all(math.isfinite(v) and v >= 0 for v in peaks + rotations), 'Invalid arm magnitudes')
            divergences = [c['first_action_divergence_control_index'] for c in comparisons
                           if c['first_action_divergence_control_index'] is not None]
            rows.append(dict(task=task, implementation=mode, arm=arm, starts=len(starts),
                             starts_exposed_to_nonsingleton_calls=sum(c['batched_policy_calls'] > 0 for c in comparisons),
                             starts_with_fixed_identical_input_batch_comparison=sum(bool(c['fixed_identical_input_nonsingleton_pairs']) for c in comparisons),
                             starts_with_action_divergence=sum(c['action_disagreement_controls'] > 0 for c in comparisons),
                             conditional_median_first_divergence_control=(statistics.median(divergences) if divergences else None),
                             mean_peak_position_drift_mm=statistics.fmean(peaks),
                             median_peak_position_drift_mm=statistics.median(peaks),
                             p95_peak_position_drift_mm=statistics.quantiles(peaks, n=100, method='inclusive')[94],
                             worst_position_drift_mm=max(peaks),
                             median_peak_rotation_drift_rad=statistics.median(rotations),
                             worst_rotation_drift_rad=max(rotations),
                             fixed_identical_input_pairs=len(pairs),
                             pooled_fixed_input_mse=(squared / elements if elements else None),
                             pooled_fixed_input_rms=(math.sqrt(squared / elements) if elements else None),
                             maximum_fixed_input_absolute_error=max((p['max_absolute_error'] for p in pairs), default=None)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'magnitude-audit', 'output', 'csv-output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not args.csv_output.exists(), 'Preserve prior summaries')
    report_hash, audit_hash = sha(args.report), sha(args.magnitude_audit)
    report = json.loads(args.report.read_text())
    audit = json.loads(args.magnitude_audit.read_text())
    require(audit['status'] == 'full_lingbot_dual_arm_stability_magnitudes_independently_verified'
            and audit['paired_starts_verified'] == 2500 and audit['episode_files_verified'] == 20000
            and audit['report_sha256'] == report_hash, 'Complete independent magnitude acceptance required')
    groups = defaultdict(list)
    roster = set()
    for start in report['starts']:
        key = (start['task'], start['initialization'])
        require(key not in roster, 'Duplicate start')
        roster.add(key)
        groups[start['task']].append(start)
    require(len(groups) == 50 and len(roster) == 2500
            and all({s['initialization'] for s in values} == set(range(50)) for values in groups.values()),
            'Complete50-by50 cohort required')
    rows = [row for task, starts in sorted(groups.items()) for row in task_rows(task, starts)]
    require(sha(args.report) == report_hash and sha(args.magnitude_audit) == audit_hash, 'Inputs changed')
    result = dict(primary_endpoint='singleton_vs_batched_stability',
                  status='lingbot_task_descriptive_tables_extracted_independent_recount_pending',
                  report=str(args.report.resolve()), report_sha256=report_hash,
                  magnitude_audit=str(args.magnitude_audit.resolve()), magnitude_audit_sha256=audit_hash,
                  reporter_sha256=sha(Path(__file__)),
                  scope='50tasks,50starts each; separate arms; native/invariant comparisons within each task',
                  units='Position in mm; rotation in radians; fixed-input normalized output errors in model units',
                  limitations='Descriptive, no confidence intervals. Numerical error is null if unobserved; no sample-variance claim. Numerical action error is shared across both arm rows, not an arm-specific quantity.',
                  rows=rows, full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Extracted200 LingBot task/mode/arm descriptive rows; independent recount pending', flush=True)


if __name__ == '__main__':
    main()

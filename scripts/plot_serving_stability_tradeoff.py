"""Plot accepted queue throughput/latency with explicit numerical references."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES = ('native_dynamic', 'batch_invariant_dynamic', 'serial_singleton', 'fixed_size_padded')
LOADS = ('underload', 'moderate', 'near_saturation', 'saturation')
RATIOS = (.25, .55, .9, 1.3)
FIELDS = ('configuration', 'offered_load', 'requests', 'offered_rate_requests_per_second',
          'observation_window_seconds', 'throughput_requests_per_second',
          'median_arrival_to_completion_seconds', 'p95_arrival_to_completion_seconds',
          'exact_reference_requests', 'violated_reference_requests', 'all_requests_exact',
          'reference_type', 'maximum_absolute_difference', 'unique_replay_fixtures')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(raw, audit):
    proof = json.loads(audit.read_text())
    if (proof['status'] != 'complete_isolated_four_model_performance_independently_verified'
            or proof['source_evidence_sha256'].get(str(raw.resolve())) != sha(raw)):
        raise ValueError('Matching complete independent performance acceptance required')
    groups = defaultdict(list)
    ids = set()
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['configuration'], row['offered_load']
            identity = (*key, row['request_id'])
            if key[0] not in MODES or key[1] not in LOADS or identity in ids:
                raise ValueError('Foreign or duplicate queue request')
            ids.add(identity)
            if row['numerical_contract'] not in ('exact', 'violated'):
                raise ValueError('Unverified numerical request')
            if not all(math.isfinite(row[k]) for k in ('arrival_time', 'completion_time',
                                                      'arrival_to_completion_seconds',
                                                      'offered_rate_requests_per_second',
                                                      'numerical_max_abs_difference')):
                raise ValueError('Nonfinite queue measurement')
            if row['arrival_to_completion_seconds'] <= 0 or row['numerical_max_abs_difference'] < 0:
                raise ValueError('Invalid queue metric')
            groups[key].append(row)
    if set(groups) != {(mode, load) for mode in MODES for load in LOADS}:
        raise ValueError('Complete four-mode/four-load table required')
    table = []
    for mode in MODES:
        for load in LOADS:
            rows = groups[mode, load]
            if len(rows) < 250:
                raise ValueError('At least 250 requests per group required')
            reference = 'fixed_padded_same_position' if mode == 'fixed_size_padded' else 'mode_matched_singleton'
            if {r['reference_type'] for r in rows} != {reference}:
                raise ValueError('Wrong numerical reference')
            rates = {r['offered_rate_requests_per_second'] for r in rows}
            if len(rates) != 1 or min(rates) <= 0:
                raise ValueError('Ambiguous offered rate')
            window = max(r['completion_time'] for r in rows) - min(r['arrival_time'] for r in rows)
            if window <= 0:
                raise ValueError('Invalid observation window')
            latency = [r['arrival_to_completion_seconds'] for r in rows]
            exact = sum(r['numerical_contract'] == 'exact' for r in rows)
            table.append(dict(configuration=mode, offered_load=load, requests=len(rows),
                              offered_rate_requests_per_second=next(iter(rates)),
                              observation_window_seconds=window, throughput_requests_per_second=len(rows) / window,
                              median_arrival_to_completion_seconds=statistics.median(latency),
                              p95_arrival_to_completion_seconds=statistics.quantiles(latency, n=100, method='inclusive')[94],
                              exact_reference_requests=exact, violated_reference_requests=len(rows) - exact,
                              all_requests_exact=exact == len(rows), reference_type=reference,
                              maximum_absolute_difference=max(r['numerical_max_abs_difference'] for r in rows),
                              unique_replay_fixtures=len({r['source_replay_index'] for r in rows})))
    return proof, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'audit', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    raw_hash, audit_hash = sha(args.raw), sha(args.audit)
    proof, rows = summarize(args.raw, args.audit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    titles = ('Native dynamic; native B1 reference', 'Invariant dynamic; invariant B1 reference',
              'Native serial singleton; B1 reference', 'Native fixed padding; B8 same-position reference')
    colors = ('#a43b20', '#1864b0', '#705e9a', '#258068')
    for mode_index, (mode, title, color) in enumerate(zip(MODES, titles, colors)):
        values = [r for r in rows if r['configuration'] == mode]
        x = [r['throughput_requests_per_second'] for r in values]
        y = [r['p95_arrival_to_completion_seconds'] for r in values]
        ax.plot(x, y, color=color, label=title, linewidth=1.3)
        for index, (a, b, row) in enumerate(zip(x, y, values)):
            marker = 'o' if row['all_requests_exact'] else 'x'
            ax.scatter([a], [b], marker=marker, color=color, s=44, zorder=4)
            ax.annotate(str(index + 1), (a, b), textcoords='offset points',
                        xytext=(4, 4 + 4 * (mode_index % 2)), color=color, fontsize=8)
    ax.set_xlabel('Observed completed throughput (requests/s)')
    ax.set_ylabel('p95 arrival-to-completion latency (s; logarithmic scale)')
    ax.set_yscale('log')
    ax.grid(which='both', alpha=.2)
    ax.legend(fontsize=8, loc='best')
    fig.suptitle(f"{proof['model_id']}: serving stability tradeoff", fontsize=13)
    fig.tight_layout(rect=(0, .16, 1, .94))
    fig.text(.5, .085, 'Point labels 1–4: offered load 0.25×, 0.55×, 0.90×, 1.30× native dynamic calibration.\n'
             'Circle: every request exact to its declared reference; ×: at least one violation.\n'
             'Fixed padding uses a same-position B8 reference, which does not establish equality to B1.',
             ha='center', fontsize=8)
    fig.text(.5, .025, 'Points connect in offered-load order. Throughput window: first arrival to last completion.\n'
             'Descriptive queue measurements on frozen replay fixtures; no confidence intervals or exclusive GPU reservation claimed.',
             ha='center', fontsize=8)
    artifacts = []
    for extension in ('png', 'pdf'):
        path = args.output_dir / f'serving_tradeoff.{extension}'
        fig.savefig(path, dpi=180)
        artifacts.append(dict(path=str(path.resolve()), sha256=sha(path)))
    plt.close(fig)
    csv_path = args.output_dir / 'serving_tradeoff.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    artifacts.append(dict(path=str(csv_path.resolve()), sha256=sha(csv_path)))
    if sha(args.raw) != raw_hash or sha(args.audit) != audit_hash:
        raise ValueError('Queue source changed during plotting')
    report = dict(model_id=proof['model_id'], raw=str(args.raw.resolve()), raw_sha256=raw_hash,
                  performance_audit=str(args.audit.resolve()), performance_audit_sha256=audit_hash,
                  rows=rows, plotter_sha256=sha(__file__), artifacts=artifacts,
                  matplotlib_version=matplotlib.__version__, confidence_intervals_claimed=False,
                  visual_inspection_complete=False, full_campaign_complete=False)
    (args.output_dir / 'serving_tradeoff.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

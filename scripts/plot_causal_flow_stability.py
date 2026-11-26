"""Export six-mode causal flow figures from accepted measurement contracts."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES = ('native', 'native_deterministic', 'existing_invariant_ops',
         'operators_plus_patch_projection', 'explicit_per_matrix_reference', 'full_invariant')
TITLES = ('Native', 'Native deterministic', 'Existing invariant ops',
          'Ops + patch projection', 'Per-matrix reference', 'Full invariant')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(raw, proof):
    acceptance = json.loads(proof.read_text())
    if (acceptance['status'] != 'complete_six_mode_causal_measurement_contracts_verified'
            or acceptance['measurement_records'] != 3600
            or acceptance['request_noise_pairs'] != 600 or acceptance['raw_sha256'] != sha(raw)):
        raise ValueError('Complete matching causal contract audit required')
    groups, rosters, seen = defaultdict(list), defaultdict(set), set()
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            mode, request, task = row['mode'], row['request_id'], str(row['task'])
            if mode not in MODES or (mode, request) in seen or row['batch_size'] != 2:
                raise ValueError('Invalid causal factorial')
            seen.add((mode, request))
            rosters[mode].add(request)
            for stage, metric in row['flow_steps'].items():
                if not re.fullmatch(r'flow\.step_\d+\.[A-Za-z_]+', stage):
                    continue
                value = metric['rmse']
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid retained error')
                groups[task, mode, stage].append((value, metric['exact']))
    if len(seen) != 3600 or set(rosters) != set(MODES) or any(
            roster != rosters['native'] or len(roster) != 600 for roster in rosters.values()):
        raise ValueError('Incomplete request roster')
    rows = []
    for (task, mode, stage), values in sorted(groups.items()):
        errors = [v for v, _ in values]
        rows.append(dict(task=task, mode=mode, stage=stage, comparisons=len(values),
                         exact_comparisons=sum(exact for _, exact in values),
                         median_rmse=statistics.median(errors),
                         p95_rmse=statistics.quantiles(errors, n=100, method='inclusive')[94],
                         max_rmse=max(errors)))
    if not rows or sha(raw) != acceptance['raw_sha256']:
        raise ValueError('No stages or raw measurements changed')
    return acceptance, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'audit', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    proof_hash = sha(args.audit)
    acceptance, rows = aggregate(args.raw, args.audit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tasks = sorted({r['task'] for r in rows})
    kinds = sorted({r['stage'].rsplit('.', 1)[1] for r in rows})
    artifacts = []
    colors = ('#a43b20', '#a17c00', '#8757ab', '#3d8682', '#566166', '#1864b0')
    for kind in kinds:
        for offset in range(0, len(tasks), 10):
            page = tasks[offset:offset + 10]
            fig, axes = plt.subplots(math.ceil(len(page) / 2), 2, squeeze=False,
                                     figsize=(12, 3 * math.ceil(len(page) / 2) + 2))
            for task, ax in zip(page, axes.flat):
                for mode, title, color in zip(MODES, TITLES, colors):
                    values = [r for r in rows if r['task'] == task and r['mode'] == mode
                              and r['stage'].endswith('.' + kind)]
                    values.sort(key=lambda r: int(r['stage'].split('.')[1].split('_')[1]))
                    if not values:
                        raise ValueError('Missing task/mode/stage series')
                    steps = [int(r['stage'].split('.')[1].split('_')[1]) for r in values]
                    ax.plot(steps, [r['median_rmse'] for r in values], color=color,
                            marker='o' if mode == 'full_invariant' else None,
                            markersize=3, linewidth=1.3, label=title)
                    ax.plot(steps, [r['p95_rmse'] for r in values], color=color,
                            linestyle=':', linewidth=1)
                exact = [r for r in rows if r['task'] == task and r['mode'] == 'full_invariant'
                         and r['stage'].endswith('.' + kind)]
                ax.set_title(f"Task {task}; full invariant exact "
                             f"{sum(r['exact_comparisons'] for r in exact)}/"
                             f"{sum(r['comparisons'] for r in exact)} stage comparisons", fontsize=9)
                ax.set_xlabel('Recorded denoising step (zero-based)')
                ax.set_ylabel('Singleton–B2 RMSE')
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(page):]:
                ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.5, .94), fontsize=8)
            fig.suptitle(f"{acceptance['model']}: causal {kind} propagation; tasks {offset + 1}–{offset + len(page)}",
                         fontsize=12)
            fig.tight_layout(rect=(0, .06, 1, .87))
            fig.text(.5, .015, 'Solid: median; dotted: p95 across retained request/noise comparisons. '
                     'Descriptive distributions, not confidence intervals.\n'
                     'B2 with diagnostic companions. Measured tensor-error summaries; '
                     'missing tensor arrays are not independently reconstructed.', ha='center', fontsize=8)
            for extension in ('png', 'pdf'):
                path = args.output_dir / f'{kind}_page{offset // 10 + 1:02d}.{extension}'
                fig.savefig(path, dpi=180)
                artifacts.append(dict(path=str(path.resolve()), sha256=sha(path)))
            plt.close(fig)
    if sha(args.audit) != proof_hash or sha(args.raw) != acceptance['raw_sha256']:
        raise ValueError('Source changed during plotting')
    table = dict(model=acceptance['model'], rows=rows, raw=str(args.raw.resolve()),
                 raw_sha256=acceptance['raw_sha256'], audit=str(args.audit.resolve()),
                 audit_sha256=proof_hash, plotter_sha256=sha(__file__),
                 matplotlib_version=matplotlib.__version__, artifacts=artifacts,
                 full_campaign_complete=False)
    (args.output_dir / 'plotted_table.json').write_text(json.dumps(table, indent=2) + '\n')


if __name__ == '__main__':
    main()

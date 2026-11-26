"""Plot within-model singleton/batch differences from audited paired reports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_frames(study):
    """Require one explicit recorded pose frame per task in versioned reports."""
    frames = {task: set() for task in study['tasks']}
    counts = {task: 0 for task in study['tasks']}
    for start in study['starts']:
        task = start['task']
        if task not in frames:
            raise ValueError('Foreign start task')
        frame = start.get('eef_pose_frame')
        if frame not in ('world', 'robot_base'):
            raise ValueError('Explicit EEF pose frame required')
        frames[task].add(frame)
        counts[task] += 1
    result = {}
    for task in frames:
        expected = ('robot_base' if study['model'] == 'cogact'
                    and task in ('open_drawer', 'close_drawer') else 'world')
        if frames[task] != {expected}:
            raise ValueError('Unexpected or mixed task pose frames')
        if any(counts[task] != study['tasks'][task][mode]['paired_starts']
               for mode in ('native', 'invariant')):
            raise ValueError('Frame roster differs from task summary')
        result[task] = expected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    if (audit['status'] != 'reported_subset_stability_metrics_independently_verified'
            or audit['report_sha256'] != digest(args.report)):
        raise ValueError('Matching independent magnitude audit required')
    if len(report['studies']) != 1:
        raise ValueError('Use one model per figure; no cross-model ranking')
    study = report['studies'][0]
    frames = task_frames(study)
    tasks = sorted(study['tasks'])
    if not tasks:
        raise ValueError('No tasks')
    series = []
    for task in tasks:
        modes = study['tasks'][task]
        if modes['native']['paired_starts'] != modes['invariant']['paired_starts']:
            raise ValueError('Unpaired start counts')
        for mode in ('native', 'invariant'):
            row = modes[mode]
            n = row['paired_starts']
            values = [row['fixed_input_rms_error'],
                      1000 * row['eef_position_median_start_max_m'],
                      100 * row['starts_with_action_divergence'] / n]
            if n <= 0 or not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError('Invalid metric')
            if row['fixed_identical_input_nonsingleton_calls'] <= 0:
                raise ValueError('No identical-input batch exposure')
            series.append(dict(task=task, mode=mode, starts=n, eef_pose_frame=frames[task],
                               normalized_action_rms=values[0],
                               median_start_peak_eef_mm=values[1],
                               action_divergence_percent=values[2]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, .45 * len(tasks) + 2)))
    fields = [('normalized_action_rms', 'Normalized action RMS difference'),
              ('median_start_peak_eef_mm', 'Median per-start peak EEF drift (mm)'),
              ('action_divergence_percent', 'Starts with action divergence (%)')]
    for ax, (field, label) in zip(axes, fields):
        for mode, offset, color, title in [('native', -.18, '#a43b20', 'Controlled native'),
                                           ('invariant', .18, '#18648b', 'Invariant')]:
            rows = [r for r in series if r['mode'] == mode]
            positions = [i + offset for i in range(len(tasks))]
            values = [r[field] for r in rows]
            ax.barh(positions, values, height=.32, color=color, label=title)
            for y, value in zip(positions, values):
                if value == 0:
                    ax.annotate('0', (0, y), xytext=(3, 0), textcoords='offset points',
                                va='center', color=color, fontsize=8)
        ax.set_yticks(range(len(tasks)),
                      [f"{t} (n={study['tasks'][t]['native']['paired_starts']})\n{frames[t].replace('_', ' ')} frame" for t in tasks])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if field == 'normalized_action_rms':
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3g}'))
        if field == 'action_divergence_percent':
            ax.set_xlim(0, 105)
            ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .88),
               ncol=2, fontsize=8)
    fig.suptitle(f"{study['model']}: singleton versus dynamic batching\n"
                 'Audited subset; full campaign qualification remains pending', fontsize=13)
    fig.tight_layout(rect=(0, .09, 1, .84))
    fig.text(.5, .035, 'Action RMS: identical prepared inputs and indexed noise only. '
             'EEF drift: closed-loop common horizon from restored starts.\n'
             'RMS is a paired difference, not centered variance. Bars are point summaries, '
             'not confidence intervals. Frames are task-specific; no pooling or model ranking.', ha='center', fontsize=8)
    artifacts = []
    for extension in ('png', 'pdf'):
        path = args.output_dir / f"{study['model']}_paired_stability.{extension}"
        fig.savefig(path, dpi=180)
        artifacts.append(dict(path=str(path.resolve()), sha256=digest(path)))
    plt.close(fig)
    provenance = dict(report=str(args.report.resolve()), report_sha256=digest(args.report),
                      independent_audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
                      plotter_sha256=digest(Path(__file__)), matplotlib_version=matplotlib.__version__,
                      plotted_rows=series, artifacts=artifacts, full_campaign_complete=False)
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f"Exported {study['model']} paired stability figure", flush=True)


if __name__ == '__main__':
    main()

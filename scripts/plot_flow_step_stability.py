"""Export task-specific propagation figures from independently recounted tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    if (audit['status'] != 'descriptive_flow_stage_distributions_independently_recounted'
            or audit['report_sha256'] != digest(args.report)):
        raise ValueError('Matching independent distribution audit required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tasks = sorted({row['task'] for row in report['rows']})
    kinds = sorted({row['stage'].rsplit('.', 1)[1] for row in report['rows']})
    artifacts = []
    for kind in kinds:
        fig, axes = plt.subplots(math.ceil(len(tasks) / 2), 2, squeeze=False,
                                 figsize=(11, 2.6 * math.ceil(len(tasks) / 2)))
        for task, ax in zip(tasks, axes.flat):
            for mode, color, title in [('native', '#a43b20', 'Native'),
                                       ('batch_invariant', '#18648b', 'Invariant')]:
                rows = [r for r in report['rows'] if r['task'] == task
                        and r['implementation'] == mode and r['batch_size'] == args.batch_size
                        and r['stage'].endswith('.' + kind)]
                rows.sort(key=lambda r: r['stage'])
                if not rows:
                    raise ValueError('Missing task/mode/batch/stage series')
                steps = [int(r['stage'].split('.')[1].split('_')[1]) for r in rows]
                median = [r['median_comparison_rmse'] for r in rows]
                tail = [r['p95_comparison_rmse'] for r in rows]
                ax.plot(steps, median, color=color, label=title + ' median', linewidth=1.5)
                ax.plot(steps, tail, color=color, linestyle=':', label=title + ' p95', linewidth=1)
                ax.fill_between(steps, median, tail, color=color, alpha=.10)
            ax.set_title('Task ' + task, fontsize=10)
            ax.set_xlabel('Recorded denoising step (zero-based)')
            ax.set_ylabel('Singleton–batch RMSE')
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
        for ax in list(axes.flat)[len(tasks):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7, loc='best')
        fig.suptitle(f"{report['model_id']}: {kind}, B={args.batch_size}\n"
                     'Identical inputs and indexed noise; diverse companions; target first', fontsize=12)
        fig.tight_layout(rect=(0, .025, 1, .95))
        fig.text(.5, .008, 'Descriptive distributions; shaded median–p95 band is not a confidence interval.',
                 ha='center', fontsize=8)
        for extension in ('png', 'pdf'):
            path = args.output_dir / f'{kind}_B{args.batch_size}.{extension}'
            fig.savefig(path, dpi=180)
            artifacts.append(dict(path=str(path.resolve()), sha256=digest(path)))
        plt.close(fig)
    provenance = dict(report=str(args.report.resolve()), report_sha256=digest(args.report),
                      independent_audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
                      plotter_sha256=digest(Path(__file__)), batch_size=args.batch_size,
                      matplotlib_version=matplotlib.__version__, artifacts=artifacts)
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Exported {len(artifacts)} figures', flush=True)


if __name__ == '__main__':
    main()

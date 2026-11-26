"""Observe compute/graphics isolation around source-bound operator timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.gpu_isolation import observe, foreign_processes
from scripts.run_isolated_frozen_benchmark import validate_freeze


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_inputs(freeze_path, manifest, plan_path):
    plan = json.loads(plan_path.read_text())
    builder = Path(__file__).with_name('build_operator_performance_plans_v2.py')
    if sha(builder) != plan['builder_sha256']:
        raise ValueError('Plan builder changed')
    runtime = validate_freeze(freeze_path, ['--adapter', plan['adapter'], '--replay-manifest', str(manifest)])
    if plan['adapter_execution_configuration'] != runtime['execution_configuration']:
        raise ValueError('Operator plan differs from frozen policy configuration')
    freeze = json.loads(freeze_path.read_text())
    bound = freeze['content_addressed_freeze']
    linked = set()
    for filename, expected in plan['source_sha256'].items():
        if sha(filename) != expected:
            raise ValueError('Operator plan source changed')
        if filename in bound and bound[filename]['sha256'] == expected:
            linked.add(filename)
            value = json.loads(Path(filename).read_text())
            # PI's bound qualifier configurations retain exact trace hashes.
            for source in value.get('source_evidence', []):
                if plan['source_sha256'].get(source['path']) == source['sha256']:
                    linked.add(source['path'])
    if linked != set(plan['source_sha256']):
        raise ValueError('Operator plan evidence is not linked to the numerical freeze')
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'replay-manifest', 'plan', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = args.output_dir / 'operator_timings.jsonl'
    journal = args.output_dir / 'gpu_observations.jsonl'
    summary_path = args.output_dir / 'isolation.json'
    report = dict(status='checking_inputs', accepted=False, gpu_uuid=args.gpu_uuid,
                  freeze=str(args.freeze.resolve()), freeze_sha256=sha(args.freeze),
                  replay_manifest=str(args.replay_manifest.resolve()), plan=str(args.plan.resolve()),
                  plan_sha256=sha(args.plan), raw=str(raw.resolve()),
                  observation_journal=str(journal.resolve()),
                  isolation_scope='Compute and graphics inventories sampled before/during/after timing; no exclusive hardware reservation claimed.',
                  full_campaign_complete=False)
    child = None
    observations = 0
    sources = [Path(__file__), Path(__file__).with_name('run_isolated_operator_benchmark.py'),
               Path(__file__).with_name('benchmark_qualified_operator_cases_v3.py'), Path(__file__).with_name('gpu_isolation.py'),
               Path(__file__).with_name('run_isolated_frozen_benchmark.py'),
               Path(__file__).with_name('benchmark_qualified_operator_cases.py'),
               Path(__file__).with_name('operator_performance_operands.py'),
               Path(__file__).with_name('build_operator_performance_plans_v2.py')]
    pins = {str(p.resolve()): sha(p) for p in sources}
    archive = args.output_dir / 'sources'
    archive.mkdir()
    for p in sources:
        (archive / pins[str(p.resolve())]).write_bytes(p.read_bytes())
    report.update(source_sha256=pins, source_archive=str(archive.resolve()))
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        runtime = validate_inputs(args.freeze, args.replay_manifest, args.plan)
        initial = observe(args.gpu_uuid)
        if initial['gpu_name'] != runtime['gpu']:
            raise ValueError('GPU hardware family differs from frozen diagnostic runtime')
        with journal.open('x') as stream:
            def retain(value):
                nonlocal observations
                stream.write(json.dumps(value) + '\n')
                stream.flush()
                observations += 1
            retain(initial)
            if foreign_processes(initial):
                raise ValueError('Initial GPU is busy; no timing started')
            command = [sys.executable, '-m', 'scripts.benchmark_qualified_operator_cases_v3',
                       '--plan', str(args.plan.resolve()), '--output', str(raw.resolve()),
                       '--gpu-uuid', args.gpu_uuid]
            report.update(status='running', command=command,
                          started_at_utc=datetime.now(timezone.utc).isoformat())
            with (args.output_dir / 'worker.log').open('x') as log:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid)
                child = subprocess.Popen(command, env=env, cwd=Path(__file__).resolve().parents[1],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                report['benchmark_pid'] = child.pid
                summary_path.write_text(json.dumps(report, indent=2) + '\n')
                while True:
                    state = observe(args.gpu_uuid)
                    retain(state)
                    if foreign_processes(state, allowed_pids=[child.pid]):
                        raise ValueError('Foreign compute/graphics GPU activity observed')
                    code = child.poll()
                    if code is not None:
                        if code != 0:
                            raise RuntimeError(f'Operator timing worker exited {code}')
                        final = observe(args.gpu_uuid)
                        retain(final)
                        if foreign_processes(final, allowed_pids=[child.pid]):
                            raise ValueError('Foreign GPU activity at timing completion')
                        break
                    time.sleep(1)
        validate_inputs(args.freeze, args.replay_manifest, args.plan)
        if sha(args.freeze) != report['freeze_sha256'] or sha(args.plan) != report['plan_sha256']:
            raise ValueError('Freeze or plan changed during timing')
        if any(sha(path) != expected for path, expected in pins.items()):
            raise ValueError('Timing/isolation source changed')
        report.update(status='completed_operator_timing_isolation_verified_independent_audit_pending',
                      accepted=True, raw_sha256=sha(raw),
                      runtime_sha256=sha(Path(str(raw) + '.runtime.json')))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        report.update(observation_count=observations,
                      observation_journal_sha256=sha(journal) if journal.exists() else None,
                      child_exit_code=None if child is None else child.returncode,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        summary_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

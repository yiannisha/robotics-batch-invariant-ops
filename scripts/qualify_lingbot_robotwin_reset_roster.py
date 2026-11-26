"""Run and independently audit the full released task reset/replay roster.

This is model-free qualification, not a paired model experiment. Failed tasks
remain explicit and never count as qualified. Freeze helpers before launching so
later repository edits cannot silently change a running task's capture contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


HELPERS = (
    'qualify_robotwin_benchmark_replay.py', 'robotwin_exact_worker.py',
    'robotwin_task_state.py', 'probe_robotwin_exact_initialization.py',
    'qualify_simpler_fixed_action_replay.py', 'audit_robotwin_benchmark_replay.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acceptance(task, directory):
    audit = directory / 'independent_audit.json'
    proof = json.loads(audit.read_text())
    if not (proof['status'] == 'complete_fixed_action_replay_independently_verified'
            and proof['task'] == task and proof['repeats'] == 3
            and proof['initial_state_and_full_trajectories_exact']
            and proof['cpu_and_cuda_rng_verified']
            and proof['raw_frames_verified'] == proof['source_defined_task_state_frames_verified']
            and proof['raw_replays_sha256'] == sha(directory / 'replays.jsonl')):
        raise ValueError('Incomplete or foreign task replay proof: ' + task)
    return dict(directory=str(directory.resolve()), independent_audit_sha256=sha(audit),
                raw_replays_sha256=proof['raw_replays_sha256'], seed=proof['seed'],
                raw_frames_verified=proof['raw_frames_verified'],
                controls_per_repeat=proof['controls_per_repeat'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'python', 'source-episodes', 'task-roster', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-adjust-bottle', type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.task_roster.read_text())['prescribed_tasks']
    if len(tasks) != 50 or len(set(tasks)) != 50 or any(
            not re.fullmatch('[a-z0-9_]+', task) for task in tasks):
        raise ValueError('Expected the complete released 50-task roster')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    frozen = args.output / 'frozen_helpers'
    frozen.mkdir()
    for name in HELPERS:
        (frozen / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    pins = {str(path): sha(path) for path in frozen.iterdir()}
    controls_sha = sha(args.source_episodes)
    qualified, failed = {}, {}

    def guard():
        if sha(args.source_episodes) != controls_sha or any(
                sha(Path(path)) != expected for path, expected in pins.items()):
            raise ValueError('Pinned source or fixed controls changed')

    def save(status, current=None):
        record = dict(status=status, current_task=current, prescribed_tasks=tasks,
                      qualified_tasks=qualified, qualified_task_count=len(qualified),
                      failed_tasks=failed, missing_tasks=[t for t in tasks if t not in qualified],
                      full_released_roster_qualified=len(qualified) == 50,
                      helper_sha256=pins, source_episode_sha256=controls_sha,
                      scope='Three full fixed-action replays per task; model-free reset qualification.',
                      model_evaluated=False, full_model_campaign_complete=False)
        temp = args.output / 'qualification_union.json.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(args.output / 'qualification_union.json')

    save('running_model_free_reset_roster')
    for task in tasks:
        guard()
        directory = args.output / task
        save('running_model_free_reset_roster', task)
        with (args.output / (task + '.log')).open('xb') as log:
            try:
                if task == 'adjust_bottle' and args.reuse_adjust_bottle:
                    directory = args.reuse_adjust_bottle
                else:
                    command = [str(args.python), str(frozen / HELPERS[0]),
                               '--source', str(args.source), '--python', str(args.python),
                               '--source-episodes', str(args.source_episodes), '--output', str(directory),
                               '--task', task, '--seed-start', '100000', '--repeats', '3',
                               '--capture-source-task-state']
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([str(args.python), str(frozen / 'audit_robotwin_benchmark_replay.py'),
                                '--directory', str(directory), '--source-episodes', str(args.source_episodes)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                guard()
                qualified[task] = acceptance(task, directory)
                print('QUALIFIED ' + task, flush=True)
            except (subprocess.CalledProcessError, ValueError, KeyError, OSError) as error:
                guard()
                failed[task] = dict(error=repr(error), retained_directory=str(directory))
                print('FAILED_RETAINED ' + task + ': ' + repr(error), flush=True)
        save('running_model_free_reset_roster')
    save('complete_model_free_reset_roster' if len(qualified) == 50 else
         'incomplete_model_free_reset_roster_failed_tasks_retained')


if __name__ == '__main__':
    main()

"""Version 2: reuse only byte-bound strong reports when upgrading frozen helpers.

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

if __package__:
    from .lingbot_rigid_qualification import CONTRACT, require_rigid_replay_proof
else:
    from lingbot_rigid_qualification import CONTRACT, require_rigid_replay_proof


HELPERS = (
    'qualify_robotwin_benchmark_replay.py', 'robotwin_exact_worker.py',
    'robotwin_task_state.py', 'probe_robotwin_exact_initialization.py',
    'qualify_simpler_fixed_action_replay.py', 'audit_robotwin_benchmark_replay.py',
    'lingbot_rigid_qualification.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acceptance(task, directory):
    audit = directory / 'independent_audit.json'
    proof = json.loads(audit.read_text())
    require_rigid_replay_proof(proof)
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


def reusable_entries(record, tasks, controls_sha):
    if record.get('physics_capture_contract') != CONTRACT or record['prescribed_tasks'] != tasks:
        raise ValueError('Wrong reusable physics contract or full task roster')
    entries = {}
    for task, entry in record['qualified_tasks'].items():
        if task not in tasks:
            raise ValueError('Foreign reusable task')
        directory = Path(entry['directory'])
        checked = acceptance(task, directory)
        if any(entry.get(k) != checked[k] for k in ('independent_audit_sha256','raw_replays_sha256')):
            raise ValueError('Reusable report or replay bytes changed')
        runtime = json.loads((directory/'runtime.json').read_text())
        if not (runtime['source_episode_sha256'] == controls_sha and runtime['seed_start'] == 100000
                and runtime['capture_source_task_state'] and runtime['capture_cuda_rng']):
            raise ValueError('Reusable control/capture contract differs')
        for source, identity in runtime['source_sha256'].items():
            if sha(Path(source)) != identity or sha(directory/Path(source).name) != identity:
                raise ValueError('Reusable execution source changed')
        helper_root = Path(next(iter(runtime['source_sha256']))).parent
        proof = json.loads((directory/'independent_audit.json').read_text())
        if sha(helper_root/'audit_robotwin_benchmark_replay.py') != proof['auditor_sha256']:
            raise ValueError('Reusable auditor source differs')
        entries[task] = directory
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'python', 'source-episodes', 'task-roster', 'output', 'helper-directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-qualification-union', type=Path)
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
        (frozen / name).write_bytes((args.helper_directory/name).read_bytes())
    pins = {str(path): sha(path) for path in frozen.iterdir()}
    controls_sha = sha(args.source_episodes)
    reused = {}
    if args.reuse_qualification_union:
        payload = args.reuse_qualification_union.read_bytes()
        (args.output/'reused_qualification_union.json').write_bytes(payload)
        reused = reusable_entries(json.loads(payload), tasks, controls_sha)
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
                      physics_capture_contract=CONTRACT,
                      reused_qualification_union_sha256=(sha(args.output/'reused_qualification_union.json') if reused else None),
                      reused_task_names=sorted(reused),
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
                if task in reused:
                    directory = reused[task]
                else:
                    command = [str(args.python), str(frozen / HELPERS[0]),
                               '--source', str(args.source), '--python', str(args.python),
                               '--source-episodes', str(args.source_episodes), '--output', str(directory),
                               '--task', task, '--seed-start', '100000', '--repeats', '3',
                               '--capture-source-task-state']
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                if task not in reused:
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

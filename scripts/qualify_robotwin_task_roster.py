"""Run model-free exact-reset/replay qualification across released clean tasks.

Every task uses its first expert-eligible released seed. The retained action
sequence is a stimulus, not an evaluation policy or a model success-rate study.
Failures are retained and do not remove tasks from the prescribed roster.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','launcher','python','source-episodes','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tasks',nargs='+',help='Optional subset; default is the complete released 50-task roster')
    args=p.parse_args()
    match=re.search(r'task_list_all=\(([^\n]+)\)',args.launcher.read_text())
    if match is None:raise ValueError('Released task roster not found')
    roster=re.findall(r'"([a-z0-9_]+)"',match.group(1))
    if len(roster)!=50 or len(set(roster))!=50:raise ValueError('Released roster changed')
    tasks=args.tasks or roster
    if len(tasks)!=len(set(tasks)) or not set(tasks)<=set(roster):raise ValueError('Invalid task subset')
    args.output.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent
    files=[Path(__file__).resolve(),args.launcher.resolve()]+[here/name for name in (
        'qualify_robotwin_benchmark_replay.py','audit_robotwin_benchmark_replay.py',
        'robotwin_exact_worker.py','robotwin_task_state.py','probe_robotwin_exact_initialization.py',
        'qualify_simpler_fixed_action_replay.py')]
    files.extend(args.source/'envs'/f'{task}.py' for task in tasks)
    hashes={str(path):sha(path) for path in files}
    snapshots=args.output/'source_snapshots';snapshots.mkdir()
    for path,expected in hashes.items():(snapshots/expected).write_bytes(Path(path).read_bytes())
    runtime=dict(scope='model_free_released_roster_reset_and_fixed_action_qualification',
        released_roster=roster,tasks=tasks,seed_start=100000,instruction_candidates=100,repeats=3,
        source_action_episode_sha256=sha(args.source_episodes),source_sha256=hashes,
        model_evaluated=False,paired_success_rate_campaign_complete=False)
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    records=[]
    def save():
        report=dict(tasks=records,prescribed_tasks=tasks,
            all_prescribed_tasks_qualified=len(records)==len(tasks) and all(row['independent_audit_passed'] for row in records),
            full_released_roster_qualified=tasks==roster and len(records)==50 and all(row['independent_audit_passed'] for row in records),
            model_evaluated=False,paired_success_rate_campaign_complete=False)
        temporary=args.output/'summary.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n');os.replace(temporary,args.output/'summary.json')
    for task in tasks:
        for path,expected in hashes.items():
            if sha(Path(path))!=expected:raise RuntimeError(f'Qualification source changed: {path}')
        folder=args.output/task
        command=[sys.executable,str(here/'qualify_robotwin_benchmark_replay.py'),
            '--source',str(args.source),'--python',str(args.python),'--source-episodes',str(args.source_episodes),
            '--output',str(folder),'--task',task,'--capture-source-task-state']
        with (args.output/f'{task}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            audit=None
            if result.returncode==0:
                audit=subprocess.run([str(args.python),str(here/'audit_robotwin_benchmark_replay.py'),
                    '--directory',str(folder),'--source-episodes',str(args.source_episodes)],stdout=log,stderr=subprocess.STDOUT)
        row=dict(task=task,replay_exit_code=result.returncode,audit_exit_code=None if audit is None else audit.returncode,
                 independent_audit_passed=audit is not None and audit.returncode==0,directory=str(folder))
        records.append(row);save();print(json.dumps(row),flush=True)
    save()
    if not all(row['independent_audit_passed'] for row in records):raise SystemExit(1)


if __name__=='__main__':main()

"""Qualify released evaluation seeds using retained nontrivial policy controls.

This is model-free simulator replay. Seed0 was an engine pilot; the released
evaluator begins at100000 for evaluator seed0 and uses100 instruction candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','python','source-episodes','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task',default='adjust_bottle');p.add_argument('--seed-start',type=int,default=100000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--capture-source-task-state',action='store_true')
    args=p.parse_args()
    if args.task!='adjust_bottle' and not args.capture_source_task_state:
        raise ValueError('Other tasks require source-defined task-state capture and new replay qualification')
    if args.repeats<2:raise ValueError('Need repeated complete replays')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source=next(json.loads(line) for line in args.source_episodes.read_text().splitlines() if json.loads(line)['condition']=='native_singleton')
    actions=np.asarray([a for c in source['calls'] for a in c['executed_actions']],dtype=np.float32)
    if actions.ndim!=2 or actions.shape[1]!=14 or not np.isfinite(actions).all():raise ValueError('Invalid retained controls')
    worker=Path(__file__).with_name('robotwin_exact_worker.py')
    source_files=[Path(__file__),worker,worker.with_name('probe_robotwin_exact_initialization.py')]
    if args.capture_source_task_state:source_files.append(worker.with_name('robotwin_task_state.py'))
    runtime=dict(scope='released_seed_fixed_action_complete_replay_qualification',model_evaluated=False,
        task=args.task,seed_start=args.seed_start,instruction_pool_limit=100,repeats=args.repeats,
        control_contract='retained seed0 native policy controls then hold final command until task success or horizon; no policy quality claim',
        source_episode_sha256=sha(args.source_episodes),source_action_bytes_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        source_action_count=len(actions),source_sha256={str(f):sha(f) for f in source_files},
        capture_cuda_rng=True,capture_source_task_state=args.capture_source_task_state,paired_success_rate_campaign_complete=False)
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    for f in source_files:(args.output/f.name).write_bytes(f.read_bytes())
    stderr=(args.output/'worker-stderr.log').open('w');logs=(args.output/'worker-stdout.log').open('w')
    worker_command=[str(args.python),str(worker),'--source',str(args.source),'--output',str(args.output/'frames'),'--capture-cuda-rng']
    if args.capture_source_task_state:worker_command.append('--capture-source-task-state')
    proc=subprocess.Popen(worker_command,
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
    def rpc(req):
        proc.stdin.write(json.dumps(req)+'\n');proc.stdin.flush()
        for line in proc.stdout:
            if 'ROBOTWIN_RESULT ' in line:return json.loads(line.split('ROBOTWIN_RESULT ',1)[1])
            logs.write(line);logs.flush()
        raise RuntimeError('Worker terminated')
    def state(frame):return {k:v for k,v in frame.items() if k!='path'}
    try:
        selection=None
        for seed in range(args.seed_start,args.seed_start+100):
            request=dict(op='expert',task=args.task,seed=seed,test_num=100,instruction_type='unseen')
            response=rpc(request)
            with (args.output/'expert_seeds.jsonl').open('a') as out:out.write(json.dumps(dict(request=request,response=response))+'\n')
            if not response['ok']:
                if response.get('error_type')=='UnStableError':continue
                raise RuntimeError(response['error'])
            if response['result']['eligible']:selection=response['result'];break
        if selection is None:raise RuntimeError('No eligible seed in100 attempts; retain evidence')
        (args.output/'selected_start.json').write_text(json.dumps(dict(seed=seed,selection=selection),indent=2)+'\n')
        trajectories=[]
        for repeat in range(args.repeats):
            reset=dict(request,op='reset',episode_info=selection['episode_info'],instruction=selection['instruction'])
            response=rpc(reset)
            if not response['ok']:raise RuntimeError(response['error'])
            frame=response['result']
            if state(frame)!=state(selection['canonical']):raise RuntimeError('Canonical physics/controller/task/CPU/CUDA RNG/observation reset differs')
            trajectory=[state(frame)];paths=[frame['path']];executed=[]
            while not frame['controller']['eval_success'] and frame['controller']['take_action_count']<frame['controller']['step_limit']:
                action=actions[min(len(executed),len(actions)-1)]
                response=rpc(dict(op='step',action=action.tolist()))
                if not response['ok']:raise RuntimeError(response['error'])
                frame=response['result'];trajectory.append(state(frame));paths.append(frame['path']);executed.append(action.tolist())
            record=dict(repeat=repeat,seed=seed,trajectory=trajectory,raw_frame_paths=paths,actions=executed)
            with (args.output/'replays.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
            trajectories.append(trajectory)
            print(f'REPLAY {repeat}: seed={seed}, controls={len(executed)}, success={frame["controller"]["eval_success"]}',flush=True)
        exact=all(x==trajectories[0] for x in trajectories)
        (args.output/'summary.json').write_text(json.dumps(dict(scope=runtime['scope'],seed=seed,repeats=args.repeats,
            complete_fixed_action_replays_exact=exact,cpu_cuda_rng_included=True,model_evaluated=False,paired_success_rate_campaign_complete=False),indent=2)+'\n')
        if not exact:raise RuntimeError('Benchmark-seed replay is not exactly repeatable')
    finally:
        proc.stdin.close();proc.wait();stderr.close();logs.close()


if __name__=='__main__':main()

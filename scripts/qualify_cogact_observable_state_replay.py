"""Qualify expanded SIMPLER snapshots against the original pi-zero replays."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from scripts.cogact_simpler_state_worker import RemoteStateEnv
from scripts.qualify_simpler_fixed_action_replay import digest


TASKS=['pick_can','move_near','open_drawer','close_drawer']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pizero-project','simpler-root','simpler-python','vulkan-icd','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    tree=ast.parse((args.pizero_project/'experiments/prepare_replay.py').read_text())
    mapping=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASKS' for t in node.targets))
    baseline_runtime=json.loads((args.baseline/'runtime.json').read_text())
    for name,expected in baseline_runtime['source_sha256'].items():
        if sha(args.pizero_project/'experiments'/name)!=expected:raise ValueError('Original pi-zero worker changed since baseline qualification')
    baseline={(row['task'],row['initialization'],row['repeat']):row for row in [json.loads(line) for line in (args.baseline/'replays.jsonl').read_text().splitlines()]}
    expected={(task,i,r) for task in TASKS for i in range(3) for r in range(3)}
    if set(baseline)!=expected:raise ValueError('Expected full original 36-replay cohort')
    files=[Path(__file__).resolve(),Path(__file__).with_name('cogact_simpler_state_worker.py').resolve(),Path(__file__).with_name('qualify_simpler_fixed_action_replay.py').resolve(),args.pizero_project/'experiments/simpler_worker.py']
    runtime=dict(scope='expanded_SIMPLER_observable_snapshot_and_full_replay_qualification',model_evaluated=False,
        tasks=TASKS,initializations=3,repeats=3,max_steps=200,baseline_replays_sha256=sha(args.baseline/'replays.jsonl'),
        reset_protocol='fresh_seeded_worker_reconstruction_exact_canonical_snapshot_comparison',
        original_worker_fidelity_role='separate_protocol_change_measurement_not_repeat_acceptance',
        source_sha256={str(f):sha(f) for f in files})
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    for file in files:(args.output/file.name).write_bytes(file.read_bytes())
    (args.output/'frames').mkdir();blocks=[];fidelity_records=[]
    for task in TASKS:
        for initial in range(3):
            canonical=None;reference=None
            for repeat in range(3):
                env=RemoteStateEnv(mapping[task],args.simpler_python,args.simpler_root,args.pizero_project/'experiments/simpler_worker.py',args.vulkan_icd,
                    args.output/f'{task}-{initial}-{repeat}-worker.log',process_seed=20261108+initial)
                try:
                    env.request(dict(command='reset',seed=20261108+initial,options={'obj_init_options':{'episode_id':initial}}))
                    captured=env.request(dict(command='capture_snapshot'));observable=captured['observable']
                    if canonical is None:canonical=observable['hashes']
                    if canonical!=observable['hashes']:raise RuntimeError('Cold reconstructed initial snapshot differs')
                    frames=[];steps=[dict(observation_sha256=digest(observable['raw']['observation']),instruction=observable['raw']['instruction'])]
                    def retain(observable):
                        raw=observable['raw'];identity=digest(raw);path=args.output/'frames'/(identity+'.pt.gz')
                        if not path.exists():
                            buf=io.BytesIO();torch.save(raw,buf);path.write_bytes(gzip.compress(buf.getvalue(),compresslevel=1,mtime=0))
                        frames.append(dict(observable['hashes'],raw_frame=path.name))
                    retain(observable)
                    for step in range(200):
                        action=np.array([0.002*np.sin(step/7),0.002*np.cos(step/9),0.001*np.sin(step/11),0.,0.,0.,-1. if (step//20)%2 else 1.],dtype=np.float64)
                        result=env.request(dict(command='step_with_state',action=action.tolist()));retain(result['observable'])
                        steps.append(dict(step=step,action=action.tolist(),observation_sha256=digest(result['observation']),reward=float(result['reward']),
                            success=bool(result['success']),truncated=bool(result['truncated']),info=result['info'],instruction=result['observable']['raw']['instruction']))
                        if result['truncated']:break
                    original_steps=baseline[(task,initial,repeat)]['steps']
                    original_exact=steps==original_steps
                    first_difference=next((i for i,(a,b) in enumerate(zip(steps,original_steps)) if a!=b),None)
                    if first_difference is None and len(steps)!=len(original_steps):first_difference=min(len(steps),len(original_steps))
                    fidelity=dict(exact=original_exact,initial_observation_exact=steps[0]['observation_sha256']==original_steps[0]['observation_sha256'],
                        first_differing_record=first_difference,
                        changed_fields=[] if first_difference is None or first_difference>=min(len(steps),len(original_steps)) else [k for k in steps[first_difference] if steps[first_difference].get(k)!=original_steps[first_difference].get(k)])
                    current=digest(dict(frames=frames,steps=steps))
                    row=dict(task=task,initialization=initial,repeat=repeat,canonical=canonical,
                        expanded_state_exact=True,reset_protocol=runtime['reset_protocol'],trajectory=frames,steps=steps,
                        original_pizero_replay_exact=original_exact,original_worker_fidelity=fidelity)
                    with (args.output/'replays.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
                    if reference is not None and reference!=current:raise RuntimeError('Expanded complete replay differs')
                    reference=current;fidelity_records.append(dict(task=task,initialization=initial,repeat=repeat,**fidelity))
                    print(f'EXACT {task}/{initial}/{repeat}: {len(steps)-1} controls; original trajectory exact={original_exact}',flush=True)
                finally:env.close()
            blocks.append(dict(task=task,initialization=initial,repeats=3,exact=True))
    (args.output/'summary.json').write_text(json.dumps(dict(blocks=blocks,original_pizero_replays_exact=all(r['exact'] for r in fidelity_records),
        original_worker_fidelity=fidelity_records,expanded_observable_replays_exact=True,
        independent_audit_pending=True,model_evaluated=False),indent=2)+'\n')


if __name__=='__main__':main()

"""Independently verify benchmark-seed fixed controls and every saved raw frame."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from qualify_simpler_fixed_action_replay import digest


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--source-episodes',type=Path,required=True);args=p.parse_args()
    d=args.directory;runtime=json.loads((d/'runtime.json').read_text());selected=json.loads((d/'selected_start.json').read_text())
    require(runtime['seed_start']==100000 and runtime['instruction_pool_limit']==100 and runtime['capture_cuda_rng'],'Released seed/language/RNG protocol differs')
    require(sha(args.source_episodes)==runtime['source_episode_sha256'],'Retained controls changed')
    for path,expected in runtime['source_sha256'].items():require(sha(d/Path(path).name)==expected,'Archived execution source changed')
    source=next(json.loads(line) for line in args.source_episodes.read_text().splitlines() if json.loads(line)['condition']=='native_singleton')
    actions=np.asarray([a for c in source['calls'] for a in c['executed_actions']],dtype=np.float32)
    require(hashlib.sha256(actions.tobytes()).hexdigest()==runtime['source_action_bytes_sha256'],'Control bytes differ')
    candidates=[json.loads(line) for line in (d/'expert_seeds.jsonl').read_text().splitlines()]
    require([r['request']['seed'] for r in candidates]==list(range(100000,selected['seed']+1)),'Missing/reordered eligibility candidates')
    for i,row in enumerate(candidates):
        request,response=row['request'],row['response']
        require(request['task']==runtime['task'] and request['test_num']==100 and request['instruction_type']=='unseen','Eligibility protocol differs')
        eligible=response['ok'] and response['result']['eligible']
        require(eligible==(i==len(candidates)-1),'Did not choose first expert-eligible seed')
    canonical={k:v for k,v in selected['selection']['canonical'].items() if k!='path'}
    rows=[json.loads(line) for line in (d/'replays.jsonl').read_text().splitlines()]
    require(len(rows)==runtime['repeats'] and len(rows)>=2,'Missing repeats')
    frame_count=0
    expanded_task_frames=0
    for repeat,row in enumerate(rows):
        require(row['repeat']==repeat and row['seed']==selected['seed'] and row['trajectory'][0]==canonical,'Repeat seed/canonical state differs')
        controls=np.asarray(row['actions'],dtype=np.float32)
        expected=actions[np.minimum(np.arange(len(controls)),len(actions)-1)]
        require(controls.shape==expected.shape and controls.tobytes()==expected.tobytes(),'Executed controls differ from fixed source')
        trajectory=row['trajectory'];require(len(trajectory)==len(controls)+1==len(row['raw_frame_paths']),'Frame/control coverage differs')
        final=trajectory[-1]['controller']
        require(final['eval_success'] or final['take_action_count']==final['step_limit'],'Replay is incomplete')
        for step,(frame,path) in enumerate(zip(trajectory,row['raw_frame_paths'])):
            raw=torch.load(path,map_location='cpu',weights_only=False)
            for key,field in (('observation','observation_sha256'),('physics','physics_sha256'),('rng','rng_sha256'),('task_state','task_state_sha256')):
                require(digest(raw[key])==frame[field],f'Changed raw {key} at repeat{repeat}/step{step}')
            require(raw['controller']==frame['controller'] and raw['controller']['take_action_count']==step,'Raw controller/index differs')
            require('cuda' in raw['rng'] and 'cuda_initialized' in raw['rng'],'CUDA RNG state omitted')
            require(raw['controller']['instruction']==selected['selection']['instruction'],'Instruction differs')
            if runtime.get('capture_source_task_state'):
                task=raw['task_state']
                require(task.get('schema')=='source_defined_task_fields_v1','Missing expanded task state')
                task_source=Path(task['source'])
                require(sha(task_source)==task['source_sha256'],'Task source changed since capture')
                # Keep source bytes bound to the hash recorded by the worker.
                archive=d/'audited_task_sources'/task['source_sha256']
                if not archive.exists():
                    archive.parent.mkdir(exist_ok=True)
                    archive.write_bytes(task_source.read_bytes())
                require(sha(archive)==task['source_sha256'],'Archived task source changed')
                expected=set()
                for node in ast.walk(ast.parse(archive.read_text())):
                    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self' and isinstance(node.ctx,(ast.Store,ast.Del)):
                        expected.add(node.attr)
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('setattr','delattr') and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='self':
                        require(len(node.args)>1 and isinstance(node.args[1],ast.Constant) and isinstance(node.args[1].value,str),'Dynamic task mutation is uncovered')
                        expected.add(node.args[1].value)
                require(set(task['fields'])==set(task['inventory']['assigned'])==expected,'Task field coverage differs from source assignments')
                require(all(isinstance(v.get('present'),bool) and ('value' in v)==v['present'] for v in task['fields'].values()),'Invalid field presence evidence')
                expanded_task_frames+=1
            frame_count+=1
        require(digest(trajectory)==digest(rows[0]['trajectory']),'Repeated raw trajectory differs')
    report=dict(status='complete_fixed_action_replay_independently_verified',task=runtime['task'],seed=selected['seed'],
                repeats=len(rows),controls_per_repeat=len(rows[0]['actions']),raw_frames_verified=frame_count,
                initial_state_and_full_trajectories_exact=True,cpu_and_cuda_rng_verified=True,
                source_defined_task_state_frames_verified=expanded_task_frames,
                model_evaluated=False,full_50_task_qualification_complete=False,
                raw_replays_sha256=sha(d/'replays.jsonl'),auditor_sha256=sha(Path(__file__)))
    (d/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

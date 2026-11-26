"""Audit full SIMPLER state reconstruction and separate old-worker fidelity."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);args=p.parse_args()
    d=args.directory;runtime=json.loads((d/'runtime.json').read_text())
    require(runtime['reset_protocol']=='fresh_seeded_worker_reconstruction_exact_canonical_snapshot_comparison','Unexpected reset protocol')
    require(sha(args.baseline/'replays.jsonl')==runtime['baseline_replays_sha256'],'Original worker records changed')
    for path,expected in runtime['source_sha256'].items():require(sha(d/Path(path).name)==expected,'Archived producer source changed')
    rows=[json.loads(line) for line in (d/'replays.jsonl').read_text().splitlines()]
    original={(r['task'],r['initialization'],r['repeat']):r for r in [json.loads(line) for line in (args.baseline/'replays.jsonl').read_text().splitlines()]}
    expected={(task,i,r) for task in runtime['tasks'] for i in range(3) for r in range(3)}
    require(set(runtime['tasks'])=={'pick_can','move_near','open_drawer','close_drawer'},'Wrong task roster')
    require(len(rows)==len(expected)==36 and {(r['task'],r['initialization'],r['repeat']) for r in rows}==expected,'Missing/duplicate replays')
    raw_cache={};references={};source_cache={};task_field_cache={};frames=0;fidelity=[]
    def verify_source(path,expected):
        key=(path,expected)
        if key not in source_cache:
            content=Path(path).read_bytes();require(hashlib.sha256(content).hexdigest()==expected,'Simulator source changed')
            archive=d/'audited_simulator_sources'/expected;archive.parent.mkdir(exist_ok=True);archive.write_bytes(content);source_cache[key]=content
        return source_cache[key]
    for row in rows:
        task,initial,repeat=row['task'],row['initialization'],row['repeat'];trajectory=row['trajectory'];steps=row['steps']
        require(row['reset_protocol']==runtime['reset_protocol'] and row['expanded_state_exact'],'Reset protocol differs')
        require(len(trajectory)==len(steps) and 1<len(steps)<=201,'Frame/action coverage differs')
        require(len(steps)==201 or steps[-1]['truncated'],'Incomplete fixed-action replay')
        require({k:v for k,v in trajectory[0].items() if k!='raw_frame'}==row['canonical'],'Initial state differs from canonical snapshot')
        signature=digest(dict(trajectory=trajectory,steps=steps));key=(task,initial)
        if key in references:require(references[key]==signature,'Repeated complete observable trajectories differ')
        else:references[key]=signature
        for i,step in enumerate(steps[1:]):
            action=np.array([0.002*np.sin(i/7),0.002*np.cos(i/9),0.001*np.sin(i/11),0.,0.,0.,-1. if (i//20)%2 else 1.],dtype=np.float64)
            require(step['step']==i and np.asarray(step['action'],dtype=np.float64).tobytes()==action.tobytes(),'Fixed control bytes differ')
        for index,frame in enumerate(trajectory):
            path=(d/'frames'/frame['raw_frame']).resolve();require(path.parent==(d/'frames').resolve(),'Raw frame escapes directory')
            if str(path) not in raw_cache:
                with gzip.open(path,'rb') as f:raw=torch.load(f,map_location='cpu',weights_only=False)
                require(digest(raw)==path.name.split('.')[0],'Raw content address differs')
                require(set(raw)=={'observation','scene_pack','physics','agent','controller','task','elapsed_steps','wrapper_elapsed_steps','rng','instruction'},'Missing observable category')
                require(np.isfinite(raw['physics']).all(),'Nonfinite observable physics')
                require(raw['rng']['torch_used_in_simulator'] is False and set(raw['rng'])=={'main','episode','numpy','python','torch_used_in_simulator'},'Simulator RNG coverage differs')
                task_state=raw['task'];task_sources=task_state['source_sha256'];source_key=tuple(sorted(task_sources.items()))
                if source_key not in task_field_cache:
                    fields=set()
                    for filename,expected_hash in task_sources.items():
                        content=verify_source(filename,expected_hash)
                        for node in ast.walk(ast.parse(content.decode())):
                            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self' and isinstance(node.ctx,(ast.Store,ast.Del)):fields.add(node.attr)
                            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('setattr','delattr') and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='self':
                                require(len(node.args)>1 and isinstance(node.args[1],ast.Constant) and isinstance(node.args[1].value,str),'Dynamic task mutation is uncovered');fields.add(node.args[1].value)
                    task_field_cache[source_key]=fields
                require(set(task_state['fields'])==task_field_cache[source_key],'Task field inventory differs')
                config=task_state['fields'].get('_agent_cfg',{}).get('value',{})
                if 'source' in config:verify_source(config['source'],config['source_sha256'])
                record={name+'_sha256':digest(value) for name,value in raw.items()}|{'raw_frame':frame['raw_frame']}
                raw_cache[str(path)]=(record,raw['elapsed_steps'],digest(raw['observation']),raw['instruction'])
            record,elapsed,observation_hash,instruction=raw_cache[str(path)]
            require(record==frame and elapsed==index,'Frame/index hash differs')
            require(steps[index]['observation_sha256']==observation_hash and steps[index]['instruction']==instruction,'Observation/instruction differs')
            frames+=1
        old=original[(task,initial,repeat)]['steps'];exact=steps==old
        first=next((i for i,(a,b) in enumerate(zip(steps,old)) if a!=b),None)
        if first is None and len(steps)!=len(old):first=min(len(steps),len(old))
        measured=dict(exact=exact,initial_observation_exact=steps[0]['observation_sha256']==old[0]['observation_sha256'],first_differing_record=first,
            changed_fields=[] if first is None or first>=min(len(steps),len(old)) else [k for k in steps[first] if steps[first].get(k)!=old[first].get(k)])
        require(row['original_worker_fidelity']==measured and row['original_pizero_replay_exact']==exact,'Original-worker fidelity measurement differs')
        fidelity.append(dict(task=task,initialization=initial,repeat=repeat,**measured))
    report=dict(status='complete_observable_state_fixed_replay_independently_verified',tasks=runtime['tasks'],initializations=3,repeats=3,
        episodes=len(rows),raw_frames_verified=frames,unique_raw_frames_verified=len(raw_cache),
        packed_scene_physics_controller_task_rng_observations_exact=True,reset_protocol=runtime['reset_protocol'],
        original_worker_exact_replays=sum(r['exact'] for r in fidelity),original_worker_nonexact_replays=sum(not r['exact'] for r in fidelity),
        original_worker_initial_observations_exact=sum(r['initial_observation_exact'] for r in fidelity),original_worker_fidelity=fidelity,
        model_evaluated=False,paired_success_rate_campaign_complete=False,raw_replays_sha256=sha(d/'replays.jsonl'),auditor_sha256=sha(Path(__file__)))
    (d/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

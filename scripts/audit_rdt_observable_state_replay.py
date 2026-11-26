"""Rehash every observable-state frame and verify prescribed replay controls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    d=args.directory;runtime=json.loads((d/'runtime.json').read_text())
    for path,expected in runtime['source_sha256'].items():
        require(hashlib.sha256((d/Path(path).name).read_bytes()).hexdigest()==expected,'Archived producer source changed')
    require(runtime['repeats']>=2 and runtime['max_steps']==400,'Insufficient replay schedule')
    rows=[json.loads(line) for line in (d/'replays.jsonl').read_text().splitlines()]
    expected={(task,initial,repeat) for task in runtime['tasks'] for initial in range(runtime['initializations']) for repeat in range(runtime['repeats'])}
    require(len(rows)==len(expected) and {(row['task'],row['initialization'],row['repeat']) for row in rows}==expected,'Incomplete or duplicate blocks')
    reference={};frames=0;raw_seen={};task_sources={}
    for row in rows:
        key=(row['task'],row['initialization']);trajectory=row['trajectory'];steps=row['steps']
        require(row['seed']==20241201+row['initialization'],'Seed schedule differs')
        require(len(trajectory)==len(steps)+1 and 0<len(steps)<=400,'Invalid trajectory coverage')
        require(len(steps)==400 or ((steps[-1]['terminated'] or steps[-1]['truncated']) and steps[-1]['success']),'Incomplete replay')
        current=digest(dict(trajectory=trajectory,steps=steps))
        if key in reference:require(reference[key]==current,'Repeated complete observable trajectory differs')
        else:reference[key]=current
        initial=torch.load(d/'frames'/trajectory[0]['raw_frame'],map_location='cpu',weights_only=False)
        start=initial['observation']['agent']['qpos'][0,:-2]
        for i,step in enumerate(steps):
            action=np.concatenate([start+0.01*np.sin(i/13),[1. if (i//20)%2 else -1.]]).astype(np.float32)
            require(np.asarray(step['action'],dtype=np.float32).tobytes()==action.tobytes(),'Fixed control bytes differ')
        for i,frame in enumerate(trajectory):
            path=(d/'frames'/frame['raw_frame']).resolve();require(path.parent==(d/'frames').resolve(),'Raw path escapes qualification')
            if str(path) not in raw_seen:
                raw=torch.load(path,map_location='cpu',weights_only=False)
                require(digest(raw)==path.stem,'Raw frame content address differs')
                require(set(raw)=={'observation','policy_rgb','physics','controller','task','drives','elapsed_steps','rng'},'Missing observable state category')
                require(set(frame)=={k+'_sha256' for k in raw}|{'raw_frame'},'Frame hash coverage differs')
                for name,value in raw.items():require(digest(value)==frame[name+'_sha256'],f'Raw {name} changed')
                require(int(np.asarray(raw['elapsed_steps']).reshape(-1)[0])==i,'Elapsed control count differs')
                require(raw['rng']['cuda_initialized'] and 'torch_cuda' in raw['rng'] and raw['rng']['environment'],'Incomplete RNG evidence')
                task=raw['task'];source=Path(task['source']);source_hash=task['source_sha256']
                if source_hash not in task_sources:
                    content=source.read_bytes();require(hashlib.sha256(content).hexdigest()==source_hash,'Task source changed')
                    expected_fields={n.attr for n in ast.walk(ast.parse(content.decode())) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and isinstance(n.ctx,(ast.Store,ast.Del))}
                    archive=d/'audited_task_sources'/source_hash;archive.parent.mkdir(exist_ok=True);archive.write_bytes(content)
                    task_sources[source_hash]=expected_fields
                require(set(task['fields'])==task_sources[source_hash],'Task field coverage differs from source')
                raw_seen[str(path)]=(dict(frame),i)
            require(raw_seen[str(path)]==(frame,i),'Repeated frame metadata or index differs from raw evidence')
            frames+=1
    report=dict(status='complete_observable_state_fixed_replay_independently_verified',tasks=runtime['tasks'],
        initializations=runtime['initializations'],repeats=runtime['repeats'],episodes=len(rows),raw_frames_verified=frames,
        unique_raw_frames_verified=len(raw_seen),physics_controller_task_drives_rng_observations_exact=True,
        model_evaluated=False,paired_success_rate_campaign_complete=False,
        raw_replays_sha256=hashlib.sha256((d/'replays.jsonl').read_bytes()).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Audit raw no-MSAA replay state and the retained failing-control regressions."""
import argparse
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--source-episodes',nargs='+',type=Path,required=True);args=p.parse_args();d=args.directory
    runtime=json.loads((d/'runtime.json').read_text())
    require(runtime['tasks']==list(range(10)) and runtime['initializations']==3 and runtime['repeats']>=3 and runtime['regression_repeats']>=5,'Insufficient prescribed coverage')
    require(runtime['offscreen_samples']==0 and runtime['renderer_protocol_changed'] and runtime['default_renderer_study_results_must_not_be_pooled'],'Renderer protocol missing')
    for filename,expected in runtime['source_sha256'].items():require(sha(d/Path(filename).name)==expected,'Archived producer source changed')
    sources={}
    for path in args.source_episodes:
        with gzip.open(path,'rt') as f:row=json.load(f)
        sources[(row['task'],row['initialization'])]=(sha(path),row)
    require(set(sources)=={(2,0),(2,1)},'Both failed native-dynamic control sequences are required')
    rows=[json.loads(line) for line in (d/'replays.jsonl').read_text().splitlines()]
    expected={('fixed_wave',task,i,r) for task in range(10) for i in range(3) for r in range(runtime['repeats'])}
    expected|={('retained_failing_controls',task,i,r) for task,i in sources for r in range(runtime['regression_repeats'])}
    require(len(rows)==len(expected) and {(r['kind'],r['task'],r['initialization'],r['repeat']) for r in rows}==expected,'Missing/duplicate replays')
    refs={};raw_cache={};frames=0;controls=0
    for row in rows:
        kind,task,initial=row['kind'],row['task'],row['initialization'];trajectory=row['trajectory'];steps=row['steps']
        require(len(trajectory)==len(steps)+1 and 0<len(steps)<=520,'Incomplete frame/control coverage')
        signature=digest(dict(trajectory=trajectory,steps=steps));key=(kind,task,initial)
        if key in refs:require(refs[key]==signature,'Repeated complete state/observation trajectory differs')
        else:refs[key]=signature
        if kind=='fixed_wave':
            require(len(steps)==520 or steps[-1]['terminated'] or steps[-1]['truncated'],'Short nonterminal fixed replay')
            for i,step in enumerate(steps):
                action=np.array([0.01*np.sin(i/7),0.01*np.cos(i/9),0.01*np.sin(i/11),0.,0.,0.,-1. if (i//20)%2 else 1.],dtype=np.float32)
                require(np.asarray(step['action'],dtype=np.float32).tobytes()==action.tobytes(),'Fixed waveform changed')
        else:
            expected_sha,old=sources[(task,initial)]
            require(row['source_episode_sha256']==expected_sha and row['original_physics_exact'],'Retained source/physics declaration differs')
            require(len(steps)==len(old['steps']) and np.asarray([s['action'] for s in steps],dtype=np.float32).tobytes()==np.asarray([s['action'] for s in old['steps']],dtype=np.float32).tobytes(),'Retained control bytes differ')
            require([t['sim_state_sha256'] for t in trajectory]==[t['sim_state_sha256'] for t in old['trajectory']],'Renderer change alters retained-control physics')
        for frame in trajectory:
            path=(d/'frames'/frame['raw_frame']).resolve();require(path.parent==(d/'frames').resolve(),'Raw path escapes qualification')
            if str(path) not in raw_cache:
                with gzip.open(path,'rb') as f:raw=torch.load(f,map_location='cpu',weights_only=False)
                require(digest(raw)==path.name.split('.')[0],'Raw content address changed')
                require(set(raw)=={'physical','controller','actuator_controls','rng','observation','derived'},'Missing raw observable state')
                measured={field:digest(raw[name]) for name,field in [('physical','sim_state_sha256'),('controller','controller_state_sha256'),
                    ('actuator_controls','actuator_control_sha256'),('rng','rng_state_sha256'),('observation','observation_sha256'),('derived','derived_sha256')]}
                require(np.isfinite(raw['physical']).all() and raw['rng']['cuda_initialized'] and 'torch_cuda' in raw['rng'],'Invalid physics/RNG evidence')
                raw_cache[str(path)]=(measured,raw['physical'].tolist(),np.asarray(raw['observation']['robot_state']['eef']['pos']).tolist(),np.asarray(raw['observation']['robot_state']['gripper']['qpos']).tolist())
            measured,physical,eef,gripper=raw_cache[str(path)]
            require(all(frame[field]==value for field,value in measured.items()),'Raw snapshot hash differs')
            require(frame['sim_state']==physical and frame['eef']==eef and frame['gripper']==gripper,'Raw numeric state differs')
            frames+=1
        controls+=len(steps)
    report=dict(status='complete_no_msaa_observable_replay_independently_verified',tasks=list(range(10)),initializations=3,
        repeats=runtime['repeats'],regression_repeats=runtime['regression_repeats'],episodes=len(rows),raw_frames_verified=frames,
        unique_raw_frames_verified=len(raw_cache),executed_controls_verified=controls,
        all_observable_state_and_rgb_repeats_exact=True,retained_failing_control_physics_unchanged=True,
        offscreen_samples=0,renderer_protocol_changed=True,default_renderer_study_results_must_not_be_pooled=True,
        model_evaluated=False,paired_success_rate_campaign_complete=False,raw_replays_sha256=sha(d/'replays.jsonl'),auditor_sha256=sha(Path(__file__)))
    (d/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

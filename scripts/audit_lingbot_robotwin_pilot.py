"""Independently check raw LingBot pilot frames, actions, noise and attribution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from qualify_simpler_fixed_action_replay import digest


def require(value, message):
    if not value:raise ValueError(message)


def action_bytes(value):return np.asarray(value,dtype=np.float32).tobytes()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    d=args.directory;runtime=json.loads((d/'runtime.json').read_text())
    rows=[json.loads(line) for line in (d/'episodes.jsonl').read_text().splitlines()]
    expected={base+suffix for base in ('native_singleton','native_dynamic','patched_singleton','patched_dynamic') for suffix in ('','_repeat')}
    require(len(rows)==8 and {r['condition'] for r in rows}==expected,'Missing, duplicate or foreign condition')
    episodes={r['condition']:r for r in rows};canonical={k:v for k,v in runtime['canonical_state'].items() if k!='path'}
    for name,episode in episodes.items():
        require(episode['task']==runtime['task'] and episode['seed']==runtime['seed'],'Task/start differs')
        trajectory=episode['trajectory'];calls=episode['calls'];paths=episode['raw_frame_paths']
        require(trajectory[0]==canonical and len(paths)==len(trajectory),'Canonical reset/frame cardinality differs')
        cursor=0;executed=[]
        for i,call in enumerate(calls):
            require(call['policy_call']==i,'Noncontiguous call index')
            size=[1,2,4,8][i%4] if 'dynamic' in name else 1
            require(call['batch_size']==size and call['position']==i%size,'Incorrect batch arrangement')
            require(len(call['companion_indices'])==size-1,'Companion cardinality differs')
            noise_id=f'{runtime["task"]}/{runtime["seed"]}/policy-call-{i}'
            seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
            noise=torch.randn(50,55,generator=torch.Generator().manual_seed(seed))
            metadata=dict(dtype=str(noise.dtype),shape=list(noise.shape),sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest())
            require(call['noise_id']==noise_id and call['noise_seed']==seed and call['model_inputs']['noise']==metadata,'Noise is not request-indexed')
            require(digest(call['model_inputs'])==call['target_input_sha256'],'Input fingerprint corrupt')
            chunk=np.asarray(call['physical_chunk'],dtype=np.float32)
            normalized=np.asarray(call['normalized_actions'],dtype=np.float32)
            actions=np.asarray(call['executed_actions'],dtype=np.float32)
            require(chunk.shape==(50,14) and normalized.shape==(50,55) and np.isfinite(chunk).all() and np.isfinite(normalized).all(),'Invalid model chunk')
            require(actions.ndim==2 and actions.shape[1]==14 and 0<len(actions)<=50 and actions.tobytes()==chunk[:len(actions)].tobytes(),'Executed controls do not match released chunk prefix')
            if i<len(calls)-1:require(len(actions)==50,'Early chunk truncation')
            cursor+=len(actions);executed.extend(call['executed_actions'])
            require(cursor<len(trajectory),'Action/frame cardinality differs')
        require(cursor+1==len(trajectory),'Extra frames/actions')
        require(episode['success']==trajectory[-1]['controller']['eval_success'],'Outcome differs from simulator record')
        require(episode['success'] or cursor==trajectory[-1]['controller']['step_limit'],'Incomplete episode')
        for step,(frame,path) in enumerate(zip(trajectory,paths)):
            raw=torch.load(path,map_location='cpu',weights_only=False)
            for raw_key,hash_key in (('observation','observation_sha256'),('physics','physics_sha256'),('rng','rng_sha256'),('task_state','task_state_sha256')):
                require(digest(raw[raw_key])==frame[hash_key],f'Raw {raw_key} changed at {name}/{step}')
            require(raw['controller']==frame['controller'] and raw['controller']['take_action_count']==step,'Raw controller/step differs')
            require(raw['controller']['instruction']==runtime['instruction'],'Instruction changes')
        episode['_actions']=executed
    for name in ('native_singleton','native_dynamic','patched_singleton','patched_dynamic'):
        for field in ('calls','trajectory','success'):
            # Hash serialization also distinguishes signed zero in float lists.
            require(digest(episodes[name][field])==digest(episodes[name+'_repeat'][field]),f'A/A differs: {name}/{field}')
    left,right=episodes['patched_singleton'],episodes['patched_dynamic']
    require(digest(left['trajectory'])==digest(right['trajectory']) and action_bytes(left['_actions'])==action_bytes(right['_actions']),'Invariant trajectories/actions differ')
    require(len(left['calls'])==len(right['calls']),'Invariant call count differs')
    for a,b in zip(left['calls'],right['calls']):
        for field in ('normalized_actions','physical_chunk','model_inputs'):
            require(digest(a[field])==digest(b[field]),f'Invariant {field} differs')
    left,right=episodes['native_singleton'],episodes['native_dynamic']
    first=next((i for i,(a,b) in enumerate(zip(left['_actions'],right['_actions'])) if action_bytes(a)!=action_bytes(b)),None)
    if first is not None:
        call=first//50
        require(left['calls'][call]['target_input_sha256']==right['calls'][call]['target_input_sha256'],'First native action mismatch has different actual input')
        require(digest(left['trajectory'][:first+1])==digest(right['trajectory'][:first+1]),'Native divergence has differing prior trajectory')
    report=dict(status='pilot_raw_evidence_independently_verified',scope='complete_episode_pilot',episodes=8,
                all_AA_exact=True,invariant_batching_exact=True,raw_frames_rehashed=True,request_noise_regenerated=True,
                native_first_action_divergence=first,success={name:r['success'] for name,r in episodes.items()},
                episode_bytes_sha256=hashlib.sha256((d/'episodes.jsonl').read_bytes()).hexdigest(),
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),paired_success_rate_campaign_complete=False)
    (d/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Measure both arms in an accepted eight-episode LingBot pilot.

This is explicitly a one-start pilot measurement, never full-campaign evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest
from scripts.summarize_paired_stability import numerical_pair


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dual_arm_poses(raw):
    value=raw['observation']['endpose']
    poses=np.asarray([value['left_endpose'],value['right_endpose']],dtype=np.float64)
    require(poses.shape==(2,7) and np.isfinite(poses).all() and (np.linalg.norm(poses[:,3:],axis=-1)>0).all(),'Invalid dual-arm poses')
    return poses


def dual_arm_drift(left,right):
    n=min(len(left),len(right));require(n>0,'Missing common trajectory')
    left=np.asarray(left[:n],dtype=np.float64);right=np.asarray(right[:n],dtype=np.float64)
    require(left.shape==right.shape and left.shape[1:]==(2,7),'Dual-arm geometry differs')
    rows={}
    for arm,name in enumerate(('left','right')):
        x,y=left[:,arm],right[:,arm];position=np.linalg.norm(x[:,:3]-y[:,:3],axis=1)
        q=x[:,3:]/np.linalg.norm(x[:,3:],axis=1,keepdims=True);r=y[:,3:]/np.linalg.norm(y[:,3:],axis=1,keepdims=True)
        r=np.where((np.sum(q*r,axis=1)<0)[:,None],-r,r)
        angle=4*np.arctan2(np.linalg.norm(q-r,axis=1),np.linalg.norm(q+r,axis=1))
        rows[name]=dict(position_max_m=float(position.max()),position_rms_m=float(np.sqrt(np.mean(position**2))),
            position_last_common_m=float(position[-1]),rotation_max_rad=float(angle.max()),rotation_rms_rad=float(np.sqrt(np.mean(angle**2))))
    return dict(common_frames=n,arms=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot-directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Preserve prior reports');d=a.pilot_directory;source=d/'episodes.jsonl';proof_path=d/'independent_audit.json';proof=json.loads(proof_path.read_text())
    require(proof['status']=='pilot_raw_evidence_independently_verified' and proof['all_AA_exact'] and proof['invariant_batching_exact'] and proof['episode_bytes_sha256']==sha(source),'Accepted pilot evidence missing/changed')
    episodes={r['condition']:r for r in map(json.loads,source.read_text().splitlines())};require(len(episodes)==8,'Eight pilot episodes required')
    runtime=json.loads((d/'runtime.json').read_text());cache={};raw_sha={};rows={}
    for mode,prefix in [('native','native'),('invariant','patched')]:
        left=episodes[prefix+'_singleton'];right=episodes[prefix+'_dynamic']
        for name in (prefix+'_singleton',prefix+'_dynamic'):
            require(all(digest(episodes[name][key])==digest(episodes[name+'_repeat'][key]) for key in ('calls','trajectory','success')),'Repeat differs')
        require(left['trajectory'][0]==right['trajectory'][0],'Canonical starts differ')
        def poses(episode):
            result=[]
            for frame,path in zip(episode['trajectory'],episode['raw_frame_paths']):
                path=Path(path).resolve();require(path.is_relative_to(d.resolve()),'Raw frame escapes pilot')
                if path not in cache:
                    raw=torch.load(path,map_location='cpu',weights_only=False)
                    require(all(digest(raw[key])==frame[key+'_sha256'] for key in ('observation','physics','rng','task_state')) and raw['controller']==frame['controller'],'Raw state/observation bytes differ')
                    cache[path]=dual_arm_poses(raw);raw_sha[str(path)]=sha(path)
                result.append(cache[path])
            require(len(result)==len(episode['trajectory']),'Missing raw frames');return result
        drift=dual_arm_drift(poses(left),poses(right));fixed=[]
        for l,r in zip(left['calls'],right['calls']):
            if l['target_input_sha256']!=r['target_input_sha256'] or r['batch_size']==1:continue
            require(l['noise_id']==r['noise_id'],'Indexed noise differs')
            pair=[]
            for call in (l,r):
                item=dict(call);item['normalized_sha256']=hashlib.sha256(np.asarray(call['normalized_actions'],dtype=np.float32).tobytes()).hexdigest();pair.append(item)
            fixed.append(dict(policy_call=l['policy_call'],batch_size=r['batch_size'],**numerical_pair(*pair)))
        total=sum(v['elements'] for v in fixed)
        rows[mode]=dict(**drift,fixed_identical_input_calls=len(fixed),fixed_input_rms=(float(np.sqrt(sum(v['squared_error_sum'] for v in fixed)/total)) if total else None),
            fixed_input_max_absolute_error=max((v['max_absolute_error'] for v in fixed),default=None),fixed_input_comparisons=fixed)
    require(sha(source)==proof['episode_bytes_sha256'],'Pilot changed during extraction')
    report=dict(scope='one_start_adjust_bottle_retained_pilot_only',primary_endpoint='singleton_vs_batched_stability',
        task=runtime['task'],seed=runtime['seed'],native_contract=runtime['native_contract'],measurements=rows,
        source_episodes_sha256=sha(source),source_audit_sha256=sha(proof_path),raw_frame_sha256=raw_sha,reporter_sha256=sha(Path(__file__)),
        independent_magnitude_audit_pending=True,full_campaign_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(a.output)


if __name__=='__main__':main()

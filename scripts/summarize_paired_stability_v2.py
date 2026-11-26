"""Quantify singleton/batched variation on independently audited paired starts.

Fixed-input normalized differences use only identical preprocessed requests at
batch sizes greater than one. Full-rollout action/pose drift is reported as a
closed-loop consequence, where later observations may already differ.
"""
import argparse
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    temporary.write_bytes(payload);temporary.replace(path)


def rotation_distance(a,b):
    """Quaternion angular distance in radians, stable near identical rotations."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    b=np.where((np.sum(a*b,axis=-1)<0)[...,None],-b,b)
    return 4*np.arctan2(np.linalg.norm(a-b,axis=-1),np.linalg.norm(a+b,axis=-1))


def pose_from_raw(model,raw):
    obs=raw['observation']
    if model=='pi05':
        eef=obs['robot_state']['eef'];return np.concatenate([np.asarray(eef['pos']).reshape(-1),np.asarray(eef['quat']).reshape(-1)]).astype(np.float64)
    if model in ('rdt','cogact') and 'tcp_pose' in obs.get('extra',{}):
        return np.asarray(obs['extra']['tcp_pose'],dtype=np.float64).reshape(7)
    if model=='cogact' and not obs.get('extra'):
        # Official BaseAgent.get_proprioception: xyz + quaternion wxyz + gripper.
        # This is relative to agent.base_pose, not the world or robot root pose.
        value=np.asarray(obs['agent']['eef_pos'],dtype=np.float64)
        require(value.shape==(8,) and np.isfinite(value).all(),'Invalid base-frame EEF observation')
        require(np.linalg.norm(value[3:7])>0,'Zero quaternion')
        return value[:7].copy()
    raise ValueError(model)


def normalized(call):
    if 'normalized_bytes' in call:
        signature=call['normalized'];require(signature['dtype']=='torch.bfloat16','Unknown normalized storage dtype')
        payload=bytes.fromhex(call['normalized_bytes']);require(hashlib.sha256(payload).hexdigest()==signature['sha256'],'Normalized bytes changed')
        bits=np.frombuffer(payload,dtype=np.uint16).reshape(signature['shape'])
        values=torch.frombuffer(bytearray(payload),dtype=torch.bfloat16).reshape(signature['shape']).float().numpy().astype(np.float64)
        return values,bits,16
    values=np.asarray(call['normalized_actions'],dtype=np.float32)
    expected=call.get('normalized_sha256',call.get('normalized',{}).get('sha256'))
    require(hashlib.sha256(values.tobytes()).hexdigest()==expected,'Normalized output changed')
    return values.astype(np.float64),values.view(np.uint32),32


def ulp_distance(a,b,width):
    sign=1<<(width-1);mask=(1<<width)-1
    a=a.astype(np.uint64);b=b.astype(np.uint64)
    ordered_a=np.where(a&sign,(~a)&mask,a|sign).astype(np.int64)
    ordered_b=np.where(b&sign,(~b)&mask,b|sign).astype(np.int64)
    return np.abs(ordered_a-ordered_b)


def numerical_pair(a,b):
    x,xb,width=normalized(a);y,yb,other_width=normalized(b)
    require(width==other_width and x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'Normalized output schema differs')
    delta=np.abs(x-y);relative=delta/np.maximum(np.maximum(np.abs(x),np.abs(y)),np.finfo(np.float64).tiny)
    return dict(exact=bool(np.array_equal(xb,yb)),elements=int(x.size),different_elements=int(np.count_nonzero(xb!=yb)),
        max_absolute_error=float(delta.max()),squared_error_sum=float(np.square(delta).sum()),rms_error=float(np.sqrt(np.mean(np.square(delta)))),
        max_symmetric_relative_error=float(relative.max()),max_ulp=int(ulp_distance(xb,yb,width).max()),source_float_bits=width)


def pose_cache(model,directory,episode,cache_directory):
    trajectory=episode['trajectory'];identity=digest(trajectory)
    extractor=hashlib.sha256(inspect.getsource(pose_from_raw).encode()).hexdigest()
    path=cache_directory/model/(identity+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            require(saved['extractor_sha256'].item()==extractor,'Cached pose extractor changed')
            pose=saved['pose'];require(sha_bytes(pose.tobytes())==saved['pose_sha256'].item(),'Pose cache bytes changed')
            require(pose.shape==(len(trajectory),7),'Cached pose coverage differs');return pose
    poses=[]
    for frame in trajectory:
        raw_path=(directory/frame['raw_frame']).resolve();require(raw_path.parent==(directory/'frames').resolve(),'Raw pose path escapes study')
        with gzip.open(raw_path,'rb') as f:raw=torch.load(f,map_location='cpu',weights_only=False)
        require(digest(raw)==raw_path.name.split('.')[0],'Raw pose content address changed')
        require(digest(raw['observation'])==frame['observation_sha256'],'Raw pose observation differs from audited trace')
        if model=='cogact':
            drawer=episode['task'] in ('open_drawer','close_drawer')
            require(('tcp_pose' not in raw['observation'].get('extra',{}))==drawer,'Task EEF frame/schema differs')
        poses.append(pose_from_raw(model,raw))
    pose=np.asarray(poses,dtype=np.float64);require(np.isfinite(pose).all(),'Nonfinite pose')
    buffer=io.BytesIO();np.savez_compressed(buffer,pose=pose,pose_sha256=sha_bytes(pose.tobytes()),extractor_sha256=extractor)
    atomic(path,buffer.getvalue());return pose


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def compare(model,directory,left,right,cache_directory):
    require(digest(left['trajectory'][0])==digest(right['trajectory'][0]),'Initial complete state differs')
    n=min(len(left['steps']),len(right['steps']));dtype=np.float64 if model=='cogact' else np.float32
    x=np.asarray([s['action'] for s in left['steps'][:n]],dtype=dtype);y=np.asarray([s['action'] for s in right['steps'][:n]],dtype=dtype)
    different=np.any(x.view(np.uint64 if dtype==np.float64 else np.uint32)!=y.view(np.uint64 if dtype==np.float64 else np.uint32),axis=1)
    delta=x.astype(np.float64)-y.astype(np.float64)
    fixed=[];same_b1=0;different_input_calls=0
    for a,b in zip(left['calls'],right['calls']):
        if a['target_input_sha256']!=b['target_input_sha256']:different_input_calls+=1;continue
        require(a['noise_id']==b['noise_id'],'Identical-input comparison has different noise')
        if b['batch_size']==1:same_b1+=1;continue
        fixed.append(dict(call=a['call'],batch_size=b['batch_size'],target_position=b['position'],input_sha256=a['target_input_sha256'],**numerical_pair(a,b)))
    p=pose_cache(model,directory,left,cache_directory)[:n+1];q=pose_cache(model,directory,right,cache_directory)[:n+1]
    position=np.linalg.norm(p[:,:3]-q[:,:3],axis=1);rotation=rotation_distance(p[:,3:7],q[:,3:7])
    frames=list(zip(left['trajectory'][:n+1],right['trajectory'][:n+1]))
    physical_key='sim_state_sha256' if model=='pi05' else 'physics_sha256'
    return dict(singleton_controls=len(left['steps']),batched_controls=len(right['steps']),aligned_controls=n,
        horizons_equal=len(left['steps'])==len(right['steps']),action_disagreement_controls=int(different.sum()),
        first_action_divergence_control_index=None if not different.any() else int(np.flatnonzero(different)[0]),
        action_component_max_absolute_difference=np.abs(delta).max(axis=0).tolist(),action_component_rms_difference=np.sqrt(np.mean(np.square(delta),axis=0)).tolist(),
        physical_state_disagreement_frames=sum(a[physical_key]!=b[physical_key] for a,b in frames),
        observation_disagreement_frames=sum(a['observation_sha256']!=b['observation_sha256'] for a,b in frames),
        eef_position_max_m=float(position.max()),eef_position_rms_m=float(np.sqrt(np.mean(np.square(position)))),
        eef_position_last_common_frame_m=float(position[-1]),eef_rotation_max_rad=float(rotation.max()),
        eef_rotation_rms_rad=float(np.sqrt(np.mean(np.square(rotation)))),outcome_disagreement=left['success']!=right['success'],
        fixed_identical_input_nonsingleton_pairs=fixed,identical_input_B1_controls=same_b1,
        later_calls_with_different_inputs=different_input_calls,
        interpretation='Physical/action drift is closed-loop; fixed-input numerical errors use only matching preprocessed inputs.')


def aggregate(rows):
    fixed=[v for r in rows for v in r['fixed_identical_input_nonsingleton_pairs']]
    elements=sum(v['elements'] for v in fixed);squared=sum(v['squared_error_sum'] for v in fixed)
    return dict(paired_starts=len(rows),starts_with_action_divergence=sum(r['action_disagreement_controls']>0 for r in rows),
        starts_with_recorded_physics_divergence=sum(r['physical_state_disagreement_frames']>0 for r in rows),
        starts_with_observation_divergence=sum(r['observation_disagreement_frames']>0 for r in rows),
        horizon_disagreements=sum(not r['horizons_equal'] for r in rows),outcome_disagreements=sum(r['outcome_disagreement'] for r in rows),
        eef_position_worst_m=max(r['eef_position_max_m'] for r in rows),
        eef_position_median_start_max_m=float(np.median([r['eef_position_max_m'] for r in rows])),
        eef_position_p95_start_max_m=float(np.quantile([r['eef_position_max_m'] for r in rows],0.95)),
        eef_rotation_worst_rad=max(r['eef_rotation_max_rad'] for r in rows),
        fixed_identical_input_nonsingleton_calls=len(fixed),fixed_input_nonexact_calls=sum(not v['exact'] for v in fixed),
        fixed_input_max_absolute_error=max((v['max_absolute_error'] for v in fixed),default=None),
        fixed_input_rms_error=float(np.sqrt(squared/elements)) if elements else None,
        fixed_input_max_ulp=max((v['max_ulp'] for v in fixed),default=None))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',nargs=2,action='append',required=True,metavar=('MODEL','DIRECTORY'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-proof',type=Path);args=p.parse_args();torch.set_num_threads(8)
    require(args.audit_proof is None or len(args.study)==1,'Explicit proof requires exactly one study')
    result=dict(primary_endpoint='singleton_vs_batched_stability',scope='independently_audited_complete_starts_only',
        success_maximization_is_not_an_objective=True,provisional_pending_full_campaign_QA=True,
        control_index_convention='Zero-based controls; frame0 is reset. Pose drift uses the common horizon without extrapolation.',
        normalized_error_scope='Only identical preprocessed input/noise, singleton B1 versus dynamic B>1; post-feedback differences excluded.',
        physical_action_units='Per-component native physical action units; do not pool components or models into one RMS.',
        eef_pose_frames='CogACT drawer tasks: robot base frame from agent.eef_pos; other tasks: world TCP pose. No pooling across frames/tasks.',studies=[],source_sha256=sha(Path(__file__)))
    atomic(args.output.parent/'stability_sources'/(result['source_sha256']+'.py'),Path(__file__).read_bytes())
    for model,folder in args.study:
        require(model in ('pi05','rdt','cogact'),'Unsupported model');d=Path(folder);audit_path=args.audit_proof if args.audit_proof is not None else d/'independent_progress_audit.json'
        payload=audit_path.read_bytes();a=json.loads(payload);require(a['all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks'],'Prior independent audit failed')
        require(sha(d/'candidate_lock.json')==a['candidate_lock_sha256'],'Candidate identity differs')
        snapshot=args.output.parent/'audit_snapshots'/(sha_bytes(payload)+'.json');atomic(snapshot,payload)
        candidate_snapshot=args.output.parent/'candidate_snapshots'/(a['candidate_lock_sha256']+'.json')
        atomic(candidate_snapshot,(d/'candidate_lock.json').read_bytes())
        study=dict(model=model,directory=str(d.resolve()),audit_snapshot=str(snapshot.resolve()),audit_sha256=sha_bytes(payload),
            candidate_snapshot=str(candidate_snapshot.resolve()),candidate_sha256=a['candidate_lock_sha256'],
            native_comparison_scope={'pi05':'Pinned LIBERO no-MSAA renderer; same checkpoint and inference path in both modes.',
                'rdt':'Controlled native and invariant both use fixed 30-token language padding.',
                'cogact':'Primary native and invariant both use fixed 64-token language padding; released-native longest-padding supplement is separate.'}[model],
            renderer_offscreen_samples=a.get('renderer_offscreen_samples'),tasks={},starts=[])
        result['studies'].append(study)
        for block in a['blocks']:
            task,initial=block['task'],block['initialization'];task_directory=f'task-{task:02d}' if model=='pi05' else str(task)
            directory=d/'blocks'/task_directory/f'start-{initial:02d}';episodes={};checks={};file_hashes={}
            names=['native_singleton','native_dynamic','patched_singleton','patched_dynamic']
            if model=='cogact':names+=['released_native_singleton','released_native_dynamic']
            for name in names:
                for suffix in ('','_repeat'):
                    path=directory/(name+suffix+'.json.gz');file_hashes[name+suffix]=sha(path)
                    with gzip.open(path,'rt') as f:e=json.load(f)
                    require(e['candidate_lock_sha256']==a['candidate_lock_sha256'] and e['task']==task and e['initialization']==initial,'Foreign episode')
                    require(e['success']==block['outcomes'][name],'Episode outcome differs from independent audit')
                    episodes[name+suffix]=e
                checks[name]=all(digest(episodes[name][field])==digest(episodes[name+'_repeat'][field]) for field in ('calls','steps','trajectory','success'))
            require(all(checks.values()),'Same-condition repeat controls differ')
            start=dict(eef_pose_frame=('robot_base' if model=='cogact' and task in ('open_drawer','close_drawer') else 'world'),task=task,initialization=initial,same_condition_repeat_controls_exact=checks,episode_file_sha256=file_hashes,comparisons={})
            for label,left,right in [('native','native_singleton','native_dynamic'),('invariant','patched_singleton','patched_dynamic')]+([('released_native','released_native_singleton','released_native_dynamic')] if model=='cogact' else []):
                metrics=compare(model,d,episodes[left],episodes[right],args.output.parent/'pose_cache_v2')
                if label=='native':
                    audited_first=(block['attribution']['native']['first_action_divergence'] if model=='cogact' else block['native_first_action_divergence'])
                    require(metrics['first_action_divergence_control_index']==audited_first,'First divergence differs from independent audit')
                if label=='released_native':metrics['interpretation']='Released longest-padding baseline; padding extent and batch arithmetic can jointly affect outputs. Only matching-input calls enter fixed-input errors.'
                if label=='invariant':require(metrics['action_disagreement_controls']==metrics['physical_state_disagreement_frames']==metrics['observation_disagreement_frames']==0 and metrics['horizons_equal'],'Invariant stability violation')
                start['comparisons'][label]=metrics
            study['starts'].append(start)
            for task_key in sorted({str(row['task']) for row in study['starts']}):
                study['tasks'][task_key]={label:aggregate([row['comparisons'][label] for row in study['starts'] if str(row['task'])==task_key]) for label in start['comparisons']}
            atomic(args.output,(json.dumps(result,indent=2)+'\n').encode())
            native=start['comparisons']['native'];print(f'{model}/{task}/{initial}: native differing controls={native["action_disagreement_controls"]}, EEF max={native["eef_position_max_m"]:.9g}m; invariant exact',flush=True)
    content=(json.dumps(result,indent=2)+'\n').encode();atomic(args.output.parent/'stability_snapshots'/(sha_bytes(content)+'.json'),content)


if __name__=='__main__':main()

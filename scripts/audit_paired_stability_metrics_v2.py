"""Independently reconstruct reported paired drift from raw episode/frame files.

Does not import the reporting script or its pose cache. This audit covers the
reported subset, not campaign completion or full operator qualification.
"""
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


def close(actual,expected,name):
    require(np.allclose(actual,expected,rtol=1e-12,atol=1e-14,equal_nan=False),'Incorrect '+name)


def numeric(call):
    if 'normalized_bytes' in call:
        payload=bytes.fromhex(call['normalized_bytes']);s=call['normalized']
        require(s['dtype']=='torch.bfloat16' and hashlib.sha256(payload).hexdigest()==s['sha256'],'Invalid BF16 bytes')
        bits=np.frombuffer(payload,dtype=np.uint16).reshape(s['shape'])
        return (bits.astype(np.uint32)<<16).view(np.float32).astype(np.float64),bits,16
    value=np.asarray(call['normalized_actions'],dtype=np.float32)
    signature=call['normalized_sha256'] if 'normalized_sha256' in call else call['normalized']['sha256']
    require(hashlib.sha256(value.tobytes()).hexdigest()==signature,'Invalid FP32 bytes')
    return value.astype(np.float64),value.view(np.uint32),32


def check_numeric(left,right,reported):
    x,xb,width=numeric(left);y,yb,other=numeric(right)
    require(width==other and x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'Numerical schema differs')
    different=int(np.count_nonzero(xb!=yb));error=x-y;absolute=np.abs(error)
    require(reported['exact']==(different==0) and reported['different_elements']==different and reported['elements']==x.size,'Incorrect numerical exactness/count')
    require(reported['source_float_bits']==width,'Incorrect numerical source precision')
    close(absolute.max(),reported['max_absolute_error'],'normalized maximum error')
    close(np.sum(error*error),reported['squared_error_sum'],'normalized squared error')
    close(np.sqrt(np.mean(error*error)),reported['rms_error'],'normalized RMS error')
    close(np.max(absolute/np.maximum(np.maximum(np.abs(x),np.abs(y)),np.finfo(np.float64).tiny)),reported['max_symmetric_relative_error'],'normalized relative error')
    sign=1<<(width-1);magnitude=sign-1
    def ordered(bits):
        bits=bits.astype(np.int64)
        return np.where(bits&sign,sign-1-(bits&magnitude),sign+(bits&magnitude))
    require(int(np.abs(ordered(xb)-ordered(yb)).max())==reported['max_ulp'],'Incorrect original-precision ULP distance')


def physical_pose(model,obs):
    if model=='pi05':
        state=obs['robot_state']['eef']
        return np.r_[np.asarray(state['pos']).reshape(-1),np.asarray(state['quat']).reshape(-1)]
    if model in ('rdt','cogact') and 'tcp_pose' in obs.get('extra',{}):
        return np.asarray(obs['extra']['tcp_pose']).reshape(-1)
    if model=='cogact' and not obs.get('extra'):
        proprio=np.asarray(obs['agent']['eef_pos'])
        require(proprio.shape==(8,) and np.isfinite(proprio).all(),'Drawer proprioception schema differs')
        require(sum(float(x)**2 for x in proprio[3:7])>0,'Zero quaternion')
        return np.concatenate((proprio[0:3],proprio[3:7]))
    raise ValueError('Unknown physical pose schema')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();payload=args.report.read_bytes();report=json.loads(payload);torch.set_num_threads(8)
    source=args.report.parent/'stability_sources'/(report['source_sha256']+'.py')
    require(sha(source)==report['source_sha256'],'Reporting source archive differs')
    verified=[];pose_cache={};frame_files=0;fixed_pairs=0
    for study in report['studies']:
        d=Path(study['directory']);model=study['model']
        audit=Path(study['audit_snapshot']);candidate=Path(study['candidate_snapshot'])
        require(sha(audit)==study['audit_sha256'] and sha(candidate)==study['candidate_sha256'],'Audit/candidate snapshot differs')
        prior=json.loads(audit.read_text());require(prior['all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks'],'Prior episode audit failed')
        require(len(study['starts'])==prior['completed_paired_starts'],'Reported cohort differs from audited cohort')
        require({(r['task'],r['initialization']) for r in study['starts']}=={(r['task'],r['initialization']) for r in prior['blocks']},'Reported starts differ')
        def poses(episode):
            nonlocal frame_files
            result=[]
            for frame in episode['trajectory']:
                path=(d/frame['raw_frame']).resolve();require(path.parent==(d/'frames').resolve(),'Frame path escapes study')
                if path not in pose_cache:
                    with gzip.open(path,'rb') as stream:raw=torch.load(stream,map_location='cpu',weights_only=False)
                    require(digest(raw)==path.name.split('.')[0] and digest(raw['observation'])==frame['observation_sha256'],'Raw frame/observation fingerprint differs')
                    obs=raw['observation']
                    if model=='cogact':
                        require(('robot_base' if 'tcp_pose' not in obs.get('extra',{}) else 'world')==start['eef_pose_frame'],'Raw EEF coordinate frame differs')
                    pose=physical_pose(model,obs)
                    require(pose.shape==(7,) and np.isfinite(pose).all(),'Invalid physical pose')
                    pose_cache[path]=(pose.astype(np.float64),frame['observation_sha256']);frame_files+=1
                require(pose_cache[path][1]==frame['observation_sha256'],'Repeated frame observation changed')
                result.append(pose_cache[path][0])
            return np.stack(result)
        for start in study['starts']:
            expected_frame='robot_base' if model=='cogact' and start['task'] in ('open_drawer','close_drawer') else 'world'
            require(start['eef_pose_frame']==expected_frame,'Incorrect EEF coordinate frame')
            task=start['task'];initial=start['initialization'];task_folder=f'task-{task:02d}' if model=='pi05' else str(task)
            folder=d/'blocks'/task_folder/f'start-{initial:02d}';episodes={}
            for condition,expected_sha in start['episode_file_sha256'].items():
                path=folder/(condition+'.json.gz');require(sha(path)==expected_sha,'Episode changed after reporting')
                with gzip.open(path,'rt') as stream:episode=json.load(stream)
                require(episode['candidate_lock_sha256']==study['candidate_sha256'],'Episode has foreign candidate')
                require(episode['task']==task and episode['initialization']==initial,'Episode has foreign task/start')
                episodes[condition]=episode
            for condition in start['same_condition_repeat_controls_exact']:
                require(start['same_condition_repeat_controls_exact'][condition],'Failed repeat control in report')
                for field in ('calls','steps','trajectory','success'):
                    require(digest(episodes[condition][field])==digest(episodes[condition+'_repeat'][field]),'Repeat control differs')
            for mode,m in start['comparisons'].items():
                prefix='patched' if mode=='invariant' else mode
                a=episodes[prefix+'_singleton'];b=episodes[prefix+'_dynamic']
                require(a['trajectory'][0]==b['trajectory'][0],'Initial recorded state differs')
                n=min(len(a['steps']),len(b['steps']));dtype=np.float64 if model=='cogact' else np.float32
                x=np.asarray([s['action'] for s in a['steps'][:n]],dtype=dtype);y=np.asarray([s['action'] for s in b['steps'][:n]],dtype=dtype)
                different=np.asarray([u.tobytes()!=v.tobytes() for u,v in zip(x,y)])
                require(int(different.sum())==m['action_disagreement_controls'],'Incorrect action disagreement count')
                first=int(np.nonzero(different)[0][0]) if different.any() else None
                require(first==m['first_action_divergence_control_index'],'Incorrect first action divergence')
                delta=x.astype(np.float64)-y.astype(np.float64)
                close(np.abs(delta).max(axis=0),m['action_component_max_absolute_difference'],'action component maxima')
                close(np.sqrt(np.mean(delta*delta,axis=0)),m['action_component_rms_difference'],'action component RMS')
                require(m['aligned_controls']==n and m['singleton_controls']==len(a['steps']) and m['batched_controls']==len(b['steps']),'Incorrect horizon coverage')
                require(m['horizons_equal']==(len(a['steps'])==len(b['steps'])),'Incorrect horizon equality')
                physical='sim_state_sha256' if model=='pi05' else 'physics_sha256'
                for field,key in ((physical,'physical_state_disagreement_frames'),('observation_sha256','observation_disagreement_frames')):
                    require(sum(u[field]!=v[field] for u,v in zip(a['trajectory'][:n+1],b['trajectory'][:n+1]))==m[key],'Incorrect '+key)
                u=poses(a)[:n+1];v=poses(b)[:n+1]
                distance=np.sqrt(np.sum((u[:,:3]-v[:,:3])**2,axis=1))
                close(distance.max(),m['eef_position_max_m'],'maximum position drift')
                close(np.sqrt(np.mean(distance**2)),m['eef_position_rms_m'],'RMS position drift')
                close(distance[-1],m['eef_position_last_common_frame_m'],'last common position drift')
                # Independent quaternion formula: relative scalar product and
                # vector magnitude, with exact equality handled before acos.
                rotations=[]
                for q,r in zip(u[:,3:],v[:,3:]):
                    if np.array_equal(q,r) or np.array_equal(q,-r):rotations.append(0.);continue
                    q=q/np.linalg.norm(q);r=r/np.linalg.norm(r)
                    w=float(np.clip(abs(q@r),0.,1.))
                    rotations.append(2*np.arctan2(np.sqrt(max(0.,1-w*w)),w))
                require(np.isclose(max(rotations),m['eef_rotation_max_rad'],rtol=1e-8,atol=1e-7),'Incorrect maximum angular drift')
                require(np.isclose(np.sqrt(np.mean(np.square(rotations))),m['eef_rotation_rms_rad'],rtol=1e-8,atol=1e-7),'Incorrect RMS angular drift')
                expected=[(l,r) for l,r in zip(a['calls'],b['calls']) if l['target_input_sha256']==r['target_input_sha256'] and r['batch_size']>1]
                require(len(expected)==len(m['fixed_identical_input_nonsingleton_pairs']),'Incorrect fixed-input numerical scope')
                for (l,r),measurement in zip(expected,m['fixed_identical_input_nonsingleton_pairs']):
                    require(l['noise_id']==r['noise_id'] and measurement['call']==l['call'] and measurement['batch_size']==r['batch_size'] and measurement['target_position']==r['position'] and measurement['input_sha256']==l['target_input_sha256'],'Numerical request/noise identity differs')
                    check_numeric(l,r,measurement);fixed_pairs+=1
                require(m['outcome_disagreement']==(a['success']!=b['success']),'Incorrect secondary outcome disagreement')
                if mode=='invariant':require(not different.any() and np.array_equal(u,v) and a['trajectory']==b['trajectory'],'Invariant full recorded trajectory differs')
            verified.append(dict(model=model,task=task,initialization=initial))
            print(f'verified {model}/{task}/{initial}: raw drift and normalized errors',flush=True)
        require(set(study['tasks'])=={str(s['task']) for s in study['starts']},'Aggregate task coverage differs')
        for task,modes in study['tasks'].items():
            for mode,summary in modes.items():
                rows=[s['comparisons'][mode] for s in study['starts'] if str(s['task'])==task]
                require(summary['paired_starts']==len(rows),'Aggregate paired-start count differs')
                for target,source in [('starts_with_action_divergence','action_disagreement_controls'),
                    ('starts_with_recorded_physics_divergence','physical_state_disagreement_frames'),
                    ('starts_with_observation_divergence','observation_disagreement_frames')]:
                    require(summary[target]==sum(r[source]>0 for r in rows),'Incorrect aggregate '+target)
                require(summary['horizon_disagreements']==sum(not r['horizons_equal'] for r in rows),'Incorrect horizon aggregate')
                require(summary['outcome_disagreements']==sum(r['outcome_disagreement'] for r in rows),'Incorrect secondary outcome aggregate')
                maxima=[r['eef_position_max_m'] for r in rows]
                close(max(maxima),summary['eef_position_worst_m'],'aggregate worst position drift')
                close(np.median(maxima),summary['eef_position_median_start_max_m'],'aggregate median position drift')
                close(np.quantile(maxima,0.95),summary['eef_position_p95_start_max_m'],'aggregate p95 position drift')
                close(max(r['eef_rotation_max_rad'] for r in rows),summary['eef_rotation_worst_rad'],'aggregate worst rotation drift')
                fixed=[v for r in rows for v in r['fixed_identical_input_nonsingleton_pairs']]
                require(summary['fixed_identical_input_nonsingleton_calls']==len(fixed),'Incorrect fixed-input aggregate count')
                require(summary['fixed_input_nonexact_calls']==sum(not v['exact'] for v in fixed),'Incorrect fixed-input nonexact count')
                if fixed:
                    close(max(v['max_absolute_error'] for v in fixed),summary['fixed_input_max_absolute_error'],'aggregate fixed-input maximum')
                    close(np.sqrt(sum(v['squared_error_sum'] for v in fixed)/sum(v['elements'] for v in fixed)),summary['fixed_input_rms_error'],'aggregate fixed-input RMS')
                    require(max(v['max_ulp'] for v in fixed)==summary['fixed_input_max_ulp'],'Incorrect aggregate ULP')
                else:require(all(summary[k] is None for k in ('fixed_input_max_absolute_error','fixed_input_rms_error','fixed_input_max_ulp')),'Invented fixed-input aggregate')
    require(args.report.read_bytes()==payload,'Report changed during audit')
    result=dict(status='reported_subset_stability_metrics_independently_verified',report_sha256=hashlib.sha256(payload).hexdigest(),
        reporting_source_sha256=report['source_sha256'],auditor_sha256=sha(Path(__file__)),raw_frame_files_verified=frame_files,
        fixed_input_nonsingleton_pairs_verified=fixed_pairs,paired_starts_verified=verified,
        angular_formula_crosscheck_absolute_tolerance_rad=1e-7,full_four_model_campaign_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Source-locked RDT paired study: five tasks x50 starts x8 complete episodes.

Candidate results remain provisional until all numerical/fidelity gates pass.
The controlled native and invariant paths both use fixed30 language padding.
"""
import argparse
from collections import deque
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from scripts.maniskill_state_evidence import capture
from scripts.qualify_rdt_observable_state_replay import TASKS,make_env,render
from scripts.qualify_simpler_fixed_action_replay import digest


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_bytes(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_json(path,value,compressed=False):
    payload=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    atomic_bytes(path,gzip.compress(payload,mtime=0) if compressed else payload)


def retain_raw(root,category,value):
    path=root/category/(digest(value)+'.pt.gz')
    if not path.exists():
        buffer=io.BytesIO();torch.save(value,buffer);atomic_bytes(path,gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
    return str(path.relative_to(root))


def tensor_record(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),
        sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def compare_block(episodes):
    base=['native_singleton','native_dynamic','patched_singleton','patched_dynamic']
    for name in base:
        for field in ('calls','steps','trajectory','success'):
            if digest(episodes[name][field])!=digest(episodes[name+'_repeat'][field]):raise RuntimeError(f'A/A differs: {name}/{field}')
    if len({digest(row['trajectory'][0]) for row in episodes.values()})!=1:raise RuntimeError('Initial canonical observable state differs')
    if len({row['calls'][0]['target_input_sha256'] for row in episodes.values()})!=1:raise RuntimeError('Initial model input differs')
    a,b=episodes['patched_singleton'],episodes['patched_dynamic']
    for field in ('steps','trajectory','success'):
        if digest(a[field])!=digest(b[field]):raise RuntimeError(f'Invariant batching changes {field}')
    if [c['normalized'] for c in a['calls']]!=[c['normalized'] for c in b['calls']]:raise RuntimeError('Invariant normalized chunks differ')
    a,b=episodes['native_singleton'],episodes['native_dynamic']
    first=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if np.asarray(x['action'],dtype=np.float32).tobytes()!=np.asarray(y['action'],dtype=np.float32).tobytes()),None)
    if first is not None:
        ca=a['calls'][a['steps'][first]['policy_call']];cb=b['calls'][b['steps'][first]['policy_call']]
        if ca['target_input_sha256']!=cb['target_input_sha256'] or digest(a['trajectory'][:first+1])!=digest(b['trajectory'][:first+1]):raise RuntimeError('Native first-action attribution failed')
    return first


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','siglip','source','manifest','qa-runtime','qa-records','reset-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tasks',nargs='+',default=TASKS,help='Worker shard; global cohort always all five tasks x50 starts')
    args=p.parse_args()
    if not args.tasks or len(set(args.tasks))!=len(args.tasks) or not set(args.tasks)<=set(TASKS):raise ValueError('Invalid task shard')
    reset=json.loads(args.reset_audit.read_text())
    if reset.get('status')!='complete_observable_state_fixed_replay_independently_verified' or set(reset['tasks'])!=set(TASKS) or reset['initializations']<3 or reset['repeats']<3:raise ValueError('Full observable reset qualification is required')
    os.environ.update(RDT_CHECKPOINT=str(args.checkpoint),RDT_SIGLIP=str(args.siglip),RDT_SOURCE=str(args.source),RDT_MANISKILL_EXACT_MANIFEST=str(args.manifest))
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters.rdt_maniskill_exact import RDTManiSkillExactAdapter
    from scripts.run_rdt_paired_server import PairedRDTBackend,verify_artifacts
    import mani_skill
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    qa=json.loads(args.qa_runtime.read_text())
    def verify_qa():
        for path,expected in qa['source_sha256'].items():
            if sha(path)!=expected:raise RuntimeError(f'QA inference source changed: {path}')
    verify_qa()
    adapter=RDTManiSkillExactAdapter();policy=adapter.load_model();policy.eval()
    manifest=adapter.manifest
    if sha(args.manifest)!=qa['replay_manifest_sha256']:raise ValueError('QA manifest differs')
    companions=[dict(request=adapter.load_replay_input(row['replay_index'],0),task=row['task'],request_id=row['request_ids'][0]) for row in manifest['observations'] if row['split']=='diagnostic']
    if len(companions)!=200:raise ValueError('Expected 200 frozen diagnostic companions')
    def infer(samples,invariant,position=0):
        inputs={}
        def trace(label,value):
            if label in ('input.language','input.state','input.noise','input.vision_pixels'):inputs[label]=tensor_record(value[position:position+1])
        adapter.trace_callback=trace;adapter.reset_model(policy)
        try:
            with torch.inference_mode(),set_batch_invariant_mode(invariant):out=adapter.run_model(policy,samples)
        finally:adapter.trace_callback=None
        if set(inputs)!= {'input.language','input.state','input.noise','input.vision_pixels'}:raise RuntimeError('Input trace coverage incomplete')
        for value in out.values():
            if not torch.isfinite(value).all():raise RuntimeError('Nonfinite policy output')
        return {k:v.detach().cpu() for k,v in out.items()},inputs
    signatures={};request_id=companions[0]['request_id']
    with gzip.open(args.qa_records,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['request_id']==request_id and row['transformation']=='singleton':signatures[row['implementation']]={k:v['candidate_sha256'] for k,v in row['output']['tensors'].items()}
            if len(signatures)==2:break
    if set(signatures)!={'native','batch_invariant'}:raise ValueError('QA singleton signatures are missing')
    for mode,invariant in [('native',False),('batch_invariant',True)]:
        out,_=infer([companions[0]['request']],invariant)
        if {f'output.{k}':tensor_record(v)['sha256'] for k,v in out.items()}!=signatures[mode]:raise RuntimeError(f'Actual GPU does not reproduce {mode} QA singleton bytes')
        for size in (2,4,8):infer([c['request'] for c in companions[:size]],invariant)
    # Exercise all simulator task classes before pinning loaded sources.
    for task in TASKS:
        random.seed(20241201);np.random.seed(20241201);torch.manual_seed(20241201)
        env=make_env(task)
        try:obs,_=env.reset(seed=20241201);capture(env,obs,render(env))
        finally:env.close()
    source=dict(qa['source_sha256'])
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file and Path(file).is_file() and name.startswith(('mani_skill','sapien','gymnasium','transformers','batch_invariant_ops')):source[str(Path(file).resolve())]=sha(file)
    for name in ('run_rdt_maniskill_paired_study.py','maniskill_state_evidence.py','qualify_rdt_observable_state_replay.py','qualify_simpler_fixed_action_replay.py'):
        file=Path(__file__).with_name(name).resolve();source[str(file)]=sha(file)
    assets={row['path']:row['sha256'] for row in verify_artifacts(args.checkpoint,args.siglip)}
    for root in (mani_skill.PACKAGE_ASSET_DIR,mani_skill.ASSET_DIR,mani_skill.PACKAGE_DIR/'utils/scene_builder/table/assets'):
        if root.exists():
            for file in root.rglob('*'):
                if file.is_file() and '.cache' not in file.parts and not file.name.endswith('.lock'):assets[str(file.resolve())]=sha(file)
    lock=dict(scope='paired_candidate_source_lock_not_final_numerical_acceptance',model='rdt_maniskill_exact',global_tasks=TASKS,
        initialization_ids=list(range(50)),conditions=['native_singleton','native_dynamic','patched_singleton','patched_dynamic'],
        repeat_controls=True,dynamic_cycle=[1,2,4,8],max_steps=400,executed_action_stride=4,
        language_padding='fixed30_in_both_controlled_primary_modes',reward_mode_deviation='PlugCharger_sparse_only_on_ManiSkill3.0.1',
        source_sha256=source,asset_sha256=assets,replay_manifest_path=str(args.manifest.resolve()),
        companion_schedule='diagnostic_other_tasks_offset_init97_call11_slot17_v1',
        replay_manifest_sha256=sha(args.manifest),qa_runtime_sha256=sha(args.qa_runtime),
        reset_audit_sha256=sha(args.reset_audit),qa_singleton_signatures=signatures,torch=torch.__version__,gpu_type=torch.cuda.get_device_name(),numerical_qa_acceptance_pending=True)
    with (args.output/'study.lock').open('w') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        path=args.output/'candidate_lock.json'
        if path.exists():
            if json.loads(path.read_text())!=lock:raise RuntimeError('Candidate lock differs')
        else:
            atomic_json(path,lock)
            for file,expected in source.items():atomic_bytes(args.output/'source_snapshots'/expected,Path(file).read_bytes())
    lock_hash=sha(args.output/'candidate_lock.json');gpu=str(torch.cuda.get_device_properties(0).uuid)
    asset_stats={file:(Path(file).stat().st_size,Path(file).stat().st_mtime_ns) for file in assets}
    released=PairedRDTBackend(adapter.inner,policy,condition='native_dynamic',checkpoint=args.checkpoint,
        language=adapter.inner.language_embeddings,artifacts=verify_artifacts(args.checkpoint,args.siglip))
    with (args.output/f'worker-{gpu}.lock').open('w') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(args.output/f'worker-{gpu}.json',dict(gpu_uuid=gpu,tasks=args.tasks,candidate_lock_sha256=lock_hash,qa_singletons_reproduced=True))
        conditions={name:(name.startswith('patched'),name.endswith('dynamic')) for name in lock['conditions']}
        conditions.update({name+'_repeat':value for name,value in list(conditions.items())})
        for task in args.tasks:
            for initialization in range(50):
                folder=args.output/'blocks'/task/f'start-{initialization:02d}';folder.mkdir(parents=True,exist_ok=True)
                with (folder/'block.lock').open('w') as guard:
                    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if (folder/'block_summary.json').exists():continue
                    owner=folder/'owner.json'
                    if owner.exists() and json.loads(owner.read_text())['gpu_uuid']!=gpu:continue
                    if not owner.exists():atomic_json(owner,dict(gpu_uuid=gpu,candidate_lock_sha256=lock_hash))
                    verify_qa()
                    for file,expected in source.items():
                        if sha(file)!=expected:raise RuntimeError(f'Locked source changed: {file}')
                    for file,expected in assets.items():
                        stat=(Path(file).stat().st_size,Path(file).stat().st_mtime_ns)
                        if stat!=asset_stats[file]:
                            if sha(file)!=expected:raise RuntimeError(f'Locked asset changed: {file}')
                            asset_stats[file]=stat
                    episodes={}
                    for condition,(invariant,dynamic) in conditions.items():
                        path=folder/(condition+'.json.gz')
                        if path.exists():
                            with gzip.open(path,'rt') as f:row=json.load(f)
                            if row['gpu_uuid']!=gpu or row['candidate_lock_sha256']!=lock_hash:raise RuntimeError('Foreign stored episode')
                            episodes[condition]=row;continue
                        seed=20241201+initialization;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        env=make_env(task)
                        try:
                            obs,_=env.reset(seed=seed);rgb=render(env);history=deque([None,rgb.copy()],maxlen=2)
                            trajectory=[];steps=[];calls=[];success=False;fidelity=None
                            def frame():
                                evidence,raw=capture(env,obs,rgb);trajectory.append(dict(evidence,raw_frame=retain_raw(args.output,'frames',raw)))
                            frame()
                            if episodes and digest(trajectory[0])!=digest(next(iter(episodes.values()))['trajectory'][0]):raise RuntimeError('Canonical reset observable state differs')
                            while len(steps)<400 and not success:
                                call=len(calls);key=f'{task}/{initialization}/policy-call-{call}';noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
                                noise=torch.randn(64,128,generator=torch.Generator().manual_seed(noise_seed)).numpy()
                                target=dict(task_name=task,paired_request_id='target',history_valid=[value is not None for value in history],
                                    image_history=np.stack([np.zeros_like(rgb) if value is None else value for value in history]),
                                    proprio=np.ascontiguousarray(obs['agent']['qpos'].cpu().numpy()[:,:-1].reshape(-1),dtype=np.float32),noise=noise)
                                size=[1,2,4,8][call%4] if dynamic else 1;position=call%size
                                pool=[c for c in companions if c['task']!=task]
                                selected=[pool[(initialization*97+call*11+j*17)%len(pool)] for j in range(size-1)]
                                samples=[c['request'] for c in selected];samples.insert(position,target)
                                out,inputs=infer(samples,invariant,position);normalized=out['normalized'][position:position+1];actions=out['physical'][position].numpy()
                                inputs.update(language_mask=[j<adapter.inner.language_embeddings[task].shape[1] for j in range(30)],control_frequency=25,history_valid=target['history_valid'])
                                raw_target_file=retain_raw(args.output,'requests',target)
                                if condition=='native_singleton' and call==0:
                                    adapter.reset_model(policy)
                                    with torch.inference_mode(),set_batch_invariant_mode(False):original=released.infer({'examples':[target]})['actions'][0]
                                    fidelity=dict(comparison='controlled_fixed30_native_vs_released_longest_singleton',exact=actions.tobytes()==original.tobytes(),max_error=float(np.max(np.abs(actions-original))))
                                calls.append(dict(call=call,batch_size=size,position=position,noise_id=key,noise_seed=noise_seed,
                                    raw_request=raw_target_file,raw_request_sha256=digest(target),target_input_sha256=digest(inputs),model_inputs=inputs,
                                    companion_ids=[c['request_id'] for c in selected],normalized=tensor_record(normalized),
                                    normalized_bytes=normalized.contiguous().view(torch.uint8).numpy().tobytes().hex(),actions=actions.tolist()))
                                for action in actions[::4]:
                                    obs,reward,terminated,truncated,info=env.step(action);rgb=render(env);history.append(rgb.copy())
                                    success=(bool(terminated) or bool(truncated)) and bool(info['success'])
                                    steps.append(dict(policy_call=call,action=action.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=bool(info['success'])))
                                    frame()
                                    if success or len(steps)>=400:break
                            row=dict(task=task,initialization=initialization,seed=seed,condition=condition,gpu_uuid=gpu,candidate_lock_sha256=lock_hash,
                                calls=calls,steps=steps,trajectory=trajectory,success=success,singleton_padding_fidelity=fidelity,numerical_qa_acceptance_pending=True)
                            atomic_json(path,row,compressed=True);episodes[condition]=row
                            print(f'EPISODE {task}/{initialization}/{condition}: success={success}, controls={len(steps)}',flush=True)
                        finally:env.close()
                    first=compare_block(episodes)
                    summary=dict(task=task,initialization=initialization,gpu_uuid=gpu,candidate_lock_sha256=lock_hash,
                        exact_reset_and_AA_controls=True,invariant_trajectories_exact=True,native_first_action_divergence=first,
                        success={name:row['success'] for name,row in episodes.items()},numerical_qa_acceptance_pending=True)
                    atomic_json(folder/'block_summary.json',summary);print(f'BLOCK {task}/{initialization}: {summary["success"]}',flush=True)


if __name__=='__main__':main()

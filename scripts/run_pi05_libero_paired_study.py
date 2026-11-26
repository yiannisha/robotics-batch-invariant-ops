"""Source-locked varied-start paired study, provisional until numerical QA passes.

All ten LIBERO-10 tasks x50 official starts are prescribed. Every block contains
four conditions and four repeat controls on one GPU. Atomic episode files and
per-block locks support restart and multiple workers without dropping failures.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qualify_simpler_fixed_action_replay import digest
from scripts.probe_pi05_libero_real_inputs import tensor_digest
from scripts.sim_state_evidence import observation_evidence
from scripts.verify_pi05_libero_checkpoint import load_verified_policy


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value,compressed=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    if compressed:payload=gzip.compress(payload,mtime=0)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def same(a,b):return digest(a)==digest(b)


def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,temporary)
    with temporary.open('rb') as f:os.fsync(f.fileno())
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','tokenizer','manifest','qa-runtime','qa-records','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tasks',nargs='+',type=int,default=list(range(10)),help='worker task subset; global study always covers all ten')
    args=p.parse_args()
    if not args.tasks or len(set(args.tasks))!=len(args.tasks) or any(t not in range(10) for t in args.tasks):raise ValueError('Invalid task shard')
    args.output.mkdir(parents=True,exist_ok=True)
    from batch_invariant_ops import set_batch_invariant_mode
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.policies.factory import make_pre_post_processors
    class EpisodeEnv(LiberoEnv):
        def step(self,action):
            obs,reward,done,info=self._env.step(action);success=self._env.check_success();terminated=done or success
            info.update(task=self.task,task_id=self.task_id,done=done,is_success=success)
            return self._format_raw_obs(obs),reward,terminated,False,info

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    qa=json.loads(args.qa_runtime.read_text())
    def verify_qa_source():
        for path,expected in qa['source_sha256'].items():
            if sha(Path(path))!=expected:raise RuntimeError(f'Inference source changed after numerical QA began: {path}')
    verify_qa_source()
    manifest=json.loads(args.manifest.read_text())
    audit=Path(manifest['corpus_audit_path'])
    if sha(audit)!=manifest['corpus_audit_sha256'] or json.loads(audit.read_text())['status']!='complete_corpus_independently_verified':raise ValueError('Unverified corpus')
    acquisition=json.loads((args.checkpoint/'acquisition.json').read_text())
    if acquisition['revision']!='7152f13e34c6c115d808aea0e5d08aca74eac14b':raise ValueError('Foreign checkpoint')
    assets={}
    for entry in acquisition['files']:
        path=args.checkpoint/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError(f'Changed checkpoint asset: {path}')
        assets[str(path.resolve())]=entry['sha256']
    policy,loading=load_verified_policy(args.checkpoint);policy.to('cuda').eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=args.checkpoint,
        preprocessor_overrides={'tokenizer_processor':{'tokenizer_name':str(args.tokenizer)},'device_processor':{'device':'cpu'}})
    suite=benchmark.get_benchmark_dict()['libero_10']()
    noises=np.load(manifest['noise_file'],mmap_mode='r')
    companions=[]
    for row in manifest['observations']:
        if row['split']!='diagnostic':continue
        path=Path(manifest['fixture_root'])/row['input_file']
        if sha(path)!=row['input_file_sha256']:raise ValueError('Changed companion input')
        noise=np.array(noises[row['replay_index'],0],copy=True)
        if hashlib.sha256(noise.tobytes()).hexdigest()!=row['noise'][0]['sha256']:raise ValueError('Changed companion noise')
        companions.append(dict(inputs=torch.load(path,map_location='cpu',weights_only=True),noise=torch.from_numpy(noise).unsqueeze(0),
                               task=row['task'],episode=row['episode_id'],request_id=row['request_ids'][0]))
    if len(companions)!=200:raise ValueError('Expected200 diagnostic companions')

    def infer(samples,invariant):
        batch={k:torch.cat([s['inputs'][k] for s in samples]).to('cuda') for k in samples[0]['inputs']}
        noise=torch.cat([s['noise'] for s in samples]).to('cuda')
        policy.reset()
        with torch.inference_mode(),set_batch_invariant_mode(invariant):
            normalized=policy.predict_action_chunk(batch,noise=noise.clone()).cpu()
            physical=post(normalized).cpu()
        if not torch.isfinite(normalized).all() or not torch.isfinite(physical).all():raise RuntimeError('Nonfinite inference output')
        return normalized,physical

    # Requalify the inference path on this actual GPU against recorded QA
    # singleton bytes, using an unchanged diagnostic request and explicit noise.
    first=manifest['observations'][0];signatures={}
    with gzip.open(args.qa_records,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['request_id']==first['request_ids'][0] and row['transformation']=='singleton':
                signatures[row['implementation']]={k:v['candidate_sha256'] for k,v in row['output']['tensors'].items()}
            if len(signatures)==2:break
    if set(signatures)!={'native','batch_invariant'}:raise ValueError('Missing recorded QA singleton signatures')
    for mode,invariant in (('native',False),('batch_invariant',True)):
        normalized,physical=infer([companions[0]],invariant)
        if tensor_digest(normalized)!=signatures[mode]['output.normalized'] or tensor_digest(physical)!=signatures[mode]['output.physical']:
            raise RuntimeError(f'{mode} singleton does not reproduce the QA GPU output bytes')

    # Initialize the renderer before taking the dependency snapshot. This
    # warm-up is unscored; every scored episode constructs another fresh scene.
    warm=EpisodeEnv(suite,0,'libero_10',obs_type='pixels_agent_pos',episode_index=0)
    try:warm.init_state_id=0;warm.reset(seed=20261108)
    finally:warm.close()
    simulator_assets={}
    for category in ('assets','bddl_files','init_states'):
        root=Path(get_libero_path(category))
        for file in root.rglob('*'):
            if file.is_file() and not any(part in ('.git','.cache','__pycache__') for part in file.parts) and not file.name.endswith('.lock'):
                simulator_assets[str(file.resolve())]=sha(file)

    source=dict(qa['source_sha256'])
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file() and name.startswith(('libero','robosuite','mujoco','transformers','lerobot','batch_invariant_ops')):
            source[str(Path(path).resolve())]=sha(Path(path))
    for file in (Path(__file__),Path(__file__).with_name('sim_state_evidence.py')):source[str(file.resolve())]=sha(file)
    tokenizer_files={str(file.resolve()):sha(file) for file in args.tokenizer.rglob('*') if file.is_file() and '.cache' not in file.parts}
    lock=dict(scope='paired_candidate_source_lock_not_final_numerical_acceptance',model='pi05_libero',
        global_tasks=list(range(10)),initialization_ids=list(range(50)),conditions=['native_singleton','native_dynamic','patched_singleton','patched_dynamic'],
        repeat_controls=True,dynamic_cycle=[1,2,4,8],max_steps=520,action_steps=policy.config.n_action_steps,
        source_sha256=source,checkpoint_asset_sha256=assets,tokenizer_sha256=tokenizer_files,simulator_asset_sha256=simulator_assets,
        replay_manifest_sha256=sha(args.manifest),qa_runtime_sha256=sha(args.qa_runtime),qa_singleton_signatures=signatures,
        torch=torch.__version__,gpu_type=torch.cuda.get_device_name(),numerical_qa_acceptance_pending=True)
    with (args.output/'study.lock').open('w') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        lock_path=args.output/'candidate_lock.json'
        if lock_path.exists():
            if json.loads(lock_path.read_text())!=lock:raise RuntimeError('Study lock differs; do not mix candidates/runtimes')
        else:
            atomic_json(lock_path,lock)
            for path,expected in source.items():
                blob=args.output/'source_snapshots'/expected;blob.parent.mkdir(exist_ok=True)
                if not blob.exists():blob.write_bytes(Path(path).read_bytes())
        lock_hash=sha(lock_path)
    gpu=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    immutable_data={**assets,**tokenizer_files,**simulator_assets}
    data_stats={path:(Path(path).stat().st_size,Path(path).stat().st_mtime_ns) for path in immutable_data}
    with (args.output/f'worker-{gpu}.lock').open('w') as gpu_guard:
        fcntl.flock(gpu_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(args.output/f'worker-{gpu}.json',dict(gpu_uuid=gpu,selected_tasks=args.tasks,candidate_lock_sha256=lock_hash,
            renderer_egl_device_id=os.environ.get('MUJOCO_EGL_DEVICE_ID'),checkpoint_loading=loading,qa_singletons_reproduced=True))
        conditions={name:(name.startswith('patched'),name.endswith('dynamic')) for name in lock['conditions']}
        conditions.update({name+'_repeat':value for name,value in list(conditions.items())})
        def batch_raw(value):
            if isinstance(value,np.ndarray):return value[None]
            if isinstance(value,dict):return {k:batch_raw(v) for k,v in value.items()}
            return value
        def process(obs,instruction):
            batch=LiberoProcessorStep().observation(preprocess_observation(batch_raw(obs)));batch['task']=[instruction]
            return {k:v.cpu() for k,v in pre(batch).items() if isinstance(v,torch.Tensor)}
        for task in args.tasks:
            for initialization in range(50):
                folder=args.output/'blocks'/f'task-{task:02d}'/f'start-{initialization:02d}';folder.mkdir(parents=True,exist_ok=True)
                with (folder/'block.lock').open('w') as guard:
                    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if (folder/'block_summary.json').exists():continue
                    owner=folder/'owner.json'
                    if owner.exists() and json.loads(owner.read_text())['gpu_uuid']!=gpu:continue
                    if not owner.exists():atomic_json(owner,dict(gpu_uuid=gpu,candidate_lock_sha256=lock_hash))
                    verify_qa_source()
                    for path,expected in source.items():
                        if sha(Path(path))!=expected:raise RuntimeError(f'Locked source changed: {path}')
                    for asset,expected in immutable_data.items():
                        current=(Path(asset).stat().st_size,Path(asset).stat().st_mtime_ns)
                        if current!=data_stats[asset]:
                            if sha(Path(asset))!=expected:raise RuntimeError(f'Locked model/simulator asset changed: {asset}')
                            data_stats[asset]=current
                    episodes={}
                    for condition,(invariant,dynamic) in conditions.items():
                        path=folder/(condition+'.json.gz')
                        if path.exists():
                            with gzip.open(path,'rt') as f:record=json.load(f)
                            if record['candidate_lock_sha256']!=lock_hash or record['gpu_uuid']!=gpu:raise RuntimeError('Stored episode belongs to another candidate/device')
                            episodes[condition]=record;continue
                        seed=20261108+initialization
                        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        env=EpisodeEnv(suite,task,'libero_10',obs_type='pixels_agent_pos',episode_index=initialization)
                        try:
                            env.init_state_id=initialization;obs,_=env.reset(seed=seed);policy.reset()
                            initial,snapshot=observation_evidence(env,obs,include_rng=True)
                            snapshot_path=folder/'initial_state.pt'
                            if not snapshot_path.exists():atomic_torch(snapshot_path,dict(snapshot,observation=obs))
                            if episodes and not same(initial,next(iter(episodes.values()))['trajectory'][0]):raise RuntimeError('Canonical reset physics/controller/RNG/observation differs')
                            trajectory=[initial];calls=[];steps=[];success=False
                            while len(steps)<520 and not success:
                                call=len(calls);inputs=process(obs,env.task_description)
                                key=f'libero_10/{task}/{initialization}/policy-call-{call}';noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
                                noise=torch.randn(1,policy.config.chunk_size,policy.config.max_action_dim,generator=torch.Generator().manual_seed(noise_seed))
                                input_sha=digest({k:v.numpy() for k,v in inputs.items()}|{'noise':noise.numpy()})
                                size=[1,2,4,8][call%4] if dynamic else 1;position=call%size
                                selected=[]
                                for other in sorted({c['task'] for c in companions if c['task']!=task}):
                                    pool=[c for c in companions if c['task']==other]
                                    selected.append(pool[(initialization*17+call)%len(pool)])
                                selected=selected[:size-1];samples=list(selected);samples.insert(position,dict(inputs=inputs,noise=noise))
                                normalized,actions=infer(samples,invariant);normalized=normalized[position:position+1];actions=actions[position]
                                raw_path=args.output/'observations'/f'{digest(obs)}.pt'
                                if not raw_path.exists():atomic_torch(raw_path,obs)
                                calls.append(dict(call=call,batch_size=size,position=position,noise_id=key,noise_seed=noise_seed,
                                    target_input_sha256=input_sha,normalized_sha256=tensor_digest(normalized),normalized_actions=normalized.tolist(),actions=actions.tolist(),
                                    raw_observation_file=str(raw_path.relative_to(args.output)),companion_ids=[c['request_id'] for c in selected]))
                                for action in actions[:policy.config.n_action_steps]:
                                    obs,reward,terminated,truncated,info=env.step(action.numpy());success=bool(info['is_success'])
                                    steps.append(dict(policy_call=call,action=action.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=success))
                                    trajectory.append(observation_evidence(env,obs)[0])
                                    if terminated or truncated or len(steps)>=520:break
                                if terminated or truncated:break
                            record=dict(task=task,initialization=initialization,condition=condition,calls=calls,steps=steps,trajectory=trajectory,success=success,
                                        gpu_uuid=gpu,candidate_lock_sha256=lock_hash,numerical_qa_acceptance_pending=True)
                            atomic_json(path,record,compressed=True);episodes[condition]=record
                            print(f'EPISODE {task}/{initialization}/{condition}: success={success}, controls={len(steps)}',flush=True)
                        finally:env.close()
                    if len({r['calls'][0]['target_input_sha256'] for r in episodes.values()})!=1:raise RuntimeError('Initial model input differs')
                    for name in lock['conditions']:
                        for field in ('calls','steps','trajectory','success'):
                            if not same(episodes[name][field],episodes[name+'_repeat'][field]):raise RuntimeError(f'A/A mismatch {task}/{initialization}/{name}/{field}')
                    a,b=episodes['patched_singleton'],episodes['patched_dynamic']
                    for field in ('steps','trajectory','success'):
                        if not same(a[field],b[field]):raise RuntimeError(f'Invariant batching changes {field}')
                    if [c['normalized_sha256'] for c in a['calls']]!=[c['normalized_sha256'] for c in b['calls']]:raise RuntimeError('Invariant normalized chunks differ')
                    a,b=episodes['native_singleton'],episodes['native_dynamic']
                    first=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if not same(x['action'],y['action'])),None)
                    if first is not None:
                        ca=a['calls'][a['steps'][first]['policy_call']];cb=b['calls'][b['steps'][first]['policy_call']]
                        if ca['target_input_sha256']!=cb['target_input_sha256'] or not same(a['trajectory'][:first+1],b['trajectory'][:first+1]):raise RuntimeError('Native first-action attribution failed')
                    summary=dict(task=task,initialization=initialization,gpu_uuid=gpu,candidate_lock_sha256=lock_hash,
                        exact_reset_and_AA_controls=True,invariant_trajectories_exact=True,native_first_action_divergence=first,
                        success={name:r['success'] for name,r in episodes.items()},numerical_qa_acceptance_pending=True)
                    atomic_json(folder/'block_summary.json',summary)
                    print(f'BLOCK {task}/{initialization}: {summary["success"]}',flush=True)


if __name__=='__main__':main()

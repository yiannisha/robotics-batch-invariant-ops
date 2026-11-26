"""Locked CogACT study with controlled inputs and released-default supplements.

Four tasks x50 starts; four fixed64 primary conditions and two released-native
conditions, all with repeats. Every episode cold-reconstructs its canonical
simulator state. Numerical/fidelity acceptance remains a separate gate.
"""
import argparse
import ast
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts.cogact_simpler_state_worker import RemoteStateEnv,encode
from scripts.qualify_simpler_fixed_action_replay import digest

TASKS=['pick_can','move_near','open_drawer','close_drawer']
PRIMARY=['native_singleton','native_dynamic','patched_singleton','patched_dynamic']
SUPPLEMENT=['released_native_singleton','released_native_dynamic']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def json_file(path,value,compressed=False):
    payload=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    atomic(path,gzip.compress(payload,mtime=0) if compressed else payload)


def retain(root,category,value):
    path=root/category/(digest(value)+'.pt.gz')
    if not path.exists():
        buffer=io.BytesIO();torch.save(value,buffer);atomic(path,gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
    return str(path.relative_to(root))


def td(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def policy_controller(state):
    return dict(policy_setup=state.policy_setup,sticky_action_is_on=state.sticky_action_is_on,
        gripper_action_repeat=state.gripper_action_repeat,sticky_gripper_action=encode(state.sticky_gripper_action),
        previous_gripper_action=encode(state.previous_gripper_action),
        ensemble=dict(action_history=[encode(v) for v in state.ensemble.action_history],maxlen=state.ensemble.action_history.maxlen,
                      adaptive_ensemble_alpha=state.ensemble.adaptive_ensemble_alpha))


def policy_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state().numpy(),torch_cuda=torch.cuda.get_rng_state().cpu().numpy())


def check_block(rows):
    names=PRIMARY+SUPPLEMENT
    if len({digest(r['trajectory'][0]) for r in rows.values()})!=1:raise RuntimeError('Canonical initial state differs')
    for name in names:
        for field in ('calls','steps','trajectory','success'):
            if digest(rows[name][field])!=digest(rows[name+'_repeat'][field]):raise RuntimeError(f'A/A differs: {name}/{field}')
    for group in (PRIMARY,SUPPLEMENT):
        if len({rows[n]['calls'][0]['target_input_sha256'] for n in group})!=1:raise RuntimeError('Initial model input differs within padding protocol')
    a,b=rows['patched_singleton'],rows['patched_dynamic']
    for field in ('steps','trajectory','success'):
        if digest(a[field])!=digest(b[field]):raise RuntimeError(f'Invariant batching changes {field}')
    if [c['normalized'] for c in a['calls']]!=[c['normalized'] for c in b['calls']]:raise RuntimeError('Invariant normalized chunks differ')
    result={}
    for prefix in ('','released_'):
        a,b=rows[prefix+'native_singleton'],rows[prefix+'native_dynamic']
        first=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if np.asarray(x['action'],dtype=np.float64).tobytes()!=np.asarray(y['action'],dtype=np.float64).tobytes()),None)
        equal_input=None
        if first is not None:
            equal_input=a['calls'][first]['target_input_sha256']==b['calls'][first]['target_input_sha256']
            if digest(a['trajectory'][:first+1])!=digest(b['trajectory'][:first+1]):raise RuntimeError('Divergence precedes first action difference')
            if not prefix and not equal_input:raise RuntimeError('Controlled-native divergence has different inputs')
        result[prefix+'native']=dict(first_action_divergence=first,identical_preprocessed_inputs_at_divergence=equal_input,
            interpretation='released_padding_extent_and_arithmetic_joint_effect' if prefix else 'controlled_identical_input_batch_arithmetic')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','source','openvla-source','llama-metadata','manifest','qa-runtime','qa-records','reset-audit',
                 'pizero-project','simpler-root','simpler-python','vulkan-icd','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--simulator-gpu-uuid',required=True);p.add_argument('--tasks',nargs='+',default=TASKS)
    args=p.parse_args()
    if not args.tasks or len(set(args.tasks))!=len(args.tasks) or not set(args.tasks)<=set(TASKS):raise ValueError('Invalid task shard')
    reset=json.loads(args.reset_audit.read_text())
    if reset.get('status')!='complete_observable_state_fixed_replay_independently_verified' or set(reset['tasks'])!=set(TASKS) or reset['episodes']!=36:raise ValueError('Complete expanded-state reset audit required')
    os.environ.update(COGACT_CHECKPOINT=str(args.checkpoint),COGACT_SOURCE=str(args.source),COGACT_OPENVLA_SOURCE=str(args.openvla_source),
        COGACT_LLAMA_METADATA=str(args.llama_metadata),COGACT_SIMPLER_EXACT_MANIFEST=str(args.manifest))
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters.cogact_simpler_exact import CogACTSimplerExactAdapter
    from scripts.model_invariance.adapters import cogact_small as implementation
    from scripts.run_cogact_paired_server import RequestState
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    qa=json.loads(args.qa_runtime.read_text())
    def verify_qa():
        for file,expected in qa['source_sha256'].items():
            if sha(file)!=expected:raise RuntimeError(f'QA source changed: {file}')
    verify_qa();adapter=CogACTSimplerExactAdapter();policy=adapter.load_model();policy.eval()
    from sim_cogact.adaptive_ensemble import AdaptiveEnsembler
    if sha(args.manifest)!=qa['replay_manifest_sha256']:raise ValueError('QA manifest differs')
    manifest=adapter.replay_manifest
    companions=[dict(request=adapter.load_replay_input(row['replay_index'],0),task=row['task'],request_id=row['request_ids'][0]) for row in manifest['observations'] if row['split']=='diagnostic']
    if len(companions)!=200:raise ValueError('Expected 200 diagnostic companions')
    def infer(samples,invariant,released=False,position=0):
        inputs={}
        def trace(label,value):
            if label.startswith('input.') or label=='diffusion.noise':inputs[label]=td(value[position:position+1])
        before=digest(policy_rng());adapter.trace_callback=trace
        try:
            # eta=0 DDIM still draws unused auxiliary noise. Preserve ambient
            # CPU/CUDA RNG so batch-sized unused draws cannot leak to a future
            # request; explicit indexed initial noise remains the model input.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.inference_mode(),set_batch_invariant_mode(invariant):
                batch=adapter.compose_released_batch(samples) if released else adapter.compose_fixed_batch(samples)
                out=adapter.run_model(policy,batch);out={k:v.detach().cpu() for k,v in out.items()}
        finally:adapter.trace_callback=None
        if digest(policy_rng())!=before:raise RuntimeError('Inference changed ambient RNG')
        if not inputs or any(not torch.isfinite(v).all() for v in out.values()):raise RuntimeError('Missing input trace/nonfinite policy output')
        return out,inputs
    signatures={}
    with gzip.open(args.qa_records,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['request_id']==companions[0]['request_id'] and row['transformation']=='singleton':signatures[row['implementation']]={k:v['candidate_sha256'] for k,v in row['output']['tensors'].items()}
            if len(signatures)==2:break
    if set(signatures)!={'native','batch_invariant'}:raise ValueError('Missing QA singleton signatures')
    for name,invariant in [('native',False),('batch_invariant',True)]:
        out,_=infer([companions[0]['request']],invariant)
        if {f'output.{k}':td(v)['sha256'] for k,v in out.items()}!=signatures[name]:raise RuntimeError('Actual GPU/RNG-scoped path differs from QA singleton bytes')
        for size in (2,4,8):infer([c['request'] for c in companions[:size]],invariant)
    def official_batch_equivalence(requests,physical):
        original_randn=torch.randn;consumed=[]
        geometry_expected=(len(requests),16,7);before=digest(policy_rng())
        def explicit(*shape,**kwargs):
            geometry=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(tuple,list)) else tuple(shape)
            if geometry==geometry_expected:
                consumed.append(geometry);return torch.cat([r['noise'] for r in requests],dim=0).to(device=kwargs.get('device','cpu'),dtype=kwargs.get('dtype',torch.float32)).clone()
            return original_randn(*shape,**kwargs)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.inference_mode(),set_batch_invariant_mode(False),patch.object(torch,'randn',side_effect=explicit):
            result,_=policy.model.predict_action_batch(image=[r['image'] for r in requests],instruction=[r['instruction'] for r in requests],unnorm_key=implementation.UNNORM_KEY,
                cfg_scale=implementation.CFG_SCALE,use_ddim=True,num_ddim_steps=implementation.DDIM_STEPS,do_sample=False)
        if digest(policy_rng())!=before:raise RuntimeError('Official sampler changed ambient RNG')
        if consumed!=[geometry_expected] or np.asarray(result).shape!=tuple(physical.shape) or np.asarray(result).tobytes()!=physical.numpy().tobytes():raise RuntimeError('Released wrapper differs from official batch sampler')
        return dict(exact=True,indexed_initial_noise_consumed_once=True,physical_sha256=hashlib.sha256(np.asarray(result).tobytes()).hexdigest())
    official_checks={}
    for task in TASKS:
        request=next(c['request'] for c in companions if c['task']==task)
        out,_=infer([request],False,released=True);official_checks[task]=official_batch_equivalence([request],out['physical'])
    mixed=[([c['request'] for c in companions if c['task']==task])[j*7] for j in range(2) for task in TASKS]
    out,_=infer(mixed,False,released=True);official_checks['mixed_task_B8']=official_batch_equivalence(mixed,out['physical'])
    for name,invariant in [('native',False),('batch_invariant',True)]:
        out,_=infer([companions[0]['request']],invariant)
        if {f'output.{k}':td(v)['sha256'] for k,v in out.items()}!=signatures[name]:raise RuntimeError('Official sampler reinitialization changes QA singleton bytes')
    tree=ast.parse((args.pizero_project/'experiments/prepare_replay.py').read_text())
    mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASKS' for t in n.targets))
    qualified={}
    reset_runtime=json.loads((args.reset_audit.parent/'runtime.json').read_text())
    for file,expected in reset_runtime['source_sha256'].items():
        if sha(file)!=expected:raise ValueError('Simulator instrumentation changed after qualification')
    for line in (args.reset_audit.parent/'replays.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['repeat']==0:qualified[(row['task'],row['initialization'])]=row['canonical']
    if sha(args.reset_audit.parent/'replays.jsonl')!=reset['raw_replays_sha256']:raise ValueError('Qualified reset evidence changed')
    source=dict(qa['source_sha256']);source.update(reset_runtime['source_sha256'])
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file() and name.startswith(('transformers','prismatic','vla','sim_cogact','batch_invariant_ops',
                'scripts.run_cogact_paired_server','scripts.cogact_simpler_state_worker','scripts.qualify_simpler_fixed_action_replay')):source[str(Path(path).resolve())]=sha(path)
    source[str(Path(__file__).resolve())]=sha(Path(__file__))
    source[str((args.pizero_project/'experiments/prepare_replay.py').resolve())]=sha(args.pizero_project/'experiments/prepare_replay.py')
    for root in (args.simpler_root,):
        for file in root.rglob('*.py'):
            if '.git' not in file.parts:source[str(file.resolve())]=sha(file)
    assets={}
    acquisition=json.loads((args.checkpoint/'acquisition.json').read_text())
    if acquisition['revision']!='ca302a300542393a0b06e291465076a7d9105c23':raise ValueError('Foreign checkpoint revision')
    for row in acquisition['files']:
        file=args.checkpoint/row['path']
        if sha(file)!=row['sha256']:raise ValueError('Changed checkpoint asset')
        assets[str(file.resolve())]=row['sha256']
    for file in args.llama_metadata.rglob('*'):
        if file.is_file() and '.cache' not in file.parts:assets[str(file.resolve())]=sha(file)
    simulator_asset_roots=[args.simpler_root/'ManiSkill2_real2sim/mani_skill2_real2sim/assets',args.simpler_root/'ManiSkill2_real2sim/data']
    for key in ('MS2_REAL2SIM_ASSET_DIR','MS2_ASSET_DIR'):
        if os.environ.get(key):simulator_asset_roots.append(Path(os.environ[key]))
    for root in simulator_asset_roots:
        if root.exists():
            for file in root.rglob('*'):
                if file.is_file() and '.git' not in file.parts and '.cache' not in file.parts and '__pycache__' not in file.parts:assets[str(file.resolve())]=sha(file)
    args.output.mkdir(parents=True,exist_ok=True)
    lock=dict(scope='paired_candidate_source_lock_not_final_numerical_acceptance',model='cogact_simpler_exact',global_tasks=TASKS,initialization_ids=list(range(50)),
        primary_conditions=PRIMARY,supplemental_released_native_conditions=SUPPLEMENT,repeat_controls=True,dynamic_cycle=[1,2,4,8],
        source_sha256=source,asset_sha256=assets,replay_manifest_path=str(args.manifest.resolve()),replay_manifest_sha256=sha(args.manifest),
        qa_runtime_sha256=sha(args.qa_runtime),qa_singleton_signatures=signatures,reset_audit_sha256=sha(args.reset_audit),
        reset_protocol=reset['reset_protocol'],simulator_gpu_uuid=args.simulator_gpu_uuid,primary_language_tokens=64,
        environment_mapping={task:mapping[task] for task in TASKS},
        primary_native_is_controlled_fixed64=True,released_native_uses_batch_longest_padding=True,
        ambient_torch_rng_preserved_per_inference=True,official_released_singleton_checks=official_checks,
        torch=torch.__version__,gpu_type=torch.cuda.get_device_name(),numerical_qa_acceptance_pending=True)
    with (args.output/'study.lock').open('w') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX);path=args.output/'candidate_lock.json'
        if path.exists():
            if json.loads(path.read_text())!=lock:raise RuntimeError('Candidate lock differs')
        else:
            json_file(path,lock)
            for file,expected in source.items():atomic(args.output/'source_snapshots'/expected,Path(file).read_bytes())
    lock_hash=sha(args.output/'candidate_lock.json');gpu=str(torch.cuda.get_device_properties(0).uuid)
    asset_stats={file:(Path(file).stat().st_size,Path(file).stat().st_mtime_ns) for file in assets}
    conditions={name:(name.startswith('patched'),name.endswith('dynamic'),name.startswith('released_')) for name in PRIMARY+SUPPLEMENT}
    conditions.update({name+'_repeat':value for name,value in list(conditions.items())})
    original_popen=subprocess.Popen
    def simulator_spawn(*positional,**kwargs):
        kwargs['env']=dict(kwargs['env'],CUDA_VISIBLE_DEVICES=args.simulator_gpu_uuid)
        return original_popen(*positional,**kwargs)
    with (args.output/f'worker-{gpu}.lock').open('w') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        json_file(args.output/f'worker-{gpu}.json',dict(gpu_uuid=gpu,tasks=args.tasks,candidate_lock_sha256=lock_hash))
        for task in args.tasks:
            for initial in range(50):
                folder=args.output/'blocks'/task/f'start-{initial:02d}';folder.mkdir(parents=True,exist_ok=True)
                with (folder/'block.lock').open('w') as guard:
                    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if (folder/'block_summary.json').exists():continue
                    owner=folder/'owner.json'
                    if owner.exists() and json.loads(owner.read_text())['gpu_uuid']!=gpu:continue
                    if not owner.exists():json_file(owner,dict(gpu_uuid=gpu,candidate_lock_sha256=lock_hash))
                    verify_qa()
                    for file,expected in source.items():
                        if sha(file)!=expected:raise RuntimeError(f'Locked source changed: {file}')
                    for file,expected in assets.items():
                        stat=(Path(file).stat().st_size,Path(file).stat().st_mtime_ns)
                        if stat!=asset_stats[file]:
                            if sha(file)!=expected:raise RuntimeError(f'Locked asset changed: {file}')
                            asset_stats[file]=stat
                    rows={}
                    for condition,(invariant,dynamic,released) in conditions.items():
                        path=folder/(condition+'.json.gz')
                        if path.exists():
                            with gzip.open(path,'rt') as f:row=json.load(f)
                            if row['gpu_uuid']!=gpu or row['candidate_lock_sha256']!=lock_hash:raise RuntimeError('Foreign committed episode')
                            rows[condition]=row;continue
                        seed=20261108+initial;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        with patch('scripts.cogact_simpler_state_worker.subprocess.Popen',side_effect=simulator_spawn):
                            env=RemoteStateEnv(mapping[task],args.simpler_python,args.simpler_root,args.pizero_project/'experiments/simpler_worker.py',args.vulkan_icd,
                                folder/(condition+'-worker.log'),process_seed=seed)
                        try:
                            env.request(dict(command='reset',seed=seed,options={'obj_init_options':{'episode_id':initial}}))
                            captured=env.request(dict(command='capture_snapshot'));observable=captured['observable']
                            if (task,initial) in qualified and observable['hashes']!=qualified[(task,initial)]:raise RuntimeError('Initial simulator state differs from qualified canonical snapshot')
                            controller=RequestState('google_robot',AdaptiveEnsembler);trajectory=[];calls=[];steps=[];success=False;truncated=False;official=None
                            def frame(value):
                                raw=dict(value['raw'],policy_controller=policy_controller(controller),policy_rng=policy_rng())
                                trajectory.append(dict({k+'_sha256':digest(v) for k,v in raw.items()},raw_frame=retain(args.output,'frames',raw)))
                            frame(observable)
                            if rows and digest(trajectory[0])!=digest(next(iter(rows.values()))['trajectory'][0]):raise RuntimeError('Canonical episode state differs')
                            while not truncated:
                                call=len(calls)
                                if call>=200:raise RuntimeError('Simulator exceeded declared horizon')
                                obs=observable['raw']['observation'];rgb=next(iter(obs['image'].values()))['rgb'];instruction=observable['raw']['instruction']
                                noise_id=f'{task}/{initial}/policy-call-{call}';noise_seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
                                noise=torch.randn(1,16,7,generator=torch.Generator().manual_seed(noise_seed))
                                target=dict(image=Image.fromarray(rgb),instruction=instruction,noise=noise)
                                size=[1,2,4,8][call%4] if dynamic else 1;position=call%size;pool=[c for c in companions if c['task']!=task]
                                selected=[pool[(initial*97+call*11+j*17)%len(pool)] for j in range(size-1)]
                                samples=[c['request'] for c in selected];samples.insert(position,target)
                                out,inputs=infer(samples,invariant,released,position);normalized=out['normalized'][position:position+1];actions=out['physical'][position].numpy()
                                if released and not dynamic and call==0:official=official_batch_equivalence([target],out['physical'])
                                selected_action,action=controller.process(actions)
                                request=dict(rgb=rgb,instruction=instruction,noise=noise.numpy())
                                calls.append(dict(call=call,batch_size=size,position=position,noise_id=noise_id,noise_seed=noise_seed,
                                    target_input_sha256=digest(inputs),model_inputs=inputs,raw_request=retain(args.output,'requests',request),
                                    companion_ids=[c['request_id'] for c in selected],normalized=td(normalized),normalized_actions=normalized.tolist(),
                                    raw_action_chunk=actions.tolist(),selected_action=selected_action.tolist(),environment_action=action.tolist()))
                                result=env.request(dict(command='step_with_state',action=action.tolist()));observable=result['observable']
                                success=bool(result['success']);truncated=bool(result['truncated'])
                                steps.append(dict(policy_call=call,action=action.tolist(),reward=float(result['reward']),success=success,truncated=truncated,info=result['info']))
                                frame(observable)
                            row=dict(task=task,initialization=initial,condition=condition,gpu_uuid=gpu,candidate_lock_sha256=lock_hash,
                                trajectory=trajectory,calls=calls,steps=steps,success=success,released_default=released,official_singleton_equivalence=official,numerical_qa_acceptance_pending=True)
                            json_file(path,row,compressed=True);rows[condition]=row
                            print(f'EPISODE {task}/{initial}/{condition}: success={success}, controls={len(steps)}',flush=True)
                        finally:env.close()
                    attribution=check_block(rows)
                    summary=dict(task=task,initialization=initial,gpu_uuid=gpu,candidate_lock_sha256=lock_hash,attribution=attribution,
                        success={name:row['success'] for name,row in rows.items()},exact_reset_and_AA_controls=True,invariant_trajectories_exact=True,numerical_qa_acceptance_pending=True)
                    json_file(folder/'block_summary.json',summary);print(f'BLOCK {task}/{initial}: {summary["success"]}',flush=True)


if __name__=='__main__':main()

"""Collect the full50-task LingBot native corpus through qualified simulators.

Four episodes/task: two diagnostic starts beginning100000 and two prospective
heldout starts beginning200000, each using the first released expert-eligible
seed and100 unseen instruction candidates. Select2 diagnostic and up to8 heldout
pre-action frames per episode. Short episodes retain fewer distinct frames.
No invariant inference or numerical-heldout acceptance occurs during collection.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp');temp.write_bytes(data);temp.replace(path)


def write_json(path,value):atomic(path,(json.dumps(value,indent=2)+'\n').encode())


def frame_indices(steps,split):
    require(steps>0,'Episode has no pre-action observations')
    # Exclude the already familiar reset image from prospective heldout when
    # an alternative pre-action observation exists. No output/error filtering.
    first=1 if split=='heldout' and steps>1 else 0
    count=min(8 if split=='heldout' else 2,steps-first)
    return np.linspace(first,steps-1,count,dtype=int).tolist()


def tensor_signature(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def content_digest(value):
    def encode(item):
        if isinstance(item,torch.Tensor):return {'torch_tensor_signature':tensor_signature(item)}
        if isinstance(item,dict):return {key:encode(v) for key,v in item.items()}
        if isinstance(item,(tuple,list)):return [encode(v) for v in item]
        return item
    return digest(encode(value))


def retain(root,category,value):
    identity=content_digest(value);path=root/category/(identity+'.pt.gz')
    if not path.exists():
        import io
        buffer=io.BytesIO();torch.save(value,buffer);atomic(path,gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
    return str(path.relative_to(root))


def qualified_worker(entry):
    directory=Path(entry['directory']);audit=directory/'independent_audit.json'
    require(sha(audit)==entry['independent_audit_sha256'],'Task reset audit changed')
    report=json.loads(audit.read_text())
    require(report['initial_state_and_full_trajectories_exact'] and report['cpu_and_cuda_rng_verified'],'Task reset qualification incomplete')
    require(report['source_defined_task_state_frames_verified']==report['raw_frames_verified'],'Task fields were not covered')
    require(sha(directory/'replays.jsonl')==report['raw_replays_sha256'],'Qualified replay evidence changed')
    runtime=json.loads((directory/'runtime.json').read_text());sources=runtime['source_sha256']
    for path,expected in sources.items():require(sha(Path(path))==expected,'Qualified task execution source changed')
    worker=next(Path(path) for path in sources if Path(path).name=='robotwin_exact_worker.py')
    return worker,sources


class Worker:
    def __init__(self,python,worker,source,folder):
        self.folder=folder;folder.mkdir(parents=True,exist_ok=False)
        self.stderr=(folder/'stderr.log').open('w');self.stdout=(folder/'stdout.log').open('w')
        command=[str(python),str(worker),'--source',str(source),'--output',str(folder/'temporary_frames'),
            '--capture-cuda-rng','--capture-source-task-state']
        self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
    def rpc(self,request,allow_unstable=False):
        self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush()
        for line in self.proc.stdout:
            if 'ROBOTWIN_RESULT ' in line:
                response=json.loads(line.split('ROBOTWIN_RESULT ',1)[1])
                if not response['ok'] and not (allow_unstable and response.get('error_type')=='UnStableError'):
                    raise RuntimeError(response.get('error','Worker request failed'))
                return response
            self.stdout.write(line);self.stdout.flush()
        raise RuntimeError('Simulator worker terminated')
    def close(self):
        self.proc.stdin.close();code=self.proc.wait();self.stderr.close();self.stdout.close()
        require(code==0,'Simulator worker failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','qwen-metadata','robotwin-source','robotwin-python','qualification-union','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    require(not args.output.exists() or args.resume,'Explicit resume required')
    args.output.mkdir(parents=True,exist_ok=True);q_bytes=args.qualification_union.read_bytes();q=json.loads(q_bytes)
    require(len(q['prescribed_tasks'])==50 and q['model_evaluated'] is False,'Wrong released task roster')
    write_json(args.output/'qualification_snapshots'/(hashlib.sha256(q_bytes).hexdigest()+'.json'),q)
    asset_root=args.checkpoint.parents[2];acquisition_path=asset_root/'acquisition.json'
    acquisition=json.loads(acquisition_path.read_text())
    require(acquisition['revision']=='0451855729ec904f970600e0aec8b84661423afe','Wrong fine-tuned checkpoint revision')
    for record in acquisition['files']:
        path=asset_root/record['path'];require(path.is_file() and sha(path)==record['sha256'],'Pinned checkpoint asset changed: '+str(path))
    os.environ.update(QWEN3VL_PATH=str(args.qwen_metadata),LINGBOT_VLA2_CHECKPOINT=str(args.checkpoint))
    sys.path.insert(0,str(args.source))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from scripts.model_invariance.adapters.lingbot_vla2 import LingBotVLA2Adapter
    from batch_invariant_ops import set_batch_invariant_mode
    from safetensors import safe_open
    from unittest.mock import patch
    class Adapter(LingBotVLA2Adapter):
        def _validate_paths(self):
            for path in (self.checkpoint/'model.safetensors.index.json',self.checkpoint.parents[2]/'lingbotvla_cli.yaml'):
                require(path.is_file(),'Pinned LingBot asset missing: '+str(path))
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    adapter=Adapter();load_dtypes={};original=LingbotVLAv2Server.load_model_weights
    def observe_load(server,*values,**kwargs):
        result=original(server,*values,**kwargs);load_dtypes.update({name:value.dtype for name,value in server.vla.state_dict().items()});return result
    with patch.object(LingbotVLAv2Server,'load_model_weights',observe_load):model=adapter.load_model()
    state=adapter.server.vla.state_dict();seen=set();elements=0
    for path in sorted(args.checkpoint.glob('*.safetensors')):
        with safe_open(path,framework='pt',device='cpu') as shard:
            for name in shard.keys():
                require(name not in seen and name in state,'Unexpected checkpoint tensor')
                actual=state[name].detach().cpu().contiguous();expected=shard.get_tensor(name).to(load_dtypes[name]).to(actual.dtype).contiguous()
                require(torch.equal(actual.reshape(-1).view(torch.uint8),expected.reshape(-1).view(torch.uint8)),'Strict loaded tensor mismatch: '+name)
                seen.add(name);elements+=actual.numel()
    require(seen==set(state) and len(seen)==1708 and elements==6375907511,'Official checkpoint coverage differs')
    del state,actual,expected
    adapter.server.use_length=50
    core_files=[Path(__file__),Path(sys.modules[LingBotVLA2Adapter.__module__].__file__),Path(__file__).with_name('qualify_simpler_fixed_action_replay.py')]
    core_files+=list((Path(__file__).resolve().parents[1]/'batch_invariant_ops').rglob('*.py'))
    core_sources={str(path.resolve()):sha(path) for path in core_files}
    lock=dict(scope='native_source_corpus_for_all50_tasks',prescribed_tasks=q['prescribed_tasks'],episodes_per_task=4,
        phases={'diagnostic':dict(seed_start=100000,episodes=2,frames_per_episode=2),'heldout':dict(seed_start=200000,episodes=2,frames_per_episode=8)},
        source_sha256=core_sources,checkpoint_files={str(path.resolve()):sha(path) for path in args.checkpoint.glob('*.safetensors')},
        acquisition_sha256=sha(acquisition_path),qwen_metadata_files={str(path.resolve()):sha(path) for path in args.qwen_metadata.rglob('*') if path.is_file()},
        strict_loaded_tensors=1708,strict_loaded_elements=elements,official_dtype_sequence=True,
        native_contract='FP32 released path with deterministic token-choice MoE shared by primary native/invariant; atomic native baseline separate',
        selection='evenly spaced distinct pre-action frames; prospective heldout skips reset frame when possible; short episodes retain fewer frames',
        numerical_heldout_acceptance_pending=True,batch_invariant_model_evaluated=False)
    lock_path=args.output/'candidate_lock.json'
    if lock_path.exists():require(json.loads(lock_path.read_text())==lock,'Cannot resume with another source/model/collection protocol')
    else:write_json(lock_path,lock)
    lock_sha=sha(lock_path)
    for path,identity in core_sources.items():atomic(args.output/'source_snapshots'/identity,Path(path).read_bytes())
    write_json(args.output/'startup_weight_audit.json',dict(status='all1708_loaded_tensors_byte_exact_after_official_dtype_sequence',elements=elements,gpu_uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)))
    manifest_path=args.output/'collection_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(model_id='lingbot_robotwin_campaign',candidate_lock_sha256=lock_sha,
        prescribed_tasks=q['prescribed_tasks'],prescribed_episodes=200,episodes=[],observations=[],status='partial_pending_independent_corpus_audit')
    require(manifest['candidate_lock_sha256']==lock_sha,'Collection manifest lock changed')
    completed={(e['task'],e['split'],e['ordinal']) for e in manifest['episodes']}
    def read_frame(frame):
        path=Path(frame['path']).resolve();require(path.is_relative_to(args.output.resolve()),'Worker frame path escapes collection')
        value=torch.load(path,map_location='cpu',weights_only=False)
        require(digest(value['observation'])==frame['observation_sha256'] and digest(value['physics'])==frame['physics_sha256'] and digest(value['rng'])==frame['rng_sha256'] and digest(value['task_state'])==frame['task_state_sha256'],'Worker raw frame changed')
        require(value['controller']==frame['controller'],'Worker controller evidence differs')
        stored=retain(args.output,'frames',value);path.unlink()
        return {k:v for k,v in frame.items() if k!='path'}|{'raw_frame':stored},value['observation']
    def prepare(obs,instruction):
        return adapter.server._prepare_model_input({'task':instruction,'observation.state':obs['joint_action']['vector'],
            'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
            'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
            'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb']})
    for task in q['prescribed_tasks']:
        if task not in q['qualified_tasks']:continue
        entry=q['qualified_tasks'][task];worker,sources=qualified_worker(entry)
        for split,start in [('diagnostic',100000),('heldout',200000)]:
            previous=None
            for ordinal in range(2):
                key=(task,split,ordinal)
                if key in completed:
                    previous=next(e['seed'] for e in manifest['episodes'] if (e['task'],e['split'],e['ordinal'])==key);continue
                require(all(sha(Path(path))==value for path,value in (core_sources|sources).items()),'Execution source changed during collection')
                folder=args.output/'episodes'/task/f'{split}-{ordinal}';sim=Worker(args.robotwin_python,worker,args.robotwin_source,folder)
                try:
                    selection=None;first=start if previous is None else previous+1
                    for seed in range(first,first+100):
                        request=dict(op='expert',task=task,seed=seed,test_num=100,instruction_type='unseen')
                        response=sim.rpc(request,allow_unstable=True)
                        with (folder/'expert_seeds.jsonl').open('a') as stream:stream.write(json.dumps(dict(request=request,response=response))+'\n')
                        if response['ok'] and response['result']['eligible']:selection=response['result'];break
                    require(selection is not None,'No expert-eligible seed in predeclared100 candidates')
                    canonical,_=read_frame(selection['canonical'])
                    selection['canonical']=canonical
                    frame=sim.rpc(dict(request,op='reset',episode_info=selection['episode_info'],instruction=selection['instruction']))['result']
                    frame,obs=read_frame(frame);require(frame==canonical,'Native episode canonical reset differs from expert selection')
                    adapter.server.global_step=0;adapter.server.last_action_chunk=None;adapter.server.last_normalized_action_chunk=None
                    trajectory=[frame];calls=[];steps=[];instruction=selection['instruction']
                    while not frame['controller']['eval_success'] and frame['controller']['take_action_count']<frame['controller']['step_limit']:
                        call=len(calls);transformed=prepare(obs,instruction);noise_id=f'{task}/{seed}/policy-call-{call}'
                        noise_seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
                        target={name:transformed[name] for name in adapter.model_keys};target['noise']=torch.randn(50,55,generator=torch.Generator().manual_seed(noise_seed))
                        adapter.reset_model(model)
                        with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]),set_batch_invariant_mode(False):
                            normalized=adapter.run_model(model,adapter.compose_batch([target]));physical=adapter.server._unapply_batched_actions([transformed],normalized)
                        chunk=np.asarray(physical['action'][0],dtype=np.float32);require(chunk.shape==(50,14) and np.isfinite(chunk).all(),'Invalid official physical action chunk')
                        call_record=dict(call=call,noise_id=noise_id,noise_seed=noise_seed,frame_index=len(steps),
                            target_input_signatures={k:tensor_signature(v) for k,v in target.items()},raw_request=retain(args.output,'requests',target),
                            normalized_output=retain(args.output,'outputs',normalized.detach().cpu()),physical_chunk=chunk.tolist())
                        calls.append(call_record)
                        for action in chunk:
                            response=sim.rpc(dict(op='step',action=action.tolist()))['result'];frame,obs=read_frame(response)
                            steps.append(dict(call=call,action=action.tolist()));trajectory.append(frame)
                            if frame['controller']['eval_success'] or frame['controller']['take_action_count']>=frame['controller']['step_limit']:break
                    require(steps,'Native episode contains no controls')
                    episode=dict(task=task,split=split,ordinal=ordinal,seed=seed,instruction=instruction,selection=selection,
                        candidate_lock_sha256=lock_sha,qualification_audit_sha256=entry['independent_audit_sha256'],worker_source_sha256=sources,
                        trajectory=trajectory,calls=calls,steps=steps,success=frame['controller']['eval_success'])
                    episode_path=folder/'episode.json.gz';atomic(episode_path,gzip.compress(json.dumps(episode).encode(),compresslevel=1,mtime=0))
                    chosen=frame_indices(len(steps),split)
                    for frame_index in chosen:
                        source_frame=trajectory[frame_index]
                        with gzip.open(args.output/source_frame['raw_frame'],'rb') as stream:raw=torch.load(stream,map_location='cpu',weights_only=False)
                        transformed=prepare(raw['observation'],instruction);request_value={name:transformed[name] for name in adapter.model_keys}
                        stored=retain(args.output,'selected_requests',request_value);index=len(manifest['observations'])
                        manifest['observations'].append(dict(replay_index=index,task=task,episode_id=f'{task}/{seed}',seed=seed,split=split,
                            observation_index=chosen.index(frame_index),frame_index=frame_index,raw_frame=source_frame['raw_frame'],request_file=stored,
                            input_signatures={k:tensor_signature(v) for k,v in request_value.items()},source_episode_file=str(episode_path.relative_to(args.output))))
                    manifest['episodes'].append(dict(task=task,split=split,ordinal=ordinal,seed=seed,success=episode['success'],
                        source_episode_file=str(episode_path.relative_to(args.output)),sha256=sha(episode_path),selected_observations=len(chosen)))
                    completed.add(key);previous=seed;write_json(manifest_path,manifest)
                    print(f'COLLECTED {task}/{split}/{ordinal}: seed={seed}, controls={len(steps)}, selected={len(chosen)}, episodes={len(completed)}/200',flush=True)
                finally:sim.close()
    manifest['all_prescribed_native_episodes_collected']=len(completed)==200;write_json(manifest_path,manifest)
    print(f'Native episodes={len(completed)}/200; observations={len(manifest["observations"])}; independent corpus/holdout audits pending',flush=True)


if __name__=='__main__':main()

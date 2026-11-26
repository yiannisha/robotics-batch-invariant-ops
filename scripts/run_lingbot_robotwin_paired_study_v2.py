"""Varied-start paired LingBot episodes across all qualified RoboTwin tasks.

Requires the complete native corpus and independently accepted diagnostics.
Recording blocks does not certify the numerical or simulator campaign complete.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.collect_lingbot_robotwin_campaign_replay_v2 import (
    Worker,qualified_worker,retain,sha,tensor_signature,write_json,atomic,require,
)
from scripts.qualify_simpler_fixed_action_replay import digest

CONDITIONS=('native_singleton','native_dynamic','patched_singleton','patched_dynamic')
DYNAMIC_CYCLE=(1,2,4,8)


def call_plan(task,seed,call,dynamic,companion_rows):
    """Request-indexed noise and immutable unrelated-task companion identities."""
    key=f'{task}/{seed}/policy-call-{call}'
    noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
    size=DYNAMIC_CYCLE[call%len(DYNAMIC_CYCLE)] if dynamic else 1
    candidates=[r for r in companion_rows if r['task']!=task]
    require(len(candidates)>=7,'Insufficient unrelated-task diagnostic companions')
    selected=[];used=set()
    for slot in range(size-1):
        index=(seed*97+call*11+slot*17)%len(candidates)
        while index in used:index=(index+1)%len(candidates)
        used.add(index);selected.append(candidates[index])
    return dict(noise_id=key,noise_seed=noise_seed,batch_size=size,target_position=call%size,
        companion_replay_indices=[r['replay_index'] for r in selected])


def compare_episodes(episodes):
    for name in CONDITIONS:
        require(all(digest(episodes[name][field])==digest(episodes[name+'_repeat'][field])
            for field in ('calls','steps','trajectory','success')),'Same-condition repeat differs: '+name)
    starts=[e['trajectory'][0] for e in episodes.values()]
    require(len({digest(v) for v in starts})==1,'Initial observable state/input differs')
    require(len({e['calls'][0]['target_input_sha256'] for e in episodes.values()})==1,'Initial neural input differs')
    a,b=episodes['patched_singleton'],episodes['patched_dynamic']
    require(digest(a['trajectory'])==digest(b['trajectory']) and digest(a['steps'])==digest(b['steps']),'Invariant trajectories differ')
    require([(c['normalized_sha256'],c['physical_sha256']) for c in a['calls']]==
        [(c['normalized_sha256'],c['physical_sha256']) for c in b['calls']],'Invariant chunks differ')
    a,b=episodes['native_singleton'],episodes['native_dynamic']
    first=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if
        np.asarray(x['action'],dtype=np.float32).tobytes()!=np.asarray(y['action'],dtype=np.float32).tobytes()),None)
    if first is not None:
        require(digest(a['trajectory'][:first+1])==digest(b['trajectory'][:first+1]),'Native divergence lacks identical physical prefix')
        ac=a['calls'][a['steps'][first]['policy_call']];bc=b['calls'][b['steps'][first]['policy_call']]
        require(ac['target_input_sha256']==bc['target_input_sha256'],'Native first action divergence lacks equal-input attribution')
    return dict(all_same_condition_repeats_exact=True,invariant_actions_and_trajectories_exact=True,native_first_action_divergence=first)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','qwen-metadata','robotwin-source','robotwin-python','qualification-union',
                 'manifest','qa-runtime','qa-records','qa-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    require(not a.output.exists() or a.resume,'Explicit resume required')
    q=json.loads(a.qualification_union.read_text());tasks=q['prescribed_tasks']
    require(len(tasks)==50 and len(set(tasks))==50 and set(q['qualified_tasks'])==set(tasks),'All50 qualified tasks required')
    manifest=json.loads(a.manifest.read_text());qa=json.loads(a.qa_audit.read_text());runtime=json.loads(a.qa_runtime.read_text())
    require(manifest['model_id']=='lingbot_robotwin_exact','Wrong exact replay model')
    require(qa['status']=='complete_four_model_diagnostic_independently_verified' and qa['model_id']=='lingbot_robotwin_exact','Complete independent model diagnostics required')
    require(qa['manifest_sha256']==sha(a.manifest) and qa['runtime_sha256']==sha(a.qa_runtime) and qa['diagnostic_sha256']==sha(a.qa_records),'Diagnostic proof bytes differ')
    require(qa['invariant_output_and_all_recorded_stages_exact'] is True,'Invariant diagnostic failed')
    require(runtime['adapter']=='scripts.model_invariance.adapters.lingbot_robotwin_exact_v2' and runtime['split']=='diagnostic','Wrong diagnostic adapter')
    corpus_audit_path=Path(manifest['corpus_audit_path'])
    require(sha(corpus_audit_path)==manifest['corpus_audit_sha256'],'Corpus audit changed')
    corpus_audit=json.loads(corpus_audit_path.read_text());corpus_lock_path=Path(manifest['fixture_root'])/'candidate_lock.json'
    require(sha(corpus_lock_path)==corpus_audit['candidate_lock_sha256'],'Source corpus checkpoint identity changed')
    corpus_lock=json.loads(corpus_lock_path.read_text())
    require(set(corpus_lock['checkpoint_files'])=={str(path.resolve()) for path in a.checkpoint.glob('*.safetensors')},'Checkpoint path differs from audited source')
    require(set(corpus_lock['qwen_metadata_files'])=={str(path.resolve()) for path in a.qwen_metadata.rglob('*') if path.is_file()},'Tokenizer metadata path differs')
    acquisition_path=a.checkpoint.parents[2]/'acquisition.json'
    assets={**corpus_lock['checkpoint_files'],**corpus_lock['qwen_metadata_files'],str(acquisition_path.resolve()):corpus_lock['acquisition_sha256']}
    for path,identity in assets.items():require(sha(Path(path))==identity,'Checkpoint/tokenizer differs from audited source')
    source_hashes={str(Path(__file__).resolve()):sha(Path(__file__))}
    for path,identity in runtime['source_sha256'].items():require(sha(Path(path))==identity,'Model source changed');source_hashes[path]=identity
    from scripts import collect_lingbot_robotwin_campaign_replay_v2 as collector
    source_hashes[str(Path(collector.__file__).resolve())]=sha(Path(collector.__file__))
    workers={}
    from scripts import lingbot_rigid_qualification as rigid_contract
    from scripts import audit_robotwin_benchmark_replay_v5 as rigid_auditor
    for helper in (rigid_contract, rigid_auditor):
        source_hashes[str(Path(helper.__file__).resolve())]=sha(Path(helper.__file__))
    for task in tasks:
        proof=json.loads((Path(q['qualified_tasks'][task]['directory'])/'independent_audit.json').read_text())
        rigid_contract.require_rigid_replay_proof(proof)
        worker,sources=qualified_worker(q['qualified_tasks'][task]);workers[task]=worker;source_hashes.update(sources)
    os.environ.update(QWEN3VL_PATH=str(a.qwen_metadata),LINGBOT_VLA2_SOURCE=str(a.source),
        LINGBOT_VLA2_CHECKPOINT=str(a.checkpoint),LINGBOT_ROBOTWIN_EXACT_MANIFEST=str(a.manifest))
    from scripts.model_invariance.adapters.lingbot_robotwin_exact_v2 import LingBotRobotwinExactAdapterV2
    from batch_invariant_ops import set_batch_invariant_mode
    torch.set_num_threads(8)
    adapter=LingBotRobotwinExactAdapterV2();model=adapter.load_model()
    require(runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False,'Diagnostic precision differs')
    adapter._require_precision()
    require(adapter.execution_configuration()==runtime['execution_configuration']==qa['adapter_execution_configuration'],'Loaded numerical path differs from accepted diagnostics')
    adapter.server.use_length=50
    diagnostic=[r for r in manifest['observations'] if r['split']=='diagnostic']
    require(set(r['task'] for r in diagnostic)==set(tasks),'Diagnostic companions omit tasks')
    lock=dict(model_id='lingbot_robotwin_exact',global_tasks=tasks,initialization_ids=list(range(50)),conditions=list(CONDITIONS),
        dynamic_cycle=list(DYNAMIC_CYCLE),eligible_seed_start=300000,expert_candidates_per_start=100,instruction_type='unseen',test_num=100,
        companion_schedule='diagnostic_other_tasks_offset_seed97_call11_slot17_unique_v1',qualification_union_sha256=sha(a.qualification_union),
        replay_manifest_path=str(a.manifest.resolve()),replay_manifest_sha256=sha(a.manifest),
        diagnostic_audit_path=str(a.qa_audit.resolve()),diagnostic_audit_sha256=sha(a.qa_audit),
        source_corpus_lock_path=str(corpus_lock_path.resolve()),source_corpus_lock_sha256=sha(corpus_lock_path),asset_sha256=assets,
        adapter_execution_configuration=adapter.execution_configuration(),source_sha256=source_hashes,
        startup_weight_audit=adapter.startup_weight_audit,native_contract=manifest['native_contract'],
        runtime=dict(adapter='scripts.model_invariance.adapters.lingbot_robotwin_exact_v2',tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_properties(0).name,
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),simulator_python=str(a.robotwin_python.resolve()),
            simulator_source=str(a.robotwin_source.resolve())),
        primary_endpoint='singleton_vs_batched_stability',physics_capture_contract=rigid_contract.CONTRACT,numerical_qa_acceptance_pending=True)
    a.output.mkdir(parents=True,exist_ok=True);lock_path=a.output/'candidate_lock.json'
    if lock_path.exists():require(json.loads(lock_path.read_text())==lock,'Candidate changed on resume')
    else:write_json(lock_path,lock)
    lock_sha=sha(lock_path)
    for path,identity in source_hashes.items():atomic(a.output/'source_snapshots'/identity,Path(path).read_bytes())

    def read_frame(frame):
        path=Path(frame['path']).resolve();require(path.is_relative_to(a.output.resolve()),'Worker frame escapes study')
        raw=torch.load(path,map_location='cpu',weights_only=False)
        rigid_auditor.verify_rigid_physics(raw)
        for key in ('observation','physics','rng','task_state'):
            require(digest(raw[key])==frame[key+'_sha256'],'Raw worker state differs: '+key)
        require(raw['controller']==frame['controller'],'Controller evidence differs')
        stored=retain(a.output,'frames',raw);path.unlink()
        return {k:v for k,v in frame.items() if k!='path'}|{'raw_frame':stored},raw['observation']

    def prepare(obs,instruction):
        return adapter.server._prepare_model_input({'task':instruction,'observation.state':obs['joint_action']['vector'],
            'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
            'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
            'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb']})

    for task in tasks:
        previous=299999
        for initial in range(50):
            folder=a.output/'blocks'/task/f'start-{initial:02d}';summary_path=folder/'block_summary.json'
            if summary_path.exists():
                summary=json.loads(summary_path.read_text());require(summary['candidate_lock_sha256']==lock_sha,'Foreign completed block')
                previous=summary['seed'];continue
            require(not folder.exists(),'Preserve and inspect unfinished block before retrying in a new study directory')
            for path,identity in source_hashes.items():require(sha(Path(path))==identity,'Pinned execution source changed')
            sim=Worker(a.robotwin_python,workers[task],a.robotwin_source,folder/'worker')
            try:
                selection=None
                for seed in range(previous+1,previous+101):
                    request=dict(op='expert',task=task,seed=seed,test_num=100,instruction_type='unseen')
                    response=sim.rpc(request,allow_unstable=True)
                    with (folder/'expert_seeds.jsonl').open('a') as f:f.write(json.dumps(dict(request=request,response=response))+'\n')
                    if response['ok'] and response['result']['eligible']:selection=response['result'];break
                require(selection is not None,'No expert-eligible start in100 prescribed candidates')
                previous=seed;canonical,_=read_frame(selection['canonical']);selection['canonical']=canonical
                write_json(folder/'selection.json',dict(request=request,selection=selection))
                episodes={}
                for condition in (*CONDITIONS,*(name+'_repeat' for name in CONDITIONS)):
                    invariant=condition.startswith('patched');dynamic='dynamic' in condition
                    adapter.server.global_step=0;adapter.server.last_action_chunk=None;adapter.server.last_normalized_action_chunk=None
                    adapter.reset_model(model)
                    frame,obs=read_frame(sim.rpc(dict(request,op='reset',episode_info=selection['episode_info'],instruction=selection['instruction']))['result'])
                    require(frame==canonical,'Canonical observable reset differs')
                    trajectory=[frame];calls=[];steps=[]
                    while not frame['controller']['eval_success'] and frame['controller']['take_action_count']<frame['controller']['step_limit']:
                        call=len(calls);plan=call_plan(task,seed,call,dynamic,diagnostic);transformed=prepare(obs,selection['instruction'])
                        target={k:transformed[k] for k in adapter.model_keys}
                        target['noise']=torch.randn(50,55,generator=torch.Generator().manual_seed(plan['noise_seed']))
                        target['transform_context']=transformed
                        requests=[adapter.load_replay_input(index,0) for index in plan['companion_replay_indices']]
                        requests.insert(plan['target_position'],target)
                        with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]),set_batch_invariant_mode(invariant):
                            output=adapter.run_model(model,adapter.compose_batch(requests))
                        index=plan['target_position'];normalized=output['normalized'][index].detach().cpu().contiguous()
                        physical=output['physical'][index].detach().cpu().contiguous();chunk=physical.numpy()
                        require(chunk.shape==(50,14) and np.isfinite(chunk).all(),'Invalid physical chunk')
                        inputs={k:v for k,v in target.items() if k!='transform_context'}
                        signatures={k:tensor_signature(v) for k,v in inputs.items()}
                        record=dict(plan,call=call,frame_index=len(steps),target_input_sha256=digest(signatures),input_signatures=signatures,
                            raw_request=retain(a.output,'requests',inputs),normalized_sha256=tensor_signature(normalized)['sha256'],
                            physical_sha256=tensor_signature(physical)['sha256'],normalized_actions=normalized.tolist(),physical_chunk=chunk.tolist())
                        calls.append(record)
                        for action in chunk:
                            frame,obs=read_frame(sim.rpc(dict(op='step',action=action.tolist()))['result'])
                            steps.append(dict(policy_call=call,action=action.tolist()));trajectory.append(frame)
                            if frame['controller']['eval_success'] or frame['controller']['take_action_count']>=frame['controller']['step_limit']:break
                    require(calls and steps,'Empty episode')
                    episode=dict(task=task,initialization=initial,seed=seed,condition=condition,candidate_lock_sha256=lock_sha,
                        instruction=selection['instruction'],calls=calls,steps=steps,trajectory=trajectory,
                        success=frame['controller']['eval_success'],numerical_qa_acceptance_pending=True)
                    atomic(folder/(condition+'.json.gz'),gzip.compress(json.dumps(episode).encode(),compresslevel=1,mtime=0));episodes[condition]=episode
                    print(f'EPISODE {task}/{initial}/{condition}: controls={len(steps)}',flush=True)
                checks=compare_episodes(episodes)
                write_json(summary_path,dict(task=task,initialization=initial,seed=seed,candidate_lock_sha256=lock_sha,
                    **checks,success={name:e['success'] for name,e in episodes.items()},independent_audit_pending=True))
                print(f'BLOCK {task}/{initial}: {checks}',flush=True)
            finally:sim.close()


if __name__=='__main__':main()

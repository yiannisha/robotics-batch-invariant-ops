"""Reconstruct LingBot paired input/action and full observable-state evidence.

No neural inference is permitted in this auditor. Producer checks and summaries
are compared with independently recomputed evidence, never accepted alone.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_lingbot_campaign_corpus import preprocessing_server,content_hash,signature,KEYS
from scripts.qualify_simpler_fixed_action_replay import digest

CONDITIONS=('native_singleton','native_dynamic','patched_singleton','patched_dynamic')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def reconstruct_physical_chunk(server,full,call):
    normalized=np.asarray(call['normalized_actions'],dtype=np.float32);physical=np.asarray(call['physical_chunk'],dtype=np.float32)
    require(normalized.shape==(50,55) and physical.shape==(50,14) and np.isfinite(normalized).all() and np.isfinite(physical).all(),'Invalid model output geometry')
    require(hashlib.sha256(normalized.tobytes()).hexdigest()==call['normalized_sha256'] and hashlib.sha256(physical.tobytes()).hexdigest()==call['physical_sha256'],'Output bytes differ')
    mapped=server._unapply_batched_actions([copy.deepcopy(full)],torch.from_numpy(normalized.copy()).unsqueeze(0))
    require(np.asarray(mapped['action'][0],dtype=np.float32).tobytes()==physical.tobytes(),'Official physical mapping differs')
    return physical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','qualification-union','source','checkpoint','qwen-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args();d=a.directory
    require(not a.output.exists(),'Preserve existing audits')
    torch.set_num_threads(8);lock_path=d/'candidate_lock.json';lock_hash=sha(lock_path);lock=json.loads(lock_path.read_text())
    q=json.loads(a.qualification_union.read_text())
    require(sha(a.qualification_union)==lock['qualification_union_sha256'],'Wrong task qualification union')
    tasks=q['prescribed_tasks'];require(len(tasks)==50 and len(set(tasks))==50 and lock['global_tasks']==tasks and set(q['qualified_tasks'])==set(tasks),'Wrong task roster')
    for entry in q['qualified_tasks'].values():
        directory=Path(entry['directory']);audit_path=directory/'independent_audit.json'
        require(sha(audit_path)==entry['independent_audit_sha256'],'Qualified simulator audit changed')
        proof=json.loads(audit_path.read_text())
        require(proof['initial_state_and_full_trajectories_exact'] and proof['cpu_and_cuda_rng_verified'] and proof['source_defined_task_state_frames_verified']==proof['raw_frames_verified'],'Incomplete simulator qualification')
        require(sha(directory/'replays.jsonl')==proof['raw_replays_sha256'],'Qualified simulator raw replay changed')
    require(lock['initialization_ids']==list(range(50)) and lock['conditions']==list(CONDITIONS) and lock['dynamic_cycle']==[1,2,4,8],'Wrong paired cohort/conditions')
    require(lock['eligible_seed_start']==300000 and lock['expert_candidates_per_start']==100 and lock['instruction_type']=='unseen' and lock['test_num']==100,'Expert-start protocol differs')
    require(lock['companion_schedule']=='diagnostic_other_tasks_offset_seed97_call11_slot17_unique_v1','Wrong companion schedule')
    for path,identity in lock['source_sha256'].items():require(sha(d/'source_snapshots'/identity)==identity,'Source archive changed')
    for path,identity in lock['asset_sha256'].items():require(sha(Path(path))==identity,'Checkpoint/tokenizer bytes changed')
    require(sha(Path(lock['source_corpus_lock_path']))==lock['source_corpus_lock_sha256'],'Native source lock changed')
    manifest_path=Path(lock['replay_manifest_path']);require(sha(manifest_path)==lock['replay_manifest_sha256'],'Companion corpus changed')
    manifest=json.loads(manifest_path.read_text());diagnostic=[r for r in manifest['observations'] if r['split']=='diagnostic']
    qa_path=Path(lock['diagnostic_audit_path']);require(sha(qa_path)==lock['diagnostic_audit_sha256'],'Diagnostic acceptance changed')
    qa=json.loads(qa_path.read_text());require(qa['status']=='complete_four_model_diagnostic_independently_verified' and qa['model_id']=='lingbot_robotwin_exact' and qa['manifest_sha256']==lock['replay_manifest_sha256'] and qa['adapter_execution_configuration']==lock['adapter_execution_configuration'] and qa['invariant_output_and_all_recorded_stages_exact'],'Wrong model diagnostic proof')
    require(lock['runtime']['adapter']=='scripts.model_invariance.adapters.lingbot_robotwin_exact_v2','Wrong corrected adapter')
    require(lock['runtime']['tf32_matmul'] is False and lock['runtime']['tf32_cudnn'] is False,'Wrong corrected simulator precision')
    server=preprocessing_server(a.source,a.checkpoint,a.qwen_metadata)
    file_hashes={};blocks=[];frames_verified=0;calls_rebuilt=0;previous={task:299999 for task in tasks};seen=set()
    def raw(relative,category):
        path=(d/relative).resolve();require(path.parent==(d/category).resolve(),'Raw path escapes category')
        with gzip.open(path,'rb') as stream:value=torch.load(stream,map_location='cpu',weights_only=False)
        require(content_hash(value)==path.name.split('.')[0],'Raw content address changed');file_hashes[str(path)]=sha(path)
        return value
    def prepare(value,instruction):
        obs=value['observation']
        return server._prepare_model_input({'task':instruction,'observation.state':obs['joint_action']['vector'],
            'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
            'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
            'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb']})
    paths=sorted((d/'blocks').glob('*/start-*/block_summary.json'),key=lambda p:(tasks.index(p.parent.parent.name),p.parent.name))
    for summary_path in paths:
        folder=summary_path.parent;summary=json.loads(summary_path.read_text());task=summary['task'];initial=summary['initialization']
        require(task in tasks and initial in range(50) and (task,initial) not in seen,'Foreign/duplicate paired start');seen.add((task,initial))
        require(initial==sum(t==task for t,i in seen)-1,'Task starts contain gaps')
        require(summary['candidate_lock_sha256']==lock_hash,'Block candidate differs')
        selection=json.loads((folder/'selection.json').read_text());chosen=selection['selection'];request=selection['request'];seed=summary['seed']
        require(request==dict(op='expert',task=task,seed=seed,test_num=100,instruction_type='unseen'),'Selected expert request differs')
        trials=[json.loads(line) for line in (folder/'expert_seeds.jsonl').read_text().splitlines()]
        require(0<len(trials)<=100 and [r['request']['seed'] for r in trials]==list(range(previous[task]+1,seed+1)),'Expert candidate sequence differs')
        for trial in trials:
            require(trial['request']==dict(op='expert',task=task,seed=trial['request']['seed'],test_num=100,instruction_type='unseen'),'Expert instruction protocol differs')
        require(all(not r['response'].get('ok') or not r['response']['result']['eligible'] for r in trials[:-1]),'Earlier eligible seed skipped')
        require(trials[-1]['response']['ok'] and trials[-1]['response']['result']['eligible'] and chosen['eligible'],'Selected seed was not expert-eligible')
        require({k:v for k,v in trials[-1]['response']['result']['canonical'].items() if k!='path'}=={k:v for k,v in chosen['canonical'].items() if k!='raw_frame'},'Expert canonical initial state differs')
        require({k:v for k,v in trials[-1]['response']['result'].items() if k!='canonical'}=={k:v for k,v in chosen.items() if k!='canonical'},'Expert selection metadata differs')
        previous[task]=seed;episodes={};frame_cache={}
        for condition in (*CONDITIONS,*(name+'_repeat' for name in CONDITIONS)):
            path=folder/(condition+'.json.gz');file_hashes[str(path.resolve())]=sha(path)
            with gzip.open(path,'rt') as stream:episode=json.load(stream)
            require(episode['task']==task and episode['initialization']==initial and episode['seed']==seed and episode['condition']==condition and episode['candidate_lock_sha256']==lock_hash,'Foreign episode')
            require(episode['instruction']==chosen['instruction'] and episode['numerical_qa_acceptance_pending'] is True,'Wrong instruction or premature acceptance')
            steps=episode['steps'];trajectory=episode['trajectory'];calls=episode['calls']
            require(steps and len(trajectory)==len(steps)+1 and trajectory[0]==chosen['canonical'],'Missing controls or canonical reset differs')
            for i,frame in enumerate(trajectory):
                if frame['raw_frame'] not in frame_cache:frame_cache[frame['raw_frame']]=raw(frame['raw_frame'],'frames')
                value=frame_cache[frame['raw_frame']]
                for key in ('observation','physics','rng','task_state'):require(digest(value[key])==frame[key+'_sha256'],'Full observable state differs: '+key)
                require(value['controller']==frame['controller'] and frame['controller']['take_action_count']==i and frame['controller']['instruction']==chosen['instruction'],'Controller progression differs')
                if i<len(steps):require(not frame['controller']['eval_success'] and i<frame['controller']['step_limit'],'Episode continued after termination')
                frames_verified+=1
            final=trajectory[-1]['controller'];require(episode['success']==final['eval_success'] and (final['eval_success'] or len(steps)==final['step_limit']),'Episode stopped early/outcome differs')
            cursor=0
            for i,call in enumerate(calls):
                require(call['call']==i and call['frame_index']==cursor,'Call/control alignment differs')
                key=f'{task}/{seed}/policy-call-{i}';noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
                require(call['noise_id']==key and call['noise_seed']==noise_seed,'Request-indexed noise differs')
                size=[1,2,4,8][i%4] if 'dynamic' in condition else 1;position=i%size
                require(call['batch_size']==size and call['target_position']==position,'Dynamic batch/target differs')
                candidates=[r for r in diagnostic if r['task']!=task];indices=[];used=set()
                for slot in range(size-1):
                    index=(seed*97+i*11+slot*17)%len(candidates)
                    while index in used:index=(index+1)%len(candidates)
                    used.add(index);indices.append(candidates[index]['replay_index'])
                require(call['companion_replay_indices']==indices,'Unrelated diagnostic companions differ')
                target=raw(call['raw_request'],'requests');full=prepare(frame_cache[trajectory[cursor]['raw_frame']],episode['instruction'])
                expected={k:full[k] for k in KEYS};expected['noise']=torch.randn(50,55,generator=torch.Generator().manual_seed(noise_seed),dtype=torch.float32)
                require(set(target)==set(expected),'Prepared input fields differ')
                signatures={k:signature(v) for k,v in target.items()}
                require(signatures=={k:signature(v) for k,v in expected.items()}==call['input_signatures'] and digest(signatures)==call['target_input_sha256'],'Reconstructed neural inputs differ')
                physical=reconstruct_physical_chunk(server,full,call)
                count=min(50,len(steps)-cursor);executed=steps[cursor:cursor+count]
                require(count>0 and all(s['policy_call']==i for s in executed) and np.asarray([s['action'] for s in executed],dtype=np.float32).tobytes()==physical[:count].tobytes(),'Executed physical commands differ')
                cursor+=count;calls_rebuilt+=1
            require(cursor==len(steps),'Uncovered controls');episodes[condition]=episode
        for name in CONDITIONS:
            for field in ('calls','steps','trajectory','success'):require(digest(episodes[name][field])==digest(episodes[name+'_repeat'][field]),'Same-condition repeat differs')
        require(len({digest(e['trajectory'][0]) for e in episodes.values()})==1 and len({e['calls'][0]['target_input_sha256'] for e in episodes.values()})==1,'Initial snapshots/inputs differ')
        left,right=episodes['patched_singleton'],episodes['patched_dynamic']
        require(digest(left['trajectory'])==digest(right['trajectory']) and digest(left['steps'])==digest(right['steps']),'Invariant trajectory differs')
        require([(r['normalized_sha256'],r['physical_sha256']) for r in left['calls']]==[(r['normalized_sha256'],r['physical_sha256']) for r in right['calls']],'Invariant output chunk differs')
        left,right=episodes['native_singleton'],episodes['native_dynamic']
        first=next((i for i,(x,y) in enumerate(zip(left['steps'],right['steps'])) if np.asarray(x['action'],dtype=np.float32).tobytes()!=np.asarray(y['action'],dtype=np.float32).tobytes()),None)
        if first is not None:
            require(digest(left['trajectory'][:first+1])==digest(right['trajectory'][:first+1]),'Native divergence physical prefix differs')
            require(left['calls'][left['steps'][first]['policy_call']]['target_input_sha256']==right['calls'][right['steps'][first]['policy_call']]['target_input_sha256'],'Native divergence inputs differ')
        require(summary['native_first_action_divergence']==first and all(summary['success'][name]==e['success'] for name,e in episodes.items()),'Producer summary differs from reconstructed evidence')
        blocks.append(dict(task=task,initialization=initial,seed=seed,native_first_action_divergence=first,outcomes={name:episodes[name]['success'] for name in CONDITIONS}))
    complete=seen=={(task,i) for task in tasks for i in range(50)}
    require(not a.require_complete or complete,'Full2500start LingBot cohort incomplete')
    require(sha(lock_path)==lock_hash,'Candidate changed during audit')
    result=dict(status='lingbot_paired_raw_inputs_actions_and_observable_trajectories_independently_verified',candidate_lock_sha256=lock_hash,
        completed_paired_starts=len(blocks),prescribed_paired_starts=2500,episodes=8*len(blocks),raw_frames_verified=frames_verified,
        reconstructed_policy_calls=calls_rebuilt,all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks=True,
        cohort_rollouts_complete=complete,official_cpu_transforms_only_no_neural_inference=True,raw_files_sha256=file_hashes,
        blocks=blocks,auditor_sha256=sha(Path(__file__)),numerical_qa_acceptance_pending=True,full_model_campaign_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.output)


if __name__=='__main__':main()

"""Independently audit committed RDT study blocks and paired task outcomes."""
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


def td(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');p.add_argument('--rebuild-inputs',action='store_true');args=p.parse_args()
    torch.set_num_threads(8)
    d=args.directory;lock_path=d/'candidate_lock.json';lock=json.loads(lock_path.read_text());lock_hash=hashlib.sha256(lock_path.read_bytes()).hexdigest()
    require(lock['initialization_ids']==list(range(50)) and len(set(lock['global_tasks']))==5,'Wrong cohort')
    for path,expected in lock['source_sha256'].items():require(hashlib.sha256((d/'source_snapshots'/expected).read_bytes()).hexdigest()==expected,'Locked source snapshot changed')
    base=lock['conditions'];conditions=base+[name+'_repeat' for name in base]
    require(base==['native_singleton','native_dynamic','patched_singleton','patched_dynamic'],'Wrong conditions')
    manifest_path=Path(lock['replay_manifest_path'])
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest()==lock['replay_manifest_sha256'],'Frozen manifest changed')
    manifest=json.loads(manifest_path.read_text())
    require(lock['companion_schedule']=='diagnostic_other_tasks_offset_init97_call11_slot17_v1','Unexpected companion schedule')
    seen_blocks=set();by_task={};blocks=[];reconstructed=0;initial_physics={}
    if args.rebuild_inputs:
        from transformers import AutoImageProcessor
        from PIL import Image
        from scripts.run_rdt_paired_server import STATE_MIN,STATE_MAX
        from scripts.model_invariance.adapters.rdt_maniskill import MANISKILL_INDICES
        processor_path=next(Path(path).parent for path in lock['asset_sha256'] if Path(path).name=='preprocessor_config.json')
        processor=AutoImageProcessor.from_pretrained(processor_path)
        language={task:torch.load(next(Path(path) for path in lock['asset_sha256'] if Path(path).name==f'text_embed_{task}.pt'),map_location='cpu',weights_only=True).to(torch.bfloat16) for task in lock['global_tasks']}
    def raw(relative,category):
        path=(d/relative).resolve();require(path.parent==(d/category).resolve(),'Raw path escapes category')
        with gzip.open(path,'rb') as f:value=torch.load(f,map_location='cpu',weights_only=False)
        require(digest(value)==path.name.split('.')[0],'Raw content address changed')
        return value
    for summary_path in sorted((d/'blocks').glob('*/start-*/block_summary.json')):
        summary=json.loads(summary_path.read_text());task=summary['task'];initial=summary['initialization'];key=(task,initial)
        require(key not in seen_blocks and task in lock['global_tasks'] and initial in range(50),'Duplicate/foreign block');seen_blocks.add(key)
        require(summary['candidate_lock_sha256']==lock_hash,'Foreign summary candidate')
        rows={}
        # Cache only this block's raw frames; large image payloads are released
        # after the block, while identical repeated frames reuse verified bytes.
        frame_cache={}
        for name in conditions:
            with gzip.open(summary_path.parent/(name+'.json.gz'),'rt') as f:row=json.load(f)
            require(row['task']==task and row['initialization']==initial and row['condition']==name,'Foreign episode')
            require(row['gpu_uuid']==summary['gpu_uuid'] and row['candidate_lock_sha256']==lock_hash,'Mixed candidate/device')
            steps=row['steps'];trajectory=row['trajectory'];calls=row['calls']
            require(len(trajectory)==len(steps)+1 and 0<len(steps)<=400,'Incomplete trajectory')
            done=(steps[-1]['terminated'] or steps[-1]['truncated']) and steps[-1]['success']
            require(row['success']==done and (done or len(steps)==400),'Incomplete outcome')
            for index,frame in enumerate(trajectory):
                path=frame['raw_frame']
                if path not in frame_cache:
                    saved=raw(path,'frames')
                    expected={k+'_sha256':digest(v) for k,v in saved.items()}|{'raw_frame':path}
                    require(set(saved)=={'observation','policy_rgb','physics','controller','task','drives','elapsed_steps','rng'},'Missing state category')
                    frame_cache[path]=(expected,saved)
                expected,saved=frame_cache[path]
                require(frame==expected and int(np.asarray(saved['elapsed_steps']).reshape(-1)[0])==index,'Raw state/hash/index differs')
                require(saved['rng']['cuda_initialized'] and 'torch_cuda' in saved['rng'],'CUDA RNG missing')
            previous_controls=0
            for call_index,call in enumerate(calls):
                size=[1,2,4,8][call_index%4] if 'dynamic' in name else 1
                require(call['call']==call_index and call['batch_size']==size and call['position']==call_index%size,'Batch schedule differs')
                require(len(call['companion_ids'])==size-1 and len(set(call['companion_ids']))==size-1,'Wrong companion schedule extent')
                pool=[row for row in manifest['observations'] if row['split']=='diagnostic' and row['task']!=task]
                require(call['companion_ids']==[pool[(initial*97+call_index*11+j*17)%len(pool)]['request_ids'][0] for j in range(size-1)],'Companions differ from frozen schedule')
                request=raw(call['raw_request'],'requests')
                require(digest(request)==call['raw_request_sha256'],'Raw request changed')
                noise_id=f'{task}/{initial}/policy-call-{call_index}';seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
                require(call['noise_id']==noise_id and call['noise_seed']==seed,'Noise schedule differs')
                noise=torch.randn(64,128,generator=torch.Generator().manual_seed(seed)).numpy()
                require(request['noise'].tobytes()==noise.tobytes(),'Indexed noise bytes differ')
                current=frame_cache[trajectory[previous_controls]['raw_frame']][1]
                rgb=current['policy_rgb'];previous_rgb=None if previous_controls==0 else frame_cache[trajectory[previous_controls-1]['raw_frame']][1]['policy_rgb']
                history=np.stack([np.zeros_like(rgb) if previous_rgb is None else previous_rgb,rgb])
                require(request['task_name']==task and request['history_valid']==[previous_rgb is not None,True],'Task/history validity differs')
                require(np.array_equal(request['image_history'],history),'Policy history differs from recorded controls')
                joints=np.ascontiguousarray(current['observation']['agent']['qpos'][:,:-1].reshape(-1),dtype=np.float32)
                require(request['proprio'].tobytes()==joints.tobytes(),'Policy proprio differs')
                require(digest(call['model_inputs'])==call['target_input_sha256'],'Model-input record changed')
                payload=bytes.fromhex(call['normalized_bytes'])
                require(call['normalized']['dtype']=='torch.bfloat16' and call['normalized']['shape']==[1,64,8] and len(payload)==1024 and hashlib.sha256(payload).hexdigest()==call['normalized']['sha256'],'Normalized action bytes differ')
                executed=[step['action'] for step in steps if step['policy_call']==call_index]
                require(0<len(executed)<=16 and np.asarray(executed,dtype=np.float32).tobytes()==np.asarray(call['actions'][::4][:len(executed)],dtype=np.float32).tobytes(),'Action stride/chunk differs')
                require(call_index==len(calls)-1 or len(executed)==16,'Short nonterminal action chunk')
                if args.rebuild_inputs:
                    color=tuple(int(v*255) for v in processor.image_mean);size_config=processor.size
                    background=Image.fromarray(np.ones((size_config['height'],size_config['width'],3),dtype=np.uint8)*np.asarray(color,dtype=np.uint8))
                    slots=[Image.fromarray(previous_rgb) if previous_rgb is not None else background,background,background,Image.fromarray(rgb),background,background]
                    pixels=[]
                    for image in slots:
                        width,height=image.size;side=max(width,height)
                        square=Image.new(image.mode,(side,side),color);square.paste(image,((side-width)//2,(side-height)//2))
                        pixels.append(processor.preprocess(square,return_tensors='pt')['pixel_values'][0])
                    language_tokens=language[task];lang=torch.zeros(1,30,4096,dtype=torch.bfloat16);lang[0,:language_tokens.shape[1]]=language_tokens[0]
                    state=torch.zeros(1,1,128,dtype=torch.bfloat16);state[0,0,list(MANISKILL_INDICES)]=torch.from_numpy((joints-STATE_MIN)/(STATE_MAX-STATE_MIN)*2-1).to(torch.bfloat16)
                    inputs={'input.language':td(lang),'input.state':td(state),'input.noise':td(torch.from_numpy(noise).to(torch.bfloat16).unsqueeze(0)),
                        'input.vision_pixels':td(torch.stack(pixels).to(torch.bfloat16).unsqueeze(0)),
                        'language_mask':[j<language_tokens.shape[1] for j in range(30)],'control_frequency':25,'history_valid':[previous_rgb is not None,True]}
                    require(digest(inputs)==call['target_input_sha256'],'Rebuilt actual model inputs differ');reconstructed+=1
                previous_controls+=len(executed)
            require(previous_controls==len(steps),'Unattributed controls')
            rows[name]=row
        for name in base:
            for field in ('calls','steps','trajectory','success'):require(digest(rows[name][field])==digest(rows[name+'_repeat'][field]),f'A/A differs: {name}/{field}')
        require(len({digest(row['trajectory'][0]) for row in rows.values()})==1,'Canonical initial state differs')
        require(len({row['calls'][0]['target_input_sha256'] for row in rows.values()})==1,'Initial policy input differs')
        a,b=rows['patched_singleton'],rows['patched_dynamic']
        for field in ('steps','trajectory','success'):require(digest(a[field])==digest(b[field]),f'Invariant trajectories differ: {field}')
        require([c['normalized'] for c in a['calls']]==[c['normalized'] for c in b['calls']],'Invariant normalized output differs')
        a,b=rows['native_singleton'],rows['native_dynamic']
        first=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if np.asarray(x['action'],dtype=np.float32).tobytes()!=np.asarray(y['action'],dtype=np.float32).tobytes()),None)
        if first is not None:
            ca=a['calls'][a['steps'][first]['policy_call']];cb=b['calls'][b['steps'][first]['policy_call']]
            require(ca['target_input_sha256']==cb['target_input_sha256'] and digest(a['trajectory'][:first+1])==digest(b['trajectory'][:first+1]),'Native first divergence is not attributable')
        require(first==summary['native_first_action_divergence'],'First-divergence summary differs')
        counts=by_task.setdefault(task,dict(paired_starts=0,successes={name:0 for name in base}))
        counts['paired_starts']+=1
        for name in base:counts['successes'][name]+=int(rows[name]['success'])
        initial_physics.setdefault(task,set()).add(rows[base[0]]['trajectory'][0]['physics_sha256'])
        blocks.append(dict(task=task,initialization=initial,outcomes={name:rows[name]['success'] for name in base},native_first_action_divergence=first))
    if args.require_complete:require(len(blocks)==250,'Full cohort is incomplete')
    for task,counts in by_task.items():
        n=counts['paired_starts'];counts['unique_initial_physics_states']=len(initial_physics[task]);counts['success_rate']={name:value/n for name,value in counts['successes'].items()}
        counts['BI_minus_native_singleton_percentage_points']=100*(counts['successes']['patched_singleton']-counts['successes']['native_singleton'])/n
        counts['BI_minus_native_dynamic_percentage_points']=100*(counts['successes']['patched_dynamic']-counts['successes']['native_dynamic'])/n
    report=dict(status='paired_rollouts_independently_audited_provisional_pending_numerical_QA',completed_paired_starts=len(blocks),
        prescribed_paired_starts=250,episodes=len(blocks)*8,by_task=by_task,blocks=blocks,reconstructed_policy_calls=reconstructed,
        all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks=True,cohort_rollouts_complete=len(blocks)==250,
        controlled_native_fixed30=True,numerical_qa_acceptance_pending=True,success_rate_campaign_complete=False,
        candidate_lock_sha256=lock_hash,auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Independent raw-state/input/controller audit of committed CogACT blocks."""
import argparse
from collections import deque
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.qualify_simpler_fixed_action_replay import digest

PRIMARY=['native_singleton','native_dynamic','patched_singleton','patched_dynamic']
SUPPLEMENT=['released_native_singleton','released_native_dynamic']


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def td(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def same64(a,b):return np.asarray(a,dtype=np.float64).tobytes()==np.asarray(b,dtype=np.float64).tobytes()


class Controller:
    """Released two-step cosine ensemble and Google sticky-gripper formula."""
    def __init__(self):
        self.history=deque(maxlen=2);self.sticky=False;self.count=0;self.grip=np.array([0.0]);self.previous=None
    def snapshot(self):
        return dict(policy_setup='google_robot',sticky_action_is_on=self.sticky,gripper_action_repeat=self.count,
            sticky_gripper_action=self.grip,previous_gripper_action=self.previous,
            ensemble=dict(action_history=list(self.history),maxlen=2,adaptive_ensemble_alpha=0.1))
    def process(self,chunk):
        from transforms3d.euler import euler2axangle
        self.history.append(chunk)
        predictions=np.stack([v[i] for i,v in zip(range(len(self.history)-1,-1,-1),self.history)])
        ref=predictions[-1];similarity=np.sum(predictions*ref,axis=1)/(np.linalg.norm(predictions,axis=1)*np.linalg.norm(ref)+1e-7)
        weights=np.exp(0.1*similarity);weights=weights/weights.sum();selected=np.sum(weights[:,None]*predictions,axis=0)
        axes,angle=euler2axangle(*np.asarray(selected[3:6],dtype=np.float64));current=selected[6:7]
        if self.previous is None:relative=np.array([0]);self.previous=current
        else:relative=self.previous-current
        if np.abs(relative)>0.5 and not self.sticky:self.sticky=True;self.grip=relative;self.previous=current
        if self.sticky:self.count+=1;relative=self.grip
        if self.count==10:self.sticky=False;self.count=0;self.grip=np.array([0.0])
        return selected,np.concatenate([selected[:3],axes*angle,relative]).astype(np.float64)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rebuild-inputs',action='store_true');p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    torch.set_num_threads(8);d=args.directory;lock=json.loads((d/'candidate_lock.json').read_text());lock_hash=sha(d/'candidate_lock.json')
    require(lock['global_tasks']==['pick_can','move_near','open_drawer','close_drawer'] and lock['initialization_ids']==list(range(50)),'Wrong cohort')
    require(lock['primary_conditions']==PRIMARY and lock['supplemental_released_native_conditions']==SUPPLEMENT,'Wrong condition protocols')
    for path,expected in lock['source_sha256'].items():require(sha(d/'source_snapshots'/expected)==expected,'Archived locked source changed')
    manifest_path=Path(lock['replay_manifest_path']);require(sha(manifest_path)==lock['replay_manifest_sha256'],'Frozen companions changed')
    manifest=json.loads(manifest_path.read_text());diagnostic=[r for r in manifest['observations'] if r['split']=='diagnostic']
    by_request={rid:row for row in diagnostic for rid in row['request_ids']}
    if args.rebuild_inputs:
        from transformers import AutoTokenizer
        import timm
        from torchvision.transforms import Compose,Resize
        meta=next(Path(p).parent for p in lock['asset_sha256'] if Path(p).name=='tokenizer.model')
        tokenizer=AutoTokenizer.from_pretrained(meta,model_max_length=2048,padding_side='right');tokenizer.add_special_tokens({'pad_token':'<PAD>'})
        path=next(Path(p) for p in lock['source_sha256'] if p.endswith('/llm/prompting/base_prompter.py'))
        spec=importlib.util.spec_from_file_location('audited_pure_prompter',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def tokens(instruction):
            prompt=module.PurePromptBuilder('llama2');prompt.add_turn(role='human',message=f'What action should the robot take to {instruction.lower()}?')
            value=tokenizer(prompt.get_prompt(),truncation=True,return_tensors='pt').input_ids.squeeze(0)
            return torch.cat([value,torch.tensor([29871,2])])
        transforms={}
        for key,name in [('dino','vit_large_patch14_reg4_dinov2.lvd142m'),('siglip','vit_so400m_patch14_siglip_224')]:
            cfg=timm.data.resolve_data_config(pretrained_cfg=timm.models.get_pretrained_cfg(name).to_dict());cfg['input_size']=(3,224,224)
            transform=timm.data.create_transform(**cfg,is_training=False)
            require(isinstance(transform,Compose) and isinstance(transform.transforms[0],Resize),'Unexpected vision preprocessing')
            transforms[key]=Compose([Resize((224,224),interpolation=transform.transforms[0].interpolation),*transform.transforms[1:]])
    names=PRIMARY+SUPPLEMENT;conditions=names+[n+'_repeat' for n in names];counts={};blocks=[];seen=set();reconstructed=0;frames_verified=0
    def raw(relative,category):
        path=(d/relative).resolve();require(path.parent==(d/category).resolve(),'Raw path escapes study')
        with gzip.open(path,'rb') as f:value=torch.load(f,map_location='cpu',weights_only=False)
        require(digest(value)==path.name.split('.')[0],'Raw content address changed');return value
    for summary_path in sorted((d/'blocks').glob('*/start-*/block_summary.json')):
        summary=json.loads(summary_path.read_text());task,initial=summary['task'],summary['initialization'];key=(task,initial)
        require(task in lock['global_tasks'] and initial in range(50) and key not in seen,'Foreign/duplicate block');seen.add(key)
        require(summary['candidate_lock_sha256']==lock_hash,'Foreign candidate');rows={};frame_cache={}
        for name in conditions:
            with gzip.open(summary_path.parent/(name+'.json.gz'),'rt') as f:row=json.load(f)
            require((row['task'],row['initialization'],row['condition'])==(task,initial,name),'Foreign episode')
            require(row['candidate_lock_sha256']==lock_hash and row['gpu_uuid']==summary['gpu_uuid'],'Mixed device/source candidate')
            calls,steps,trajectory=row['calls'],row['steps'],row['trajectory'];released=name.startswith('released_')
            require(row['released_default']==released and len(calls)==len(steps)==len(trajectory)-1 and 0<len(calls)<=200,'Incomplete trajectory')
            require(steps[-1]['truncated'] and row['success']==steps[-1]['success'],'Incomplete outcome/horizon')
            controller=Controller();initial_rng=None
            for i,frame in enumerate(trajectory):
                path=frame['raw_frame']
                if path not in frame_cache:
                    value=raw(path,'frames');expected={k+'_sha256':digest(v) for k,v in value.items()}|{'raw_frame':path}
                    require(set(value)=={'observation','scene_pack','physics','agent','controller','task','elapsed_steps','wrapper_elapsed_steps','rng','instruction','policy_controller','policy_rng'},'Missing state category')
                    for filename,expected_hash in value['task']['source_sha256'].items():require(lock['source_sha256'].get(filename)==expected_hash,'Simulator task source differs from candidate lock')
                    frame_cache[path]=(expected,value)
                expected,value=frame_cache[path];require(expected==frame and value['elapsed_steps']==i,'Raw state/index/hash differs')
                require(digest(controller.snapshot())==frame['policy_controller_sha256'],'Episode policy/controller state differs')
                if initial_rng is None:initial_rng=frame['policy_rng_sha256']
                require(frame['policy_rng_sha256']==initial_rng,'Model auxiliary RNG leaked between requests')
                frames_verified+=1
                if i==len(calls):break
                call=calls[i];step=steps[i];size=[1,2,4,8][i%4] if 'dynamic' in name else 1
                require(call['call']==step['policy_call']==i and call['batch_size']==size and call['position']==i%size,'Batch/step schedule differs')
                pool=[c for c in diagnostic if c['task']!=task]
                ids=[pool[(initial*97+i*11+j*17)%len(pool)]['request_ids'][0] for j in range(size-1)]
                require(call['companion_ids']==ids,'Companion schedule differs')
                request=raw(call['raw_request'],'requests');rgb=next(iter(value['observation']['image'].values()))['rgb']
                require(np.array_equal(request['rgb'],rgb) and request['instruction']==value['instruction'],'Raw policy input differs from simulator')
                noise_id=f'{task}/{initial}/policy-call-{i}';seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
                noise=torch.randn(1,16,7,generator=torch.Generator().manual_seed(seed))
                require(call['noise_id']==noise_id and call['noise_seed']==seed and request['noise'].tobytes()==noise.numpy().tobytes(),'Indexed noise differs')
                require(digest(call['model_inputs'])==call['target_input_sha256'],'Input trace changed')
                normalized=torch.tensor(call['normalized_actions'],dtype=torch.float32)
                require(list(normalized.shape)==[1,16,7] and td(normalized)==call['normalized'],'Normalized output bytes changed')
                chunk=np.asarray(call['raw_action_chunk'],dtype=np.float64);require(chunk.shape==(16,7) and np.isfinite(chunk).all(),'Invalid predicted chunk')
                selected,action=controller.process(chunk)
                require(same64(selected,call['selected_action']) and same64(action,call['environment_action']) and same64(action,step['action']),'Released controller action reconstruction differs')
                if args.rebuild_inputs:
                    target_tokens=tokens(request['instruction']);width=64
                    if released:width=max([len(target_tokens)]+[len(tokens(by_request[rid]['instruction'])) for rid in ids])
                    input_ids=torch.full((1,width),tokenizer.pad_token_id,dtype=torch.long);input_ids[0,:len(target_tokens)]=target_tokens
                    inputs={'input.input_ids':td(input_ids),'input.attention_mask':td(input_ids.ne(tokenizer.pad_token_id)),'diffusion.noise':td(noise)}
                    image=Image.fromarray(rgb)
                    for k,transform in transforms.items():inputs['input.pixel_values.'+k]=td(transform(image).unsqueeze(0))
                    require(digest(inputs)==call['target_input_sha256'],'Rebuilt actual model inputs differ');reconstructed+=1
            if released and 'dynamic' not in name:
                require(row['official_singleton_equivalence']['exact'] and row['official_singleton_equivalence']['indexed_initial_noise_consumed_once'],'Official released singleton equivalence missing')
                require(row['official_singleton_equivalence']['physical_sha256']==hashlib.sha256(np.asarray([calls[0]['raw_action_chunk']],dtype=np.float64).tobytes()).hexdigest(),'Official singleton signature differs from executed first chunk')
            rows[name]=row
        require(len({digest(row['trajectory'][0]) for row in rows.values()})==1,'Canonical starts differ')
        for group in (PRIMARY,SUPPLEMENT):require(len({rows[n]['calls'][0]['target_input_sha256'] for n in group})==1,'Initial inputs differ within protocol')
        for name in names:
            for field in ('calls','steps','trajectory','success'):require(digest(rows[name][field])==digest(rows[name+'_repeat'][field]),f'A/A differs: {name}/{field}')
        a,b=rows['patched_singleton'],rows['patched_dynamic']
        for field in ('steps','trajectory','success'):require(digest(a[field])==digest(b[field]),f'Invariant trajectories differ: {field}')
        require([c['normalized'] for c in a['calls']]==[c['normalized'] for c in b['calls']],'Invariant normalized chunks differ')
        attribution={}
        for prefix in ('','released_'):
            a,b=rows[prefix+'native_singleton'],rows[prefix+'native_dynamic']
            first=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if not same64(x['action'],y['action'])),None)
            equal_input=None
            if first is not None:
                equal_input=a['calls'][first]['target_input_sha256']==b['calls'][first]['target_input_sha256']
                require(digest(a['trajectory'][:first+1])==digest(b['trajectory'][:first+1]),'First divergence has an earlier state difference')
                if not prefix:require(equal_input,'Controlled native first divergence has different inputs')
            attribution[prefix+'native']=dict(first_action_divergence=first,identical_preprocessed_inputs_at_divergence=equal_input,
                interpretation='released_padding_extent_and_arithmetic_joint_effect' if prefix else 'controlled_identical_input_batch_arithmetic')
        require(attribution==summary['attribution'],'First-divergence summary differs')
        counts_row=counts.setdefault(task,dict(paired_starts=0,successes={n:0 for n in names}));counts_row['paired_starts']+=1
        for name in names:counts_row['successes'][name]+=int(rows[name]['success'])
        blocks.append(dict(task=task,initialization=initial,outcomes={n:rows[n]['success'] for n in names},attribution=attribution))
    if args.require_complete:require(len(blocks)==200,'Full prescribed cohort is incomplete')
    for row in counts.values():row['success_rate']={name:value/row['paired_starts'] for name,value in row['successes'].items()}
    report=dict(status='paired_rollouts_independently_audited_provisional_pending_numerical_QA',completed_paired_starts=len(blocks),prescribed_paired_starts=200,
        episodes=len(blocks)*12,by_task=counts,blocks=blocks,reconstructed_policy_calls=reconstructed,raw_frames_verified=frames_verified,
        all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks=True,controlled_primary_fixed64=True,released_native_supplements_separate=True,
        cohort_rollouts_complete=len(blocks)==200,numerical_qa_acceptance_pending=True,success_rate_campaign_complete=False,
        candidate_lock_sha256=lock_hash,auditor_sha256=sha(Path(__file__)))
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

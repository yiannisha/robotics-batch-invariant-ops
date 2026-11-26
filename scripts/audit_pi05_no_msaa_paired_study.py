"""Independently audit no-MSAA varied-start blocks with every raw frame.

Partial outcome summaries remain provisional while numerical QA is pending.
Repeat episodes are controls, never additional independent success trials.
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


def action_bytes(value):return np.asarray(value,dtype=np.float32).tobytes()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rebuild-inputs',action='store_true');p.add_argument('--require-complete',action='store_true')
    args=p.parse_args();d=args.directory;lock_path=d/'candidate_lock.json';lock=json.loads(lock_path.read_text());lock_hash=sha(lock_path)
    require(lock['global_tasks']==list(range(10)) and lock['initialization_ids']==list(range(50)),'Study cohort changed')
    for path,expected in lock['source_sha256'].items():require(sha(d/'source_snapshots'/expected)==expected,'Locked source snapshot missing or changed')
    require(lock.get('renderer_offscreen_samples')==0 and lock.get('renderer_protocol_changed') and lock.get('default_renderer_results_must_not_be_pooled'),'Wrong renderer protocol')
    reset_path=Path(lock['reset_audit_path']);require(sha(reset_path)==lock['reset_audit_sha256'],'Reset audit changed')
    reset=json.loads(reset_path.read_text());require(reset['status']=='complete_no_msaa_observable_replay_independently_verified','Reset qualification incomplete')
    require(sha(reset_path.parent/'replays.jsonl')==reset['raw_replays_sha256'],'Reset qualification raw records changed')
    canonical={}
    for line in (reset_path.parent/'replays.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['kind']=='fixed_wave' and row['repeat']==0:canonical[(row['task'],row['initialization'])]={k:v for k,v in row['trajectory'][0].items() if k!='raw_frame'}
    raw_frame_count=0
    base=lock['conditions'];conditions=base+[name+'_repeat' for name in base]
    counts={};blocks=[];initial_states={};reconstructed=0
    pre=None
    if args.rebuild_inputs:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor.env_processor import LiberoProcessorStep
        from lerobot.envs.utils import preprocess_observation
        weight=next(Path(path) for path in lock['checkpoint_asset_sha256'] if Path(path).name=='model.safetensors')
        tokenizer=next(Path(path).parent for path in lock['tokenizer_sha256'] if Path(path).name=='tokenizer.model')
        config=PreTrainedConfig.from_pretrained(weight.parent);config.device='cpu';config.compile_model=False
        pre,_=make_pre_post_processors(config,pretrained_path=weight.parent,
            preprocessor_overrides={'tokenizer_processor':{'tokenizer_name':str(tokenizer)},'device_processor':{'device':'cpu'}})
        def batch(value):
            if isinstance(value,np.ndarray):return value[None]
            if isinstance(value,dict):return {k:batch(v) for k,v in value.items()}
            return value
    for summary_path in sorted((d/'blocks').glob('task-*/start-*/block_summary.json')):
        summary=json.loads(summary_path.read_text());task,initial=summary['task'],summary['initialization'];folder=summary_path.parent
        require(task in range(10) and initial in range(50) and summary['candidate_lock_sha256']==lock_hash,'Foreign completed block')
        rows={};raw_frame_cache={}
        for condition in conditions:
            path=folder/(condition+'.json.gz')
            with gzip.open(path,'rt') as f:row=json.load(f)
            require(row['task']==task and row['initialization']==initial and row['condition']==condition,'Foreign episode')
            require(row['gpu_uuid']==summary['gpu_uuid'] and row['candidate_lock_sha256']==lock_hash,'Mixed device/candidate')
            require(row['numerical_qa_acceptance_pending'] is True,'Premature numerical acceptance')
            calls,steps,trajectory=row['calls'],row['steps'],row['trajectory']
            require(0<len(steps)<=520 and len(trajectory)==len(steps)+1,'Incomplete action/state coverage')
            require(steps[-1]['terminated'] or steps[-1]['truncated'] or len(steps)==520,'Stopped early')
            require(row['success']==steps[-1]['success'],'Outcome differs from simulator steps')
            for frame in trajectory:
                raw_path=(d/frame['raw_frame']).resolve();require(raw_path.parent==(d/'frames').resolve(),'Raw frame escapes study')
                if str(raw_path) not in raw_frame_cache:
                    with gzip.open(raw_path,'rb') as f:raw=torch.load(f,map_location='cpu',weights_only=False)
                    require(digest(raw)==raw_path.name.split('.')[0],'Raw frame content address changed')
                    require(set(raw)=={'physical','controller','actuator_controls','rng','observation','derived'},'Missing raw state category')
                    expected={field:digest(raw[name]) for name,field in [('physical','sim_state_sha256'),('controller','controller_state_sha256'),('actuator_controls','actuator_control_sha256'),('rng','rng_state_sha256'),('observation','observation_sha256'),('derived','derived_sha256')]}
                    raw_frame_cache[str(raw_path)]=expected
                require(all(frame[k]==v for k,v in raw_frame_cache[str(raw_path)].items()),'Raw frame state/observation hashes differ')
                raw_frame_count+=1
            if (task,initial) in canonical:require({k:v for k,v in trajectory[0].items() if k!='raw_frame'}==canonical[(task,initial)],'Initial snapshot differs from independently qualified canonical state')
            for frame in trajectory:
                require(digest(np.asarray(frame['sim_state'],dtype=np.float64))==frame['sim_state_sha256'],'Raw physics trajectory changed')
            for i,call in enumerate(calls):
                size=[1,2,4,8][i%4] if 'dynamic' in condition else 1
                require(call['call']==i and call['batch_size']==size and call['position']==i%size,'Batch schedule differs')
                key=f'libero_10/{task}/{initial}/policy-call-{i}';seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
                require(call['noise_id']==key and call['noise_seed']==seed,'Condition-dependent request noise')
                normalized=np.asarray(call['normalized_actions'],dtype=np.float32)
                require(normalized.shape==(1,50,7) and hashlib.sha256(normalized.tobytes()).hexdigest()==call['normalized_sha256'],'Normalized model bytes changed')
                executed=[s['action'] for s in steps if s['policy_call']==i]
                require(0<len(executed)<=lock['action_steps'] and action_bytes(executed)==action_bytes(call['actions'][:len(executed)]),'Controls differ from predicted chunk')
                require(i==len(calls)-1 or len(executed)==lock['action_steps'],'Short nonterminal chunk')
                require(len(call['companion_ids'])==size-1 and all(not rid.startswith(f'libero_10/{task}/') for rid in call['companion_ids']),'Companions include target task or wrong count')
                raw_path=(d/call['raw_observation_file']).resolve();require(raw_path.is_relative_to(d.resolve()),'Raw input escapes study directory')
                raw=torch.load(raw_path,map_location='cpu',weights_only=False)
                require(digest(raw)==raw_path.stem,'Raw observation bytes changed')
                control_index=sum(1 for step in steps if step['policy_call']<i)
                require(digest(raw)==trajectory[control_index]['observation_sha256'],'Model input differs from raw trajectory frame')
                if pre is not None:
                    inputs=LiberoProcessorStep().observation(preprocess_observation(batch(raw)));inputs['task']=[trajectory[0]['instruction']]
                    inputs={k:v for k,v in pre(inputs).items() if isinstance(v,torch.Tensor)}
                    noise=torch.randn(1,50,32,generator=torch.Generator().manual_seed(seed))
                    require(digest({k:v.numpy() for k,v in inputs.items()}|{'noise':noise.numpy()})==call['target_input_sha256'],'Rebuilt input/noise differs')
                    reconstructed+=1
            rows[condition]=row
        snapshot=torch.load(folder/'initial_state.pt',map_location='cpu',weights_only=False);first=rows[base[0]]['trajectory'][0]
        for key,field in (('physical','sim_state_sha256'),('controller','controller_state_sha256'),('actuator_controls','actuator_control_sha256'),('rng','rng_state_sha256'),('observation','observation_sha256')):
            require(digest(snapshot[key])==first[field],'Canonical initial snapshot differs')
        require(len({digest(row['trajectory'][0]) for row in rows.values()})==1,'Initial physics/controller/RNG/observations differ')
        require(len({row['calls'][0]['target_input_sha256'] for row in rows.values()})==1,'Initial model inputs differ')
        for condition in base:
            for field in ('calls','steps','trajectory','success'):
                require(digest(rows[condition][field])==digest(rows[condition+'_repeat'][field]),f'A/A differs: {condition}/{field}')
        a,b=rows['patched_singleton'],rows['patched_dynamic']
        require(digest(a['trajectory'])==digest(b['trajectory']) and digest(a['steps'])==digest(b['steps']),'Invariant batching changes trajectory')
        require([c['normalized_sha256'] for c in a['calls']]==[c['normalized_sha256'] for c in b['calls']],'Invariant normalized chunks differ')
        a,b=rows['native_singleton'],rows['native_dynamic']
        first_mismatch=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if action_bytes(x['action'])!=action_bytes(y['action'])),None)
        if first_mismatch is not None:
            ac=a['calls'][a['steps'][first_mismatch]['policy_call']];bc=b['calls'][b['steps'][first_mismatch]['policy_call']]
            require(ac['target_input_sha256']==bc['target_input_sha256'] and digest(a['trajectory'][:first_mismatch+1])==digest(b['trajectory'][:first_mismatch+1]),'Native divergence lacks equal-input/prefix attribution')
        require(first_mismatch==summary['native_first_action_divergence'],'Summary divergence differs')
        outcomes={name:rows[name]['success'] for name in base}
        require(all(summary['success'][name]==rows[name]['success'] for name in conditions),'Summary outcomes differ')
        task_counts=counts.setdefault(str(task),{'paired_starts':0,'successes':{name:0 for name in base}});task_counts['paired_starts']+=1
        for name in base:task_counts['successes'][name]+=outcomes[name]
        initial_states.setdefault(str(task),set()).add(first['sim_state_sha256'])
        blocks.append(dict(task=task,initialization=initial,outcomes=outcomes,native_first_action_divergence=first_mismatch))
    for task,values in counts.items():
        n=values['paired_starts'];values['unique_initial_physics_states']=len(initial_states[task]);values['success_rate']={k:v/n for k,v in values['successes'].items()}
        values['BI_minus_native_singleton_percentage_points']=100*(values['successes']['patched_singleton']-values['successes']['native_singleton'])/n
        values['BI_minus_native_dynamic_percentage_points']=100*(values['successes']['patched_dynamic']-values['successes']['native_dynamic'])/n
    complete=len(blocks)==500 and len({(b['task'],b['initialization']) for b in blocks})==500
    report=dict(status='paired_rollouts_independently_audited_provisional_pending_numerical_QA',candidate_lock_sha256=lock_hash,
        completed_paired_starts=len(blocks),prescribed_paired_starts=500,episodes=8*len(blocks),by_task=counts,
        raw_inputs_and_noise_reconstructed=args.rebuild_inputs,reconstructed_policy_calls=reconstructed,
        all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks=True,cohort_rollouts_complete=complete,
        renderer_offscreen_samples=0,renderer_protocol_changed=True,default_renderer_results_must_not_be_pooled=True,raw_frames_verified=raw_frame_count,numerical_qa_acceptance_pending=True,success_rate_campaign_complete=False,blocks=blocks,auditor_sha256=sha(Path(__file__)))
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='blocks'},indent=2))
    if args.require_complete and not complete:raise SystemExit(1)


if __name__=='__main__':main()

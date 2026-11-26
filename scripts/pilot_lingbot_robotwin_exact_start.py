"""Eight-condition complete-episode pilot on an expert-qualified RoboTwin start."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from qualify_simpler_fixed_action_replay import digest


def tensor_record(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-cuda-rng',action='store_true')
    for name in ('source','checkpoint','qwen-metadata','robotwin-source','robotwin-python','expert-seed','companions','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    expert=json.loads(args.expert_seed.read_text())
    if not expert['response']['ok'] or not expert['response']['result']['eligible']:
        raise ValueError('Unqualified expert seed')
    selection=expert['response']['result']; specification=expert['request']
    if specification['task']!='adjust_bottle':raise ValueError('Task-state capture currently qualified for adjust_bottle only')
    os.environ.update(QWEN3VL_PATH=str(args.qwen_metadata),LINGBOT_VLA2_CHECKPOINT=str(args.checkpoint))
    sys.path.insert(0,str(args.source))
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters.lingbot_vla2 import LingBotVLA2Adapter
    class Adapter(LingBotVLA2Adapter):
        def _validate_paths(self):
            for path in (self.checkpoint/'model.safetensors.index.json',self.checkpoint.parent.parent.parent/'lingbotvla_cli.yaml'):
                if not path.is_file():raise FileNotFoundError(path)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    adapter=Adapter(); model=adapter.load_model()
    adapter.server.use_length=50
    companions=torch.load(args.companions,map_location='cpu',weights_only=False)
    worker=Path(__file__).with_name('robotwin_exact_worker.py')
    stderr=(args.output/'worker-stderr.log').open('w')
    stdout_log=(args.output/'worker-stdout.log').open('w')
    worker_command=[str(args.robotwin_python),str(worker),'--source',str(args.robotwin_source),'--output',str(args.output/'frames')]
    if args.capture_cuda_rng:worker_command.append('--capture-cuda-rng')
    proc=subprocess.Popen(worker_command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                          stderr=stderr,text=True,bufsize=1)
    def rpc(request):
        proc.stdin.write(json.dumps(request)+'\n');proc.stdin.flush()
        for line in proc.stdout:
            if 'ROBOTWIN_RESULT ' in line:
                response=json.loads(line.split('ROBOTWIN_RESULT ',1)[1])
                if not response['ok']:raise RuntimeError(response['error'])
                return response['result']
            stdout_log.write(line);stdout_log.flush()
        raise RuntimeError('RoboTwin worker terminated')
    def comparable(frame):return {k:v for k,v in frame.items() if k!='path'}
    def prepare(frame):
        raw=torch.load(frame['path'],map_location='cpu',weights_only=False)['observation']
        transformed=adapter.server._prepare_model_input({'task':selection['instruction'],
            'observation.state':raw['joint_action']['vector'],
            'observation.images.cam_high':raw['observation']['head_camera']['rgb'],
            'observation.images.cam_left_wrist':raw['observation']['left_camera']['rgb'],
            'observation.images.cam_right_wrist':raw['observation']['right_camera']['rgb']})
        return transformed
    conditions={'native_singleton':(False,False),'native_dynamic':(False,True),
                'patched_singleton':(True,False),'patched_dynamic':(True,True)}
    conditions.update({k+'_repeat':v for k,v in list(conditions.items())})
    runtime=dict(scope='complete_episode_pilot',model='lingbot_vla2',task=specification['task'],seed=specification['seed'],
        capture_cuda_rng=args.capture_cuda_rng,
        conditions=conditions,dynamic_cycle=[1,2,4,8],instruction=selection['instruction'],instruction_type=specification['instruction_type'],
        canonical_state=selection['canonical'],expert_seed_sha256=hashlib.sha256(args.expert_seed.read_bytes()).hexdigest(),
        native_contract='released FP32 operators with deterministic token-choice MoE shared by both modes; atomic MoE baseline separately measured',
        chunk_length=50,action_type='qpos',source_client='experiment/robotwin/eval_policy_client_lingbotvla.py',
        companion_contract='real frames from qualified single-episode numerical probe; not unrelated-task evidence',
        capture_contract='full physics/controller/task/RNG and RGB recorded after every action; released client renders at chunk boundaries',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),worker,Path(sys.modules[LingBotVLA2Adapter.__module__].__file__),args.companions)},
        paired_success_rate_campaign_complete=False)
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output/worker.name).write_bytes(worker.read_bytes())
    episodes={}
    try:
        for condition,(invariant,dynamic) in conditions.items():
            adapter.server.global_step=0;adapter.server.last_action_chunk=None;adapter.server.last_normalized_action_chunk=None
            frame=rpc(dict(specification,op='reset',episode_info=selection['episode_info'],instruction=selection['instruction']))
            if comparable(frame)!=comparable(selection['canonical']):raise RuntimeError('Canonical fresh-world reset mismatch')
            trajectory=[comparable(frame)];calls=[];frames=[frame['path']]
            while not frame['controller']['eval_success'] and frame['controller']['take_action_count']<frame['controller']['step_limit']:
                call=len(calls); transformed=prepare(frame)
                noise_id=f"{specification['task']}/{specification['seed']}/policy-call-{call}"
                seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
                target={key:transformed[key] for key in adapter.model_keys}
                target['noise']=torch.randn(50,55,generator=torch.Generator().manual_seed(seed))
                size=[1,2,4,8][call%4] if dynamic else 1;position=call%size
                indices=[(call+j)%len(companions) for j in range(size-1)]
                requests=[companions[j] for j in indices];requests.insert(position,target)
                adapter.reset_model(model)
                with torch.inference_mode(),set_batch_invariant_mode(invariant):
                    normalized=adapter.run_model(model,adapter.compose_batch(requests))
                    physical=adapter.server._unapply_batched_actions([transformed],normalized[position:position+1])
                if set(physical)!={'action'}:raise ValueError('Unexpected official action mapping')
                chunk=np.asarray(physical['action'][0],dtype=np.float32)
                if chunk.shape!=(50,14) or not np.isfinite(chunk).all():raise ValueError('Invalid model chunk')
                inputs={k:tensor_record(v) for k,v in target.items()}
                record=dict(policy_call=call,batch_size=size,position=position,companion_indices=indices,
                            noise_id=noise_id,noise_seed=seed,model_inputs=inputs,target_input_sha256=digest(inputs),
                            normalized_actions=normalized[position].tolist(),physical_chunk=chunk.tolist(),executed_actions=[])
                for action in chunk:
                    frame=rpc(dict(op='step',action=action.tolist()))
                    record['executed_actions'].append(action.tolist());trajectory.append(comparable(frame));frames.append(frame['path'])
                    if frame['controller']['eval_success'] or frame['controller']['take_action_count']>=frame['controller']['step_limit']:break
                calls.append(record)
                print(f'CALL {condition}/{call}: simulator_step={frame["controller"]["take_action_count"]}',flush=True)
            episode=dict(task=specification['task'],seed=specification['seed'],condition=condition,calls=calls,
                         trajectory=trajectory,raw_frame_paths=frames,success=frame['controller']['eval_success'])
            with (args.output/'episodes.jsonl').open('a') as out:out.write(json.dumps(episode)+'\n')
            episodes[condition]=episode
            print(f'FINISHED {condition}: success={episode["success"]}, steps={len(trajectory)-1}',flush=True)
    finally:
        proc.stdin.close();proc.wait();stderr.close();stdout_log.close()
    repeat_exact={k:all(episodes[k][field]==episodes[k+'_repeat'][field] for field in ('calls','trajectory','success')) for k in list(conditions)[:4]}
    a,b=episodes['patched_singleton'],episodes['patched_dynamic']
    invariant_exact=a['trajectory']==b['trajectory'] and all(x['normalized_actions']==y['normalized_actions'] and x['physical_chunk']==y['physical_chunk'] for x,y in zip(a['calls'],b['calls'])) and len(a['calls'])==len(b['calls'])
    summary=dict(scope='complete_episode_pilot',episodes=8,all_AA_exact=all(repeat_exact.values()),repeat_checks=repeat_exact,
                 invariant_trajectories_exact=invariant_exact,paired_success_rate_campaign_complete=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['all_AA_exact'] or not invariant_exact:raise RuntimeError('Pilot repeat/batch trajectory equality failed; preserve raw evidence')


if __name__=='__main__':main()

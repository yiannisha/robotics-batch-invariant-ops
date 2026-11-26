"""Recover historical pilot input tensors only through exact archived hash matches.

No neural inference. Replay retained executed commands, check each observation
against historical evidence, and rebuild preprocessing plus indexed CPU noise.
Partial recovery never establishes complete historical exposure coverage.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from scripts.audit_lingbot_campaign_corpus_v2 import preprocessing_server, signature, KEYS
from scripts.collect_lingbot_robotwin_campaign_replay_v2 import Worker, retain, write_json
from scripts.qualify_simpler_fixed_action_replay import digest


def require(value, message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_request(request, expected):
    actual = {k:signature(v) for k,v in request.items()}
    require(set(actual) == set(expected), 'Historical request field coverage differs')
    for key in actual:
        require(actual[key] == expected[key], 'Historical tensor signature differs: ' + key)
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','qwen-metadata','robotwin-source','robotwin-python','worker',
                 'episodes','expert-seed','pilot-runtime','history-index','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--condition',action='append',required=True)
    a=p.parse_args();require(not a.output.exists(),'Retain previous recovery attempts');a.output.mkdir(parents=True)
    payload=gzip.decompress(a.episodes.read_bytes());history=json.loads(a.history_index.read_text())
    historical_path=next(k for k in history['sources_sha256'] if k.endswith('lingbot-closed-loop-pilot-001/episodes.jsonl'))
    require(hashlib.sha256(payload).hexdigest()==history['sources_sha256'][historical_path], 'Archived historical pilot bytes differ')
    (a.output/'episodes.jsonl').write_bytes(payload)
    episodes=[json.loads(line) for line in payload.decode().splitlines()];selected=[e for e in episodes if e['condition'] in a.condition]
    require(len(selected)==len(set(a.condition)), 'Missing or duplicate requested historical conditions')
    expert=json.loads(a.expert_seed.read_text());runtime=json.loads(a.pilot_runtime.read_text())
    require(sha(a.expert_seed)==runtime['expert_seed_sha256'], 'Historical expert selection differs')
    specification=expert['request'];selection=expert['response']['result']
    torch.set_num_threads(8);server=preprocessing_server(a.source,a.checkpoint,a.qwen_metadata)
    records=[];progress=dict(scope='historical tensor recovery only; not a policy or paired experiment',
        historical_pilot_sha256=hashlib.sha256(payload).hexdigest(),historical_exposure_history_sha256=sha(a.history_index),
        expert_seed_sha256=sha(a.expert_seed),requested_conditions=a.condition,recovered_requests=records,
        neural_inference_performed=False,complete_historical_exposure_coverage=False)
    progress['source_sha256']={str(Path(__file__).resolve()):sha(Path(__file__)),
        **{str(path.resolve()):sha(path) for path in a.worker.parent.glob('*.py')}}
    write_json(a.output/'progress.json',progress)
    for episode in selected:
        worker=Worker(a.robotwin_python,a.worker,a.robotwin_source,a.output/'replays'/episode['condition'])
        try:
            frame=worker.rpc(dict(specification,op='reset',episode_info=selection['episode_info'],instruction=selection['instruction']))['result']
            steps=0
            for call in episode['calls']:
                old=episode['trajectory'][steps]
                require(frame['observation_sha256']==old['observation_sha256'] and frame['controller']==old['controller'], 'Historical observation/controller differs before call')
                raw=torch.load(frame['path'],map_location='cpu',weights_only=False);obs=raw['observation']
                full=server._prepare_model_input({'task':selection['instruction'],'observation.state':obs['joint_action']['vector'],
                    'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
                    'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
                    'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb']})
                request={k:full[k] for k in KEYS};request['noise']=torch.randn(50,55,generator=torch.Generator().manual_seed(call['noise_seed']),dtype=torch.float32)
                signatures=check_request(request,call['model_inputs']);relative=retain(a.output,'requests',request)
                records.append(dict(condition=episode['condition'],policy_call=call['policy_call'],frame_index=steps,
                    raw_request=relative,request_file_sha256=sha(a.output/relative),input_signatures=signatures,
                    archived_signatures_exact=True,observation_sha256=frame['observation_sha256']))
                write_json(a.output/'progress.json',progress)
                print('RECOVERED',episode['condition'],call['policy_call'],flush=True)
                for action in call['executed_actions']:
                    frame=worker.rpc(dict(op='step',action=action))['result'];steps+=1
                    require(frame['observation_sha256']==episode['trajectory'][steps]['observation_sha256'] and
                            frame['controller']==episode['trajectory'][steps]['controller'], 'Historical replay observation/controller differs')
            require(steps+1==len(episode['trajectory']), 'Historical replay incomplete')
        finally:worker.close()
    progress['status']='requested_historical_pilot_tensors_recovered_by_exact_archived_signatures'
    progress['recovered_request_count']=len(records);write_json(a.output/'recovery_report.json',progress)


if __name__=='__main__':main()

"""Rebuild LingBot native corpus inputs/actions using the official transforms.

The server's model constructor/weight loader are replaced only to instantiate
its CPU preprocessing/postprocessing without a neural network. No model output
is generated, and this does not certify numerical inference or heldout freshness.
"""
import argparse
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest

KEYS=('images','img_masks','lang_tokens','lang_masks','state','image_grid_thw')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(value):
    array=value.detach().cpu().contiguous()
    return dict(dtype=str(array.dtype),shape=list(array.shape),sha256=hashlib.sha256(array.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def content_hash(value):
    if isinstance(value,torch.Tensor):return digest({'torch_tensor_signature':signature(value)})
    def visit(item):
        if isinstance(item,torch.Tensor):return {'torch_tensor_signature':signature(item)}
        if isinstance(item,dict):return {key:visit(v) for key,v in item.items()}
        if isinstance(item,(list,tuple)):return [visit(v) for v in item]
        return item
    return digest(visit(value))


def preprocessing_server(source,checkpoint,metadata):
    os.environ['QWEN3VL_PATH']=str(metadata);sys.path.insert(0,str(source))
    from deploy import lingbot_vla_v2_policy as official
    from unittest.mock import patch
    server=official.LingbotVLAv2Server.__new__(official.LingbotVLAv2Server)
    server.use_compile=False;server.use_bf16=False;server.use_fp32=True;server.robot_norm_path=None
    server.load_model_weights=lambda *args,**kwargs:None
    def no_inference(*args,**kwargs):raise RuntimeError('Neural model inference is forbidden in the input reconstruction auditor')
    def transforms_only_policy(*args,**kwargs):return SimpleNamespace(model=SimpleNamespace(sample_actions=no_inference),feature_transform=None)
    with contextlib.chdir(source),patch.object(official,'LingBotVlaV2InferencePolicy',transforms_only_policy):
        server.vla=server.load_vla(str(checkpoint));server.reset('robotwin')
    return server


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','source','checkpoint','qwen-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args();d=a.directory
    payload=(d/'collection_manifest.json').read_bytes();m=json.loads(payload);lock_path=d/'candidate_lock.json';lock=json.loads(lock_path.read_text())
    snapshot=d/'corpus_audit_snapshots'/(hashlib.sha256(payload).hexdigest()+'.json')
    snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(payload)
    require(sha(lock_path)==m['candidate_lock_sha256'],'Native corpus candidate changed')
    for path,expected in lock['source_sha256'].items():require(sha(d/'source_snapshots'/expected)==expected,'Archived producer source changed')
    require(lock['strict_loaded_tensors']==1708 and lock['strict_loaded_elements']==6375907511 and lock['batch_invariant_model_evaluated'] is False,'Wrong model or collection mode')
    require(len(lock['prescribed_tasks'])==50 and lock['episodes_per_task']==4,'Wrong corpus roster')
    require(lock.get('precision_guard')=='post_constructor_tf32_off_and_each_call_checked','Missing corrected precision contract')
    require(all(lock.get('runtime',{}).get(k) is False for k in ('tf32_matmul','tf32_cudnn','cudnn_benchmark')),'Wrong corrected corpus precision')
    require(not a.require_complete or len(m['episodes'])==200,'Full source corpus is incomplete')
    torch.set_num_threads(8);server=preprocessing_server(a.source,a.checkpoint,a.qwen_metadata)
    verified_files={};frames=0;calls=0;selected=0;episode_map={};observations_by_episode={}
    def read(relative):
        path=(d/relative).resolve();require(path.is_relative_to(d.resolve()),'Raw input path escapes corpus')
        with gzip.open(path,'rb') as stream:value=torch.load(stream,map_location='cpu',weights_only=False)
        require(content_hash(value)==path.name.split('.')[0],'Content-addressed native corpus value changed')
        verified_files[str(path)]=sha(path);return value
    def transformed(raw,instruction):
        obs=raw['observation']
        return server._prepare_model_input({'task':instruction,'observation.state':obs['joint_action']['vector'],
            'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
            'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
            'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb']})
    for e in m['episodes']:
        path=d/e['source_episode_file'];require(sha(path)==e['sha256'],'Source native episode changed')
        with gzip.open(path,'rt') as stream:episode=json.load(stream)
        key=(e['task'],e['split'],e['ordinal']);require(key not in episode_map,'Duplicate source episode')
        require(episode['candidate_lock_sha256']==m['candidate_lock_sha256'] and (episode['task'],episode['split'],episode['ordinal'])==key,'Foreign native episode')
        require(key[0] in lock['prescribed_tasks'] and key[1] in ('diagnostic','heldout') and key[2] in (0,1),'Unprescribed native episode')
        require(e['seed']==episode['seed'] and (episode['seed']>=200000 if e['split']=='heldout' else 100000<=episode['seed']<200000),'Wrong released evaluation seed phase')
        trajectory=episode['trajectory'];steps=episode['steps'];require(len(trajectory)==len(steps)+1 and steps,'Incomplete source trajectory')
        require(trajectory[0]==episode['selection']['canonical'],'Native reset differs from selected canonical state')
        require(episode['success']==e['success']==trajectory[-1]['controller']['eval_success'],'Native outcome differs')
        require(e['success'] or trajectory[-1]['controller']['take_action_count']==trajectory[-1]['controller']['step_limit'],'Failed native episode stopped early')
        for index,frame in enumerate(trajectory):
            raw=read(frame['raw_frame'])
            require(raw['controller']==frame['controller'] and raw['controller']['take_action_count']==index,'Raw controller/step differs')
            for field in ('observation','physics','rng','task_state'):require(digest(raw[field])==frame[field+'_sha256'],'Raw frame category changed')
            frames+=1
        cursor=0
        for i,call in enumerate(episode['calls']):
            require(all(call.get(k) is False for k in ('tf32_matmul','tf32_cudnn','cudnn_benchmark')),'Missing or changed per-call precision')
            require(call['call']==i and call['frame_index']==cursor,'Source call/control alignment differs')
            noise_id=f'{e["task"]}/{e["seed"]}/policy-call-{i}';seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
            require(call['noise_id']==noise_id and call['noise_seed']==seed,'Indexed native noise schedule differs')
            request=read(call['raw_request']);full=transformed(read(trajectory[cursor]['raw_frame']),episode['instruction'])
            expected={key:full[key] for key in KEYS};expected['noise']=torch.randn(50,55,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
            require(set(request)==set(expected),'Unexpected model input field')
            for field in expected:require(signature(request[field])==signature(expected[field])==call['target_input_signatures'][field],'Rebuilt native model input differs: '+field)
            normalized=read(call['normalized_output']);physical=server._unapply_batched_actions([full],normalized)
            chunk=np.asarray(physical['action'][0],dtype=np.float32);actual=np.asarray(call['physical_chunk'],dtype=np.float32)
            require(chunk.shape==(50,14) and chunk.tobytes()==actual.tobytes(),'Official physical action mapping differs')
            count=min(50,len(steps)-cursor);executed=np.asarray([s['action'] for s in steps[cursor:cursor+count]],dtype=np.float32)
            require(all(s['call']==i for s in steps[cursor:cursor+count]) and executed.tobytes()==chunk[:count].tobytes(),'Executed source commands differ')
            cursor+=count;calls+=1
        require(cursor==len(steps),'Missing source commands');episode_map[key]=episode;observations_by_episode[e['source_episode_file']]=[]
    for row in m['observations']:
        episode=next(value for value in episode_map.values() if value['task']==row['task'] and value['seed']==row['seed'])
        index=row['frame_index'];require(0<=index<len(episode['steps']),'Selected source frame is not pre-action')
        require(row['raw_frame']==episode['trajectory'][index]['raw_frame'],'Selected raw frame differs')
        request=read(row['request_file']);full=transformed(read(row['raw_frame']),episode['instruction'])
        require(set(request)==set(KEYS),'Unexpected selected model input field')
        for key in KEYS:require(signature(request[key])==signature(full[key])==row['input_signatures'][key],'Rebuilt selected model input differs: '+key)
        observations_by_episode[row['source_episode_file']].append(index);selected+=1
    for e in m['episodes']:
        episode=episode_map[(e['task'],e['split'],e['ordinal'])];n=len(episode['steps']);first=1 if e['split']=='heldout' and n>1 else 0
        expected=np.linspace(first,n-1,min(8 if e['split']=='heldout' else 2,n-first),dtype=int).tolist()
        require(observations_by_episode[e['source_episode_file']]==expected and e['selected_observations']==len(expected),'Selection or short-episode handling differs')
    report=dict(status='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt',manifest_snapshot=str(snapshot.resolve()),manifest_snapshot_sha256=hashlib.sha256(payload).hexdigest(),
        candidate_lock_sha256=m['candidate_lock_sha256'],completed_native_episodes=len(episode_map),prescribed_native_episodes=200,
        all_prescribed_native_episodes_rebuilt=len(episode_map)==200,raw_frames_verified=frames,model_calls_rebuilt=calls,selected_inputs_rebuilt=selected,
        corrected_precision_records_verified=True,official_cpu_transforms_only_no_neural_inference=True,raw_files_sha256=verified_files,
        auditor_sha256=sha(Path(__file__)),numerical_and_heldout_acceptance_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='raw_files_sha256'},indent=2))


if __name__=='__main__':main()

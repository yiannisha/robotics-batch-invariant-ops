"""Rebuild the fresh numerical holdout candidate; exposure audit is a separate gate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from collect_rdt_maniskill_campaign_replay import TASKS


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    d=args.directory;manifest_path=d/'replay_manifest.json';m=json.loads(manifest_path.read_text())
    require(len(m['observations'])==1000 and len(m['episodes'])==100,'Incomplete corpus')
    noise_path=d/m['noise_file'];require(sha(noise_path)==m['noise_file_sha256'],'Noise file changed')
    noise=np.load(noise_path,allow_pickle=False,mmap_mode='r')
    require(noise.shape==(1000,3,64,128) and noise.dtype==np.float32,'Noise shape/dtype differs')
    require(sha(d/'runtime.json')==m['runtime_sha256'],'Runtime changed')
    runtime=json.loads((d/'runtime.json').read_text());starts=runtime['initialization_ids']
    require(len(starts)==20 and starts==list(range(starts[0],starts[0]+20)) and min(starts)>=50,'Source starts overlap prescribed paired cohort')
    require(runtime['batch_invariant_model_evaluated'] is False and runtime['replay_noise_seed_base']==1980000,'Invalid collection/exposure protocol')
    require(sha(d/'collect_rdt_unexposed_holdout_replay.py')==runtime['runner_sha256'],'Archived collector source changed')
    for asset in runtime['artifact_files']:require(sha(Path(asset['path']))==asset['sha256'],'Pinned model artifact changed')
    expected={(t,i) for t in TASKS for i in starts};by_episode={}
    for entry in m['episodes']:
        key=(entry['task'],entry['initialization']);require(key in expected and key not in by_episode,'Unexpected/duplicate episode')
        path=d/entry['source_episode_file'];require(sha(path)==entry['sha256'],'Raw episode changed')
        episode=json.loads(path.read_text());by_episode[key]=episode
        require((episode['task'],episode['initialization'])==key and episode['seed']==20241201+key[1],'Source task/seed differs')
        require(episode['reward_mode']==('sparse' if key[0]=='PlugCharger-v1' else 'dense'),'Reward mode differs')
        frames,calls,steps=episode['frames'],episode['calls'],episode['steps']
        require(len(frames)==len(steps)+1 and 10<=len(steps)<=400,'Incomplete source frames')
        require(episode['success']==entry['success'],'Source outcome differs')
        last=steps[-1]
        require(episode['success']==((last['terminated'] or last['truncated']) and last['success']),'Source stopping semantics differ')
        require(episode['success'] or len(steps)==400,'Incomplete failed episode')
        for frame in frames:
            require(sha(d/frame['image_file'])==frame['image_sha256'],'Raw episode PNG changed')
            require(np.asarray(frame['proprio']).shape==(8,),'Invalid robot state')
        cursor=0
        for i,call in enumerate(calls):
            require(call['policy_call']==i and call['frame_index']==cursor,'Call index/frame differs')
            noise_id=f'{key[0]}/{key[1]}/policy-call-{i}'
            require(call['noise_id']==noise_id and call['noise_seed']==int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little'),'Source request noise indexing differs')
            chunk=np.asarray(call['raw_action_chunk'],dtype=np.float32)
            require(chunk.shape==(64,8) and np.isfinite(chunk).all(),'Invalid source action chunk')
            count=min(16,len(steps)-cursor)
            actual=np.asarray([step['action'] for step in steps[cursor:cursor+count]],dtype=np.float32)
            require(exact(actual,chunk[::4][:count]),'Source controls differ from released action stride')
            require(all(step['policy_call']==i for step in steps[cursor:cursor+count]),'Step call attribution differs')
            cursor+=count
        require(cursor==len(steps),'Source actions missing')
    require(set(by_episode)==expected,'Episode roster incomplete')
    selected=Counter();splits=Counter();indices=set()
    for row in m['observations']:
        index=row['observation_index'];key=(row['task'],row['episode_id']);frame_index=row['frame_index']
        require(index not in indices and 0<=index<1000,'Duplicate/invalid observation ID');indices.add(index)
        episode=by_episode[key];frames=episode['frames']
        chosen=np.linspace(0,len(frames)-2,10,dtype=int).tolist()
        require(frame_index in chosen,'Selection is not evenly spaced pre-action frames')
        selected[(key,frame_index)]+=1;split='diagnostic' if key[1]<starts[0]+4 else 'heldout'
        require(row['split']==split,'Split leakage');splits[split]+=1
        path=d/row['request_file'];require(sha(path)==row['request_sha256'],'Processed request changed')
        request=torch.load(path,map_location='cpu',weights_only=False)
        current=np.asarray(Image.open(d/frames[frame_index]['image_file']).convert('RGB'))
        previous=np.asarray(Image.open(d/frames[frame_index-1]['image_file']).convert('RGB')) if frame_index else np.zeros_like(current)
        require(exact(request['image_history'],np.stack([previous,current])),'Rebuilt RGB history differs')
        require(exact(request['proprio'],np.asarray(frames[frame_index]['proprio'],dtype=np.float32)),'Rebuilt proprio differs')
        require(request['task_name']==key[0] and request['history_valid']==[frame_index>0,True],'Task/history validity differs')
        require(row['source_episode_file']==m['episodes'][list(by_episode).index(key)]['source_episode_file'],'Observation episode reference differs')
        require(sha(d/row['source_episode_file'])==row['source_episode_sha256'],'Observation source changed')
        for j in range(3):
            regenerated=torch.randn(64,128,generator=torch.Generator().manual_seed(1980000+3*index+j)).numpy()
            require(exact(regenerated,noise[index,j]) and hashlib.sha256(regenerated.tobytes()).hexdigest()==row['noise_sha256'][j],'Replay noise differs')
    require(len(selected)==1000 and all(n==1 for n in selected.values()),'Duplicate selected frames')
    require(dict(splits)=={'diagnostic':200,'heldout':800},'Split counts differ')
    report=dict(status='complete_corpus_independently_verified',observations=1000,episodes=100,split_counts=dict(splits),
                raw_inputs_rebuilt=True,all_noise_regenerated=True,all_source_executed_controls_verified=True,
                initialization_ids=starts,heldout_acceptance_pending_observation_disjointness_audit=True,
                manifest_sha256=sha(manifest_path),auditor_sha256=sha(Path(__file__)))
    audit_path=d/'corpus_audit.json'
    audit_path.write_text(json.dumps(report,indent=2)+'\n')
    # Preserve the raw collection manifest. The evaluated manifest receives
    # canonical request identities and a split-isolated companion schedule.
    from scripts.run_model_replay_invariance import split_local_companion_rows
    evaluated=json.loads(json.dumps(m))
    evaluated.update(model_id='rdt_maniskill_exact',fixture_root=str(d.resolve()),noise_file=str(noise_path.resolve()),
                     corpus_audit_path=str(audit_path.resolve()),corpus_audit_sha256=sha(audit_path))
    for row in evaluated['observations']:
        index=row['observation_index'];episode=by_episode[(row['task'],row['episode_id'])]
        ordinal=np.linspace(0,len(episode['frames'])-2,10,dtype=int).tolist().index(row['frame_index'])
        row.update(source_observation_index=index,replay_index=index,observation_index=ordinal,
                   request_ids=[f'maniskill/{row["task"]}/{row["episode_id"]}/observation-{ordinal}/noise-{j}' for j in range(3)],
                   noise=[dict(sha256=value,generator_seed=1980000+3*index+j) for j,value in enumerate(row['noise_sha256'])],
                   image_sha256={str(i):sha(d/path) for i,path in enumerate(row['image_files']) if path is not None})
    for row in evaluated['observations']:
        companions=split_local_companion_rows(evaluated['observations'],row,row['replay_index'],31)
        require(len({r['replay_index'] for r in companions})==31,'Duplicate companion observations')
        require(all(r['split']==row['split'] and (r['task'],r['episode_id'])!=(row['task'],row['episode_id']) for r in companions),'Companion split/episode leakage')
        row['companion_replay_indices']=[r['replay_index'] for r in companions]
    evaluated['companion_schedule']=dict(version='split_local_unrelated_observations_v1',max_companions=31,
        description='Same split, outside target episode; distinct episodes prioritized, then distinct observations.')
    (d/'audited_manifest.json').write_text(json.dumps(evaluated,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

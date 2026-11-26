"""Independently reconstruct fresh CogACT cold-start corpus evidence."""
import argparse
import gzip
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from scripts.qualify_simpler_fixed_action_replay import digest
TASKS=('pick_can','move_near','open_drawer','close_drawer')
PROTOCOL='fresh_seeded_worker_reconstruction_exact_canonical_snapshot_comparison'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def same(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    return a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','cogact-source','openvla-source'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();d=args.directory
    sys.path[:0]=[str(args.cogact_source),str(args.openvla_source)]
    from sim_cogact.adaptive_ensemble import AdaptiveEnsembler
    from scripts.run_cogact_paired_server import RequestState
    from scripts.run_model_replay_invariance import split_local_companion_rows
    manifest=d/'replay_manifest.json';m=json.loads(manifest.read_text());runtime=json.loads((d/'runtime.json').read_text())
    require(sha(d/'runtime.json')==m['runtime_sha256'],'Runtime changed')
    require(runtime['language_tokens']==64,'Controlled input extent differs')
    require(runtime['reset_protocol']==PROTOCOL,'Wrong cold reconstruction protocol')
    starts=runtime['initialization_ids'];require(starts==list(range(50,75)),'Wrong fresh start roster')
    for path,identity in runtime['source_sha256'].items():require(sha(Path(path))==identity,'Source changed')
    require(len(m['observations'])==1000 and len(m['episodes'])==100,'Incomplete corpus')
    noise_path=d/m['noise_file'];require(sha(noise_path)==m['noise_file_sha256'],'Noise archive changed')
    noise=np.load(noise_path,allow_pickle=False,mmap_mode='r')
    require(noise.shape==(1000,3,1,16,7) and noise.dtype==np.float32,'Noise shape/dtype differs')
    episodes={};expected={(task,i) for task in TASKS for i in starts}
    for entry in m['episodes']:
        key=(entry['task'],entry['initialization']);require(key in expected and key not in episodes,'Foreign/duplicate episode')
        path=d/entry['source_episode_file'];require(sha(path)==entry['sha256'],'Episode changed')
        e=json.loads(path.read_text());episodes[key]=e
        require((e['task'],e['initialization'])==key and e['success']==entry['success'],'Episode identity/outcome differs')
        require(e['restore_checks'] and all(e['restore_checks'].values()),'Inexact cold reconstruction')
        require(e['reset_protocol']==PROTOCOL and e['snapshot']['reset_protocol']==PROTOCOL,'Episode reset protocol differs')
        require(e['snapshot']['canonical_hashes']==e['frames'][0]['observable_hashes'],'Cold initial state differs')
        require(e['snapshot']['canonical_raw_frame']==e['frames'][0]['raw_frame'],'Cold raw canonical differs')
        frames,calls=e['frames'],e['calls'];require(len(frames)==len(calls)+1 and 10<=len(calls)<=200,'Incomplete source episode')
        require(calls[-1]['truncated'] and e['success']==calls[-1]['success'],'Incomplete source horizon/outcome')
        controller=RequestState('google_robot',AdaptiveEnsembler)
        for i,call in enumerate(calls):
            require(call['policy_call']==i and call['frame_index']==i,'Source call/frame order differs')
            noise_id=f'{key[0]}/{key[1]}/policy-call-{i}'
            require(call['noise_id']==noise_id and call['noise_seed']==int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little'),'Source noise indexing differs')
            chunk=np.asarray(call['raw_action_chunk'],dtype=np.float64)
            require(chunk.shape==(16,7) and np.isfinite(chunk).all(),'Invalid source action chunk')
            selected,action=controller.process(chunk)
            require(same(selected,call['selected_action']) and same(action,call['environment_action']),'Released controller reconstruction differs')
        for frame in frames:
            raw_path=d/frame['raw_frame']
            with gzip.open(raw_path,'rb') as stream:raw=torch.load(stream,map_location='cpu',weights_only=False)
            require(digest(raw)==raw_path.name.split('.')[0],'Raw content address differs')
            require({k+'_sha256':digest(v) for k,v in raw.items()}==frame['observable_hashes'],'Raw expanded state differs')
            require(digest(raw['observation'])==frame['observation_sha256'],'Raw observation differs')
            raw_rgb=next(iter(raw['observation']['image'].values()))['rgb']
            require(hashlib.sha256(raw_rgb.tobytes()).hexdigest()==frame['raw_rgb_sha256'],'Raw frame RGB differs')
            path=d/frame['image_file'];require(sha(path)==frame['image_sha256'],'Raw PNG changed')
            rgb=np.asarray(Image.open(path).convert('RGB'))
            require(hashlib.sha256(rgb.tobytes()).hexdigest()==frame['raw_rgb_sha256'],'PNG RGB bytes differ')
    require(set(episodes)==expected,'Episode roster incomplete')
    counts=Counter();selected=Counter();indices=set();evaluated=json.loads(json.dumps(m))
    for row in evaluated['observations']:
        index=row['observation_index'];key=(row['task'],row['episode_id']);frame=row['frame_index'];episode=episodes[key]
        require(index not in indices and 0<=index<1000,'Duplicate/invalid request index');indices.add(index)
        selection=np.linspace(0 if key[1]<55 else 1,len(episode['calls'])-1,10,dtype=int).tolist()
        require(frame in selection,'Selection differs from evenly spaced pre-action frames');ordinal=selection.index(frame)
        selected[(key,frame)]+=1
        split='diagnostic' if key[1]<55 else 'heldout';require(row['split']==split,'Split leakage');counts[split]+=1
        original=episode['frames'][frame]
        for field in ('image_file','image_sha256','raw_rgb_sha256'):require(row[field]==original[field],'Replayed image differs from source')
        require(row['instruction']==episode['calls'][frame]['instruction'],'Replayed instruction differs')
        require(sha(d/row['source_episode_file'])==row['source_episode_sha256'],'Observation source changed')
        require(row['request_ids']==[f'simpler/{key[0]}/{key[1]}/observation-{frame}/noise-{j}' for j in range(3)],'Request IDs differ')
        for j in range(3):
            value=torch.randn(1,16,7,generator=torch.Generator().manual_seed(1990000+3*index+j)).numpy()
            require(value.tobytes()==noise[index,j].tobytes() and hashlib.sha256(value.tobytes()).hexdigest()==row['noise_sha256'][j],'Request noise differs')
        row.update(source_observation_index=index,replay_index=index,observation_index=ordinal,
                   noise=[dict(sha256=value,generator_seed=1990000+3*index+j) for j,value in enumerate(row['noise_sha256'])])
    require(len(selected)==1000 and all(v==1 for v in selected.values()),'Repeated selected source frames')
    require(dict(counts)=={'diagnostic':200,'heldout':800},'Split counts differ')
    for row in evaluated['observations']:
        companions=split_local_companion_rows(evaluated['observations'],row,row['replay_index'],31)
        require(len({r['replay_index'] for r in companions})==31,'Repeated companion observations')
        require(all(r['split']==row['split'] and (r['task'],r['episode_id'])!=(row['task'],row['episode_id']) for r in companions),'Companion split/episode leak')
        row['companion_replay_indices']=[r['replay_index'] for r in companions]
    report=dict(status='complete_corpus_independently_verified',episodes=100,observations=1000,split_counts=dict(counts),
                all_raw_rgb_rehashed=True,all_expanded_raw_states_rehashed=True,reset_protocol=PROTOCOL,heldout_input_disjointness_still_pending=True,all_controller_actions_recomputed=True,all_request_noise_regenerated=True,
                all_companions_split_local=True,manifest_sha256=sha(manifest),auditor_sha256=sha(Path(__file__)),
                native_contract=runtime['native_contract'],paired_success_rate_campaign_complete=False)
    audit=d/'corpus_audit.json';audit.write_text(json.dumps(report,indent=2)+'\n')
    evaluated.update(model_id='cogact_simpler_exact',fixture_root=str(d.resolve()),noise_file=str(noise_path.resolve()),
        corpus_audit_path=str(audit.resolve()),corpus_audit_sha256=sha(audit),
        companion_schedule=dict(version='split_local_unrelated_observations_v1',max_companions=31,
            description='Same split, outside target episode; distinct episodes prioritized, then distinct observations.'))
    (d/'audited_manifest.json').write_text(json.dumps(evaluated,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

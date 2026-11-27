"""Derive prospective heldout inputs from audited frames without neural inference.

Preserve source episodes and diagnostic inputs. For each of eight evenly spaced
anchors, take the nearest frame whose prepared identity was not consumed by
native collection, diagnostics or a previous heldout selection. Ties use the
lower frame index. Preserve rejected candidates and every original selection.
This is not a final historical/paired exposure seal.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.audit_lingbot_campaign_corpus_v2 import KEYS,preprocessing_server,signature,content_hash

VERSION='native_diagnostic_exclusion_nearest_anchor_v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(signatures):
    return hashlib.sha256(json.dumps({k:signatures[k] for k in KEYS},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def select(steps,exposed,prepare):
    require(steps>8,'Need eight non-reset pre-action frames; preserve short episode')
    chosen=[];trace=[]
    for anchor in np.linspace(1,steps-1,8,dtype=int).tolist():
        attempts=[]
        for index in sorted(range(1,steps),key=lambda i:(abs(i-anchor),i)):
            request=prepare(index);key=identity({k:signature(request[k]) for k in KEYS})
            seen=key in exposed
            attempts.append(dict(frame_index=index,prepared_input_identity=key,already_exposed=seen))
            if not seen:
                chosen.append(index);exposed.add(key);break
        else:raise ValueError('Fewer than eight distinct unexposed retained inputs; do not filter episode')
        trace.append(dict(anchor=anchor,attempts=attempts))
    return chosen,trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','audit','source','checkpoint','qwen-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit=json.loads(a.audit.read_text())
    require(audit['status']=='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt'
            and audit['corrected_precision_records_verified'] is True,'Independently rebuilt corrected corpus required')
    snapshot=Path(audit['manifest_snapshot']);require(sha(snapshot)==audit['manifest_snapshot_sha256'],'Source snapshot changed')
    original=json.loads(snapshot.read_text());require(len(original['episodes'])==audit['completed_native_episodes'],'Source scope changed')
    require(sha(a.directory/'candidate_lock.json')==audit['candidate_lock_sha256']==original['candidate_lock_sha256'],'Source lock changed')
    require(not a.output.exists(),'Preserve prior derived attempts');a.output.mkdir(parents=True)
    def link(source,relative,expected):
        target=a.output/relative;require(source.resolve().is_relative_to(a.directory.resolve()),'Source escapes corpus')
        require(sha(source)==expected,'Audited source bytes changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(source,target)
        require(sha(target)==expected,'Derived retained bytes changed')
    for file,h in audit['raw_files_sha256'].items():
        path=Path(file);link(path,path.resolve().relative_to(a.directory.resolve()),h)
    link(a.directory/'candidate_lock.json',Path('candidate_lock.json'),audit['candidate_lock_sha256'])
    lock=json.loads((a.directory/'candidate_lock.json').read_text())
    for h in set(lock['source_sha256'].values()):link(a.directory/'source_snapshots'/h,Path('source_snapshots')/h,h)
    episodes={};exposed=set()
    for e in original['episodes']:
        path=a.directory/e['source_episode_file'];link(path,Path(e['source_episode_file']),e['sha256'])
        with gzip.open(path,'rt') as f:episode=json.load(f)
        episodes[e['source_episode_file']]=episode
        exposed.update(identity(call['target_input_signatures']) for call in episode['calls'])
    for row in original['observations']:
        if row['split']=='diagnostic':exposed.add(identity(row['input_signatures']))
    source_exposures=len(exposed);torch.set_num_threads(8);server=preprocessing_server(a.source,a.checkpoint,a.qwen_metadata)
    def request(frame,instruction):
        path=a.output/frame['raw_frame']
        with gzip.open(path,'rb') as f:raw=torch.load(f,map_location='cpu',weights_only=False)
        require(content_hash(raw)==path.name.split('.')[0],'Retained frame content changed')
        obs=raw['observation'];value=server._prepare_model_input({'task':instruction,'observation.state':obs['joint_action']['vector'],
            'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
            'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
            'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb']})
        return {k:value[k] for k in KEYS}
    from scripts.collect_lingbot_robotwin_campaign_replay_v3 import retain
    result=copy.deepcopy(original);rows=[];scans=[]
    for e in original['episodes']:
        old_rows=[row for row in original['observations'] if row['source_episode_file']==e['source_episode_file']]
        require(len(old_rows)==e['selected_observations'],'Original row scope differs')
        if e['split']=='diagnostic':rows.extend(copy.deepcopy(old_rows));continue
        episode=episodes[e['source_episode_file']];cache={}
        def prepare(index):
            if index not in cache:cache[index]=request(episode['trajectory'][index],episode['instruction'])
            return cache[index]
        chosen,trace=select(len(episode['steps']),exposed,prepare)
        require(len(old_rows)==8,'Preserve prescribed eight heldout inputs per episode')
        for ordinal,index in enumerate(chosen):
            value=prepare(index);row=copy.deepcopy(old_rows[ordinal]);row.update(frame_index=index,
                raw_frame=episode['trajectory'][index]['raw_frame'],request_file=retain(a.output,'selected_requests',value),
                input_signatures={k:signature(value[k]) for k in KEYS})
            rows.append(row)
        scans.append(dict(source_episode_file=e['source_episode_file'],anchors=trace))
        print('RESELECTED',e['task'],e['seed'],chosen,flush=True)
    require([row['replay_index'] for row in rows]==list(range(len(rows))),'Original canonical replay order changed')
    require([x for x in rows if x['split']=='diagnostic']==[x for x in original['observations'] if x['split']=='diagnostic'],'Diagnostic inputs changed')
    result['observations']=rows
    result['derived_selection']=dict(version=VERSION,source_audit_path=str(a.audit.resolve()),source_audit_sha256=sha(a.audit),
        source_manifest_snapshot=str(snapshot.resolve()),source_manifest_sha256=sha(snapshot),source_directory=str(a.directory.resolve()),
        source_native_and_diagnostic_exposed_identities=source_exposures,episodes=scans,producer_sha256=sha(__file__),
        neural_inference_executed=False,final_historical_and_paired_freshness_pending=True)
    (a.output/'collection_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'reselect_lingbot_source_holdout.py').write_bytes(Path(__file__).read_bytes())
    print('Derived',len(rows),'inputs; independent reconstruction and final historical/paired freshness still required',flush=True)


if __name__=='__main__':main()

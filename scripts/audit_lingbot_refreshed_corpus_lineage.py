"""Compose the complete native-corpus proof with audited replacement inputs.

Native episodes are inherited unchanged. Only the 13 independently rebuilt
selected requests change. Final exposure sealing remains a separate gate.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Output exists');c=json.loads(a.candidate.read_text());root=Path(c['fixture_root'])
    loaded={}
    for field in ('original_corpus_audit','original_source_snapshot','refreshed_source_snapshot','replacement_producer','replacement_independent_audit'):
        path=Path(c[field]);require(sha(path)==c[field+'_sha256'],'Lineage bytes differ: '+field);loaded[field]=json.loads(path.read_text())
    old=loaded['original_corpus_audit'];before=loaded['original_source_snapshot'];after=loaded['refreshed_source_snapshot']
    require(old['status']=='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt' and old['all_prescribed_native_episodes_rebuilt'] and old['completed_native_episodes']==200,'Incomplete parent native proof')
    require(old['manifest_snapshot_sha256']==c['original_source_snapshot_sha256'],'Foreign parent snapshot')
    producer=loaded['replacement_producer'];proof=loaded['replacement_independent_audit']
    require(proof['status']=='thirteen_replacement_requests_independently_rebuilt_from_retained_frames' and proof['producer_sha256']==c['replacement_producer_sha256'],'Missing replacement reconstruction')
    changes={x['replay_index']:x for x in producer['replacements']};verified={x['replay_index']:x for x in proof['verified']}
    require(len(changes)==len(verified)==13 and set(changes)==set(verified)==set(c['changed_replay_indices']),'Wrong replacement roster')
    expected=copy.deepcopy(before);raw_files=copy.deepcopy(old['raw_files_sha256'])
    keys=('images','img_masks','lang_tokens','lang_masks','state','image_grid_thw')
    for index,value in changes.items():
        row=expected['observations'][index];require(row['split']=='heldout' and row['task']==value['task'] and row['seed']==value['seed'] and row['frame_index']==value['original_frame_index'],'Wrong replaced source row')
        identity=hashlib.sha256(json.dumps({k:value['input_signatures'][k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(identity==verified[index]['prepared_identity'],'Replacement tensor signatures differ')
        request=(root/value['request_file']).resolve();frame=(root/value['raw_frame']).resolve()
        require(request.is_relative_to(root.resolve()) and frame.is_relative_to(root.resolve()),'Source escapes corpus')
        require(sha(request)==verified[index]['request_sha256'],'Replacement request changed')
        require(sha(frame)==verified[index]['raw_frame_sha256']==raw_files[str(frame)],'Replacement source frame differs from full native audit')
        raw_files[str(request)]=verified[index]['request_sha256']
        for key in ('raw_frame','request_file','input_signatures'):row[key]=value[key]
        row['frame_index']=value['replacement_frame_index']
    require(expected==after,'Refreshed snapshot includes unaudited changes')
    require(len(after['episodes'])==200 and len(after['observations'])==1000,'Corpus roster changed')
    result=copy.deepcopy(old)
    result.update(manifest_snapshot=str(Path(c['refreshed_source_snapshot']).resolve()),manifest_snapshot_sha256=c['refreshed_source_snapshot_sha256'],raw_files_sha256=raw_files,
                  auditor_sha256=sha(Path(__file__)),proof_method='unchanged_complete_native_episode_proof_plus_independently_rebuilt_selected_request_replacements',
                  parent_corpus_audit=c['original_corpus_audit'],parent_corpus_audit_sha256=c['original_corpus_audit_sha256'],
                  replacement_independent_audit=c['replacement_independent_audit'],replacement_independent_audit_sha256=c['replacement_independent_audit_sha256'],
                  candidate_sha256=sha(a.candidate),replaced_selected_requests=13,final_heldout_exposure_seal=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'native_episode_proof_inherited_unchanged':200,'selected_requests':1000,'independently_rebuilt_replacements':13,'final_seal':False}))


if __name__=='__main__':main()

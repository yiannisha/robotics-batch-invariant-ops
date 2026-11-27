"""Version 2: include native collection itself in prepared-input exposure."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from scripts.audit_lingbot_campaign_corpus import KEYS,content_hash,signature


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(signatures):
    return hashlib.sha256(json.dumps({k:signatures[k] for k in KEYS},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepared(path,expected_hash=None):
    if expected_hash is not None:require(sha(path)==expected_hash,'Retained request bytes changed')
    with gzip.open(path,'rb') as stream:value=torch.load(stream,map_location='cpu',weights_only=False)
    require(content_hash(value)==path.name.split('.')[0],'Request content address differs')
    signatures={k:signature(value[k]) for k in KEYS}
    return identity(signatures),signatures


def native_collection_exposure(manifest):
    root=Path(manifest['fixture_root']);audit_path=Path(manifest['corpus_audit_path'])
    require(sha(audit_path)==manifest['corpus_audit_sha256'],'Native collection audit changed')
    audit=json.loads(audit_path.read_text())
    require(audit['status']=='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt'
            and audit['completed_native_episodes']==200 and audit['all_prescribed_native_episodes_rebuilt'] is True,
            'Full independently reconstructed source collection required')
    bank=(root/'requests').resolve()
    files={str(Path(path).resolve()):h for path,h in audit['raw_files_sha256'].items()
           if Path(path).parent.resolve()==bank}
    require(files and set(files)=={str(p.resolve()) for p in bank.glob('*.pt.gz')},
            'Native collection request bank differs from independent audit')
    exposed=set()
    for path,h in files.items():
        key,_=prepared(Path(path),h);exposed.add(key)
    return exposed,files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','prior-request-index','prior-history-index','paired-directory','paired-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve prior seals')
    paired=json.loads(a.paired_audit.read_text())
    require(paired.get('status')=='lingbot_paired_raw_inputs_actions_and_observable_trajectories_independently_verified'
            and paired.get('cohort_rollouts_complete') is True and paired.get('completed_paired_starts')==2500
            and paired.get('episodes')==20000 and paired.get('all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks') is True,
            'A complete independently audited 2500-start paired cohort is required')
    lock_path=a.paired_directory/'candidate_lock.json';require(sha(lock_path)==paired['candidate_lock_sha256'],'Paired candidate changed')
    lock=json.loads(lock_path.read_text());manifest=json.loads(a.manifest.read_text())
    require(manifest['model_id']=='lingbot_robotwin_exact' and len(manifest['observations'])==1000,'Wrong replay corpus')
    torch.set_num_threads(2)
    # Validate the exact adapter metadata contract, without constructing a model.
    from scripts.model_invariance.adapters.lingbot_robotwin_exact import LingBotRobotwinExactAdapter
    adapter=LingBotRobotwinExactAdapter();adapter.replay_manifest_path=a.manifest;adapter._load_replay_metadata()
    source_manifest=Path(lock['replay_manifest_path']);require(sha(source_manifest)==lock['replay_manifest_sha256'],'Paired companion manifest changed')
    companion=json.loads(source_manifest.read_text());exposed,native_files=native_collection_exposure(manifest);bound=dict(native_files)
    def bind(path):bound[str(path.resolve())]=sha(path)
    prior=json.loads(a.prior_request_index.read_text());history=json.loads(a.prior_history_index.read_text())
    require(prior['status']=='retained_native_request_tensor_bytes_and_prior_pilot_signatures_screened' and prior['retained_request_files']==1172,'Incomplete retained native-request index')
    for path,h in history['sources_sha256'].items():require(sha(Path(path))==h,'Prior exposure-history source changed');bound[path]=h
    require(set(history['source_counts'])=={'lingbot-robotwin-campaign-001','lingbot-robotwin-campaign-002','lingbot-closed-loop-pilot-001','lingbot-official-seed-pilot-002','lingbot-real-inputs-001','lingbot-real-inputs-002'},'Missing historical exposure category')
    for row in prior['request_files']:
        key,_=prepared(Path(row['path']),row['sha256']);require(key==row['prepared_input_identity'],'Native request identity differs');exposed.add(key)
    require(len(prior['request_files'])==1172,'Native request index roster differs')
    for path in history['sources_sha256']:
        path=Path(path)
        if path.name=='episodes.jsonl':
            for line in path.read_text().splitlines():
                for call in json.loads(line)['calls']:exposed.add(identity(call['model_inputs']))
        elif path.name=='requests.pt':
            for value in torch.load(path,map_location='cpu',weights_only=False):exposed.add(identity({k:signature(value[k]) for k in KEYS}))
    root=Path(manifest['fixture_root']);held=[]
    for row in manifest['observations']:
        key,s=prepared(root/row['request_file'],row['request_sha256']);require(s==row['input_signatures'],'Replay request signatures differ')
        if row['split']=='heldout':held.append(key)
        else:require(row['split']=='diagnostic','Unknown split');exposed.add(key)
    for row in companion['observations']:
        if row['split']=='diagnostic':
            key,s=prepared(Path(companion['fixture_root'])/row['request_file'],row['request_sha256'])
            require(s==row['input_signatures'],'Paired companion signatures differ');exposed.add(key)
    requests=sorted((a.paired_directory/'requests').glob('*.pt.gz'))
    require(requests,'Paired request bank missing')
    audited_requests={path for path in paired['raw_files_sha256'] if Path(path).parent.resolve()==(a.paired_directory/'requests').resolve()}
    require({str(x.resolve()) for x in requests}==audited_requests,'Paired request bank differs from independent audit')
    for index,path in enumerate(requests):
        key,_=prepared(path,paired['raw_files_sha256'][str(path.resolve())]);exposed.add(key)
        if index%1000==0:print('checked paired requests',index,flush=True)
    require(len(held)==len(set(held))==800,'Heldout roster must contain 800 distinct inputs')
    collisions=set(held)&exposed;require(not collisions,'Heldout inputs were exposed before freeze')
    for path in (a.manifest,a.prior_request_index,a.prior_history_index,a.paired_audit,lock_path,source_manifest,Path(manifest['corpus_audit_path'])):bind(path)
    require(sha(lock_path)==paired['candidate_lock_sha256'],'Paired candidate changed during audit')
    report=dict(status='complete_observation_disjoint_heldout_independently_verified',candidate_manifest_sha256=sha(a.manifest),
                identity_kind='prepared_neural_inputs_before_first_model_operation',prepared_input_disjointness_verified=True,
                model_id='lingbot_robotwin_exact',heldout_observations=800,heldout_unique_prepared_inputs=800,
                prior_exposed_unique_prepared_inputs=len(exposed),collision_count=0,complete_paired_starts=2500,
                native_collection_request_files=len(native_files),native_collection_inputs_screened=True,
                paired_request_files=len(requests),source_evidence_sha256=bound,auditor_sha256=sha(Path(__file__)),
                neural_inference_executed=False,noise_excluded_from_input_identity=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_evidence_sha256'},indent=2))


if __name__=='__main__':main()

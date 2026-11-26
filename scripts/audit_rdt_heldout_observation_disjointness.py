"""Reject RDT heldout observations exposed by prior corpora or paired inference.

Noise and request identifiers are deliberately excluded from observation
identity. This final gate requires the paired exposure cohort to be complete.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def observation_identity(request):
    return digest({key:request[key] for key in ('task_name','history_valid','image_history','proprio')})


def corpus_inputs(path,identity_function=observation_identity):
    manifest=json.loads(path.read_text());root=Path(manifest['fixture_root'])
    require(manifest['model_id']=='rdt_maniskill_exact','Wrong source model')
    for row in manifest['observations']:
        file=root/row['request_file'];require(sha(file)==row['request_sha256'],'Source request file changed')
        yield row,identity_function(torch.load(file,map_location='cpu',weights_only=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','paired-directory','paired-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-manifest',type=Path,action='append',required=True)
    p.add_argument('--progress-only',action='store_true',help='Measure current exposure snapshot without certifying heldout acceptance')
    a=p.parse_args();torch.set_num_threads(4)
    candidate=json.loads(a.manifest.read_text());audit=Path(candidate['corpus_audit_path'])
    require(sha(audit)==candidate['corpus_audit_sha256'] and json.loads(audit.read_text())['status']=='complete_corpus_independently_verified','Fresh corpus audit missing/changed')
    paired=json.loads(a.paired_audit.read_text())
    require(paired['all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks'] and (a.progress_only or paired['cohort_rollouts_complete']),'Final exposure seal requires completed, audited paired cohort')
    require(sha(a.paired_directory/'candidate_lock.json')==paired['candidate_lock_sha256'],'Wrong paired exposure cohort')
    lock=json.loads((a.paired_directory/'candidate_lock.json').read_text())
    from transformers import AutoImageProcessor
    from scripts.rdt_prepared_observation_identity import PreparedIdentity
    processor_path=next(Path(path).parent for path in lock['asset_sha256'] if Path(path).name=='preprocessor_config.json')
    relevant={path:expected for path,expected in lock['asset_sha256'].items() if Path(path).parent==processor_path or Path(path).name.startswith('text_embed_')}
    # Only load preprocessing configuration and pinned language embeddings;
    # this audit never constructs a neural model.
    for path,expected in relevant.items():
        if Path(path).suffix in ('.json','.pt'):require(sha(Path(path))==expected,'Prepared-input asset changed')
    processor=AutoImageProcessor.from_pretrained(processor_path)
    language={task:torch.load(next(Path(path) for path in relevant if Path(path).name==f'text_embed_{task}.pt'),map_location='cpu',weights_only=True) for task in lock['global_tasks']}
    prepared=PreparedIdentity(processor,language)
    candidate_ids={};exposed=set();source_files=[]
    for row,identity in corpus_inputs(a.manifest,prepared):
        if row['split']=='heldout':candidate_ids.setdefault(identity,[]).extend(row['request_ids'])
        else:exposed.add(identity)
    require(sum(len(v) for v in candidate_ids.values())==2400,'Expected800 heldout observations with three noises each')
    for path in a.prior_manifest:
        require(sha(path)!=sha(a.manifest),'Candidate cannot be its own prior corpus')
        source_files.append(dict(path=str(path.resolve()),sha256=sha(path)))
        for _,identity in corpus_inputs(path,prepared):exposed.add(identity)
    request_files=[]
    for path in sorted((a.paired_directory/'requests').glob('*.gz')):
        with gzip.open(path,'rb') as stream:request=torch.load(stream,map_location='cpu',weights_only=False)
        require(digest(request)==path.name.split('.')[0],'Paired raw request content address changed')
        exposed.add(prepared(request));request_files.append(dict(path=str(path.resolve()),sha256=sha(path)))
    require(request_files,'Paired request exposure universe is empty')
    collisions=sorted(set(candidate_ids)&exposed)
    result=dict(status='failed_observation_disjointness' if collisions else 'prepared_input_exposure_progress_only' if a.progress_only else 'complete_observation_disjoint_heldout_independently_verified',
        model='rdt',candidate_manifest_sha256=sha(a.manifest),candidate_corpus_audit_sha256=sha(audit),
        paired_exposure_audit_sha256=sha(a.paired_audit),paired_candidate_lock_sha256=paired['candidate_lock_sha256'],
        prior_manifests=source_files,paired_request_files=request_files,exposed_unique_observations=len(exposed),
        heldout_unique_observations=len(candidate_ids),heldout_request_noise_pairs=2400,
        identity_fields=['vision_pixels_bfloat16','language_bfloat16','language_mask_bool','state_bfloat16','action_mask_bfloat16','control_frequency_int64'],noise_and_request_ids_excluded=True,
        identity_kind='prepared_neural_inputs_before_first_model_operation',prepared_input_disjointness_verified=not collisions and not a.progress_only,
        preprocessing_assets=relevant,identity_source_sha256=sha(Path(__file__).with_name('rdt_prepared_observation_identity.py')),
        collisions=[dict(observation_sha256=h,request_ids=candidate_ids[h]) for h in collisions],
        auditor_sha256=sha(Path(__file__)),full_model_campaign_complete=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'heldout unique observations={len(candidate_ids)}; exposed={len(exposed)}; overlaps={len(collisions)}',flush=True)
    require(not collisions,'Heldout observation exposure detected; retain evidence and do not evaluate as unseen')


if __name__=='__main__':main()

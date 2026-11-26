"""Check fresh PI05 consumed inputs against all retained paired exposure.

Includes rejected/default-renderer sibling campaigns. Uncommitted observations
with no saved episode attribution are conservatively checked with all ten task
instructions. No selection depends on success, numerical error, or collisions.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

from scripts.pi05_prepared_observation_identity import PreparedIdentity
from scripts.qualify_simpler_fixed_action_replay import digest


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','paired-directory','paired-audit','checkpoint','tokenizer','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-manifest',type=Path,action='append',required=True)
    p.add_argument('--progress-only',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    m=json.loads(a.manifest.read_text());audit=Path(m['corpus_audit_path'])
    require(sha(audit)==m['corpus_audit_sha256'] and json.loads(audit.read_text())['status']=='complete_corpus_independently_verified','Candidate corpus not independently audited')
    paired=json.loads(a.paired_audit.read_text());lock_path=a.paired_directory/'candidate_lock.json'
    require(sha(lock_path)==paired['candidate_lock_sha256'] and paired['all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks'],'Wrong paired evidence')
    require(a.progress_only or paired['cohort_rollouts_complete'],'Final seal requires complete independently audited paired cohort')
    lock=json.loads(lock_path.read_text())
    require(a.checkpoint.resolve()==next(Path(path).parent.resolve() for path in lock['checkpoint_asset_sha256'] if Path(path).name=='model.safetensors'),'Checkpoint preprocessing root differs from paired runtime')
    require(a.tokenizer.resolve()==next(Path(path).parent.resolve() for path in lock['tokenizer_sha256'] if Path(path).name=='tokenizer.model'),'Tokenizer root differs from paired runtime')
    for path,expected in lock['source_sha256'].items():require(sha(Path(path))==expected,'Pinned source changed before input reconstruction')
    for path,expected in lock['tokenizer_sha256'].items():require(sha(Path(path))==expected,'Tokenizer changed')
    for path,expected in lock['checkpoint_asset_sha256'].items():
        if Path(path).suffix=='.json':require(sha(Path(path))==expected,'Checkpoint preprocessing metadata changed')
    prepared=PreparedIdentity(a.checkpoint,a.tokenizer);candidate={};exposed=set();sources=[]
    def corpus(path):
        manifest=json.loads(path.read_text());root=Path(manifest['fixture_root'])
        require(manifest['model_id']=='pi05_libero','Foreign PI05 corpus')
        for row in manifest['observations']:
            file=root/row['input_file'];require(sha(file)==row['input_file_sha256'],'Prepared source input changed')
            yield row,prepared(torch.load(file,map_location='cpu',weights_only=True))
    instructions={}
    for row,identity in corpus(a.manifest):
        task=row['task'];require(task not in instructions or instructions[task]==row['instruction'],'Task instruction changed')
        instructions[task]=row['instruction']
        if row['split']=='heldout':candidate.setdefault(identity,[]).extend(row['request_ids'])
        else:exposed.add(identity)
    require(set(instructions)==set(range(10)) and sum(map(len,candidate.values()))==2400,'Wrong candidate cohort')
    for path in a.prior_manifest:
        require(sha(path)!=sha(a.manifest),'Candidate cannot be its own prior corpus')
        sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
        for _,identity in corpus(path):exposed.add(identity)
    directories=sorted({a.paired_directory.resolve(),*(d.resolve() for d in a.paired_directory.parent.glob('pi05*') if (d/'candidate_lock.json').is_file())})
    retained=[];unattributed=0
    for directory in directories:
        names={}
        for episode_path in sorted((directory/'blocks').glob('task-*/start-*/*.json.gz')):
            with gzip.open(episode_path,'rt') as stream:episode=json.load(stream)
            if 'calls' not in episode:continue
            instruction=instructions[episode['task']]
            for call in episode['calls']:
                path=(directory/call['raw_observation_file']).resolve()
                require(path.parent==(directory/'observations').resolve(),'Raw observation escapes campaign')
                names.setdefault(path,set()).add(instruction)
        files=sorted((directory/'observations').glob('*.pt'))
        require(files,'Empty prior paired exposure directory')
        for path in files:
            obs=torch.load(path,map_location='cpu',weights_only=False)
            require(digest(obs)==path.stem,'Raw paired observation address differs')
            choices=names.get(path.resolve())
            if not choices:choices=set(instructions.values());unattributed+=1
            for instruction in choices:exposed.add(prepared(prepared.prepared(obs,instruction)))
            retained.append(dict(path=str(path.resolve()),sha256=sha(path),instructions_checked=sorted(choices)))
        if not a.progress_only:require(files==sorted((directory/'observations').glob('*.pt')),'Exposure universe grew during final seal')
    collisions=sorted(set(candidate)&exposed)
    result=dict(status='failed_observation_disjointness' if collisions else 'prepared_input_exposure_progress_only' if a.progress_only else 'complete_observation_disjoint_heldout_independently_verified',
        model='pi05',candidate_manifest_sha256=sha(a.manifest),candidate_corpus_audit_sha256=sha(audit),paired_exposure_audit_sha256=sha(a.paired_audit),
        paired_directories=[str(d) for d in directories],prior_manifests=sources,paired_observation_files=retained,
        unattributed_observations_checked_with_all_task_instructions=unattributed,identity_kind='prepared_neural_inputs_before_first_model_operation',
        prepared_input_disjointness_verified=not collisions and not a.progress_only,identity_fields=['official_GPU_resized_normalized_float32_images','image_masks','language_tokens','language_mask'],
        unused_state_tensor_excluded=True,noise_and_request_ids_excluded=True,heldout_unique_observations=len(candidate),heldout_request_noise_pairs=2400,exposed_unique_observations=len(exposed),
        collisions=[dict(observation_sha256=h,request_ids=candidate[h]) for h in collisions],auditor_sha256=sha(Path(__file__)),identity_source_sha256=sha(Path(__file__).with_name('pi05_prepared_observation_identity.py')),full_model_campaign_complete=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Candidate unique={len(candidate)}, exposed={len(exposed)}, collisions={len(collisions)}',flush=True)
    require(not collisions,'Prepared heldout observation exposure detected')


if __name__=='__main__':main()

"""Seal independently reconstructed full LingBot corpus as numerical inputs.

This creates explicit replay noises and companions. It does not certify unseen
heldout observations, numerical invariance, fidelity, or simulator outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_model_replay_invariance import split_local_companion_rows


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit=json.loads(a.audit.read_text())
    if audit['status']!='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt' or not audit['all_prescribed_native_episodes_rebuilt'] or audit['completed_native_episodes']!=200:
        raise ValueError('All 200 native episodes must be independently reconstructed before sealing numerical inputs')
    snapshot=Path(audit['manifest_snapshot'])
    if sha(snapshot)!=audit['manifest_snapshot_sha256']:raise ValueError('Audited corpus snapshot changed')
    m=json.loads(snapshot.read_text())
    if m['candidate_lock_sha256']!=audit['candidate_lock_sha256'] or sha(a.directory/'candidate_lock.json')!=audit['candidate_lock_sha256']:raise ValueError('Native corpus identity changed')
    for file,expected in audit['raw_files_sha256'].items():
        if sha(Path(file))!=expected:raise ValueError('Audited source input changed')
    rows=m['observations'];count=len(rows)
    if count!=audit['selected_inputs_rebuilt'] or count<64:raise ValueError('Invalid audited input coverage')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    noise=np.empty((count,3,50,55),dtype=np.float32)
    for index,row in enumerate(rows):
        if row['replay_index']!=index:raise ValueError('Noncanonical replay order')
        row['request_sha256']=sha(a.directory/row['request_file'])
        row['request_ids']=[];row['noise']=[]
        for j in range(3):
            key=f'lingbot-robotwin/{row["task"]}/{row["seed"]}/frame-{row["frame_index"]}/noise-{j}'
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
            value=torch.randn(50,55,generator=torch.Generator().manual_seed(seed)).numpy()
            noise[index,j]=value;row['request_ids'].append(key)
            row['noise'].append(dict(generator_seed=seed,sha256=hashlib.sha256(value.tobytes()).hexdigest()))
        row['companion_replay_indices']=[v['replay_index'] for v in split_local_companion_rows(rows,row,index,31)]
    if len({key for row in rows for key in row['request_ids']})!=3*count:raise ValueError('Duplicate request identity')
    noise_path=a.output/'noise.npy';np.save(noise_path,noise)
    report=dict(model_id='lingbot_robotwin_exact',observations=rows,fixture_root=str(a.directory.resolve()),noise_file=str(noise_path.resolve()),noise_shape=list(noise.shape),noise_sha256=sha(noise_path),
        corpus_audit_path=str(a.audit.resolve()),corpus_audit_sha256=sha(a.audit),source_manifest_snapshot=str(snapshot),source_manifest_sha256=sha(snapshot),
        builder_sha256=sha(Path(__file__)),numerical_freeze=None,heldout_prepared_input_disjointness_pending=True,
        native_contract='FP32 official dtype sequence; deterministic token-choice MoE shared by native/invariant; released atomic MoE separate',
        companion_schedule=dict(version='split_local_unrelated_observations_v1',max_companions=31))
    (a.output/'replay_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f'Sealed {count} observations / {count*3} explicit input-noise pairs; numerical and heldout acceptance pending')


if __name__=='__main__':main()

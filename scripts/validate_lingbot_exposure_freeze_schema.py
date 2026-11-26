"""Validate a completed LingBot exposure proof for the common freeze schema."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    d=json.loads(a.input.read_text())
    from scripts.audit_lingbot_heldout_observation_disjointness import identity,KEYS
    if not (d.get('status')=='complete_observation_disjoint_heldout_independently_verified'
            and d.get('model_id')=='lingbot_robotwin_exact' and d.get('complete_paired_starts')==2500
            and d.get('prepared_input_disjointness_verified') is True and d.get('collision_count')==0
            and d.get('heldout_observations')==d.get('heldout_unique_prepared_inputs')==800
            and d.get('noise_excluded_from_input_identity') is True):
        raise ValueError('Completed LingBot exposure proof required')
    source=Path(__file__).with_name('audit_lingbot_heldout_observation_disjointness.py')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if d['auditor_sha256']!=sha(source):raise ValueError('Exposure auditor changed')
    # Exercise the actual identity function: metadata/noise must have no effect;
    # every prepared neural input must still affect the fingerprint.
    values={k:{'sha256':k,'shape':[1],'dtype':'test'} for k in KEYS}
    expected=identity(values)
    if identity(dict(values,noise='different',request_id='different'))!=expected:
        raise ValueError('Noise or request identifiers affect exposure identity')
    for key in KEYS:
        changed=copy.deepcopy(values);changed[key]['sha256']='changed'
        if identity(changed)==expected:raise ValueError('Neural input omitted: '+key)
    for path,h in d['source_evidence_sha256'].items():
        if sha(Path(path))!=h:raise ValueError('Exposure evidence changed: '+path)
    d.update(noise_and_request_ids_excluded=True,collisions=[],underlying_exposure_proof=str(a.input.resolve()),
             underlying_exposure_proof_sha256=sha(a.input),freeze_schema_validator_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()

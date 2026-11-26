"""Require a freeze-bound observation-disjointness audit before heldout replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(manifest,freeze,disjointness,evidence_paths):
    identity=sha(manifest)
    for path in evidence_paths:
        evidence=json.loads(path.read_text())
        if evidence.get('status')=='old_numerical_heldout_observation_partition_exposed_before_freeze' and evidence['candidate_manifest_sha256']==identity:
            raise ValueError('Selected numerical heldout observation partition was exposed before freeze; fresh observations are required')
    record=json.loads(disjointness.read_text())
    if record.get('status')!='complete_observation_disjoint_heldout_independently_verified' or record.get('candidate_manifest_sha256')!=identity:
        raise ValueError('Selected manifest lacks a completed observation-disjointness audit')
    if record.get('identity_kind')!='prepared_neural_inputs_before_first_model_operation' or record.get('prepared_input_disjointness_verified') is not True:
        raise ValueError('Heldout acceptance requires disjoint prepared neural inputs; raw observation differences alone are insufficient')
    bound=json.loads(freeze.read_text()).get('content_addressed_freeze',{})
    for path in (manifest,disjointness):
        expected=bound.get(str(path.resolve()),{})
        if expected.get('sha256')!=sha(path) or expected.get('bytes')!=path.stat().st_size:
            raise ValueError('Numerical freeze must bind both the replay manifest and its observation-disjointness audit')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','freeze','disjointness-audit'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exposure-evidence',type=Path,action='append',default=[])
    p.add_argument('--split',choices=['heldout'],default='heldout');a,other=p.parse_known_args()
    known=list((Path(__file__).resolve().parents[1]/'results/four_model').glob('heldout_observation_exposure*.json'))
    if not known:raise FileNotFoundError('Known observation-exposure evidence registry is missing')
    evidence=[*known,*a.exposure_evidence]
    validate(a.manifest,a.freeze,a.disjointness_audit,evidence)
    command=[sys.executable,'-m','scripts.run_model_replay_invariance','--manifest',str(a.manifest),
        '--freeze',str(a.freeze),'--split','heldout',*other]
    raise SystemExit(subprocess.call(command,cwd=Path(__file__).resolve().parents[1]))


if __name__=='__main__':main()

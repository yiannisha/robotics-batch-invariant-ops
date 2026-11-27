"""Audit byte-guarded recovery without claiming historical trajectory equivalence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from scripts.audit_lingbot_campaign_corpus import KEYS,content_hash,signature

AUTHORITIES={
 '24d67091b45b1987e0410993a3daac57404f32c712bc4fff8581b2a8037ed267':1077,
 'b3134e2aece77a3ad009498186fa5f31552c46d2a79e7a3a992cf3e89434eaa5':82,
}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_record(root,row,authorities):
    path=Path(row['path']).resolve();require(path.is_relative_to(root.resolve()),'Recovery input escapes retained root')
    actual=sha(path);require(actual==row['sha256'],'Recovery input bytes changed')
    matches=[dict(original_path=file,original_sha256=h,authority_sha256=identity)
             for identity,report in authorities.items() for file,h in report['raw_files_sha256'].items()
             if Path(file).parent.name=='requests' and Path(file).name==path.name and h==actual]
    require(matches and row['historical_authorities']==matches,'Recovery forward lacked original byte authority')
    require(row['inference_attempt_authorized'] is True and row['prepared_input_already_archived'] is True,'Forward not authorized before use')
    with gzip.open(path,'rb') as stream:value=torch.load(stream,map_location='cpu',weights_only=False)
    require(content_hash(value)==path.name.split('.')[0] and set(value)==set(KEYS)|{'noise'},'Recovery request content differs')
    identity=hashlib.sha256(json.dumps({k:signature(value[k]) for k in KEYS},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(path=str(path),sha256=actual,prepared_input_identity=identity,original_authorities=matches)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.directory;require(not a.output.exists(),'Preserve earlier audit attempts')
    lock_path=root/'candidate_lock.json';lock_identity=sha(lock_path);lock=json.loads(lock_path.read_text())
    require(lock['scope']=='guarded_historical_tf32_on_regeneration_only_not_primary_evidence'
            and lock['forward_guard']=='byte_exact_archived_request_before_every_inference' and lock['tf32_matmul'] is True,
            'Wrong recovery execution protocol')
    authorities={}
    for file,h in lock['historical_authorities'].items():
        require(h in AUTHORITIES and sha(file)==h,'Historical byte authority changed');report=json.loads(Path(file).read_text())
        require(report['auditor_sha256']=='809a9c4087c389e3e16c5f27b5ef88d77573065e70a8f71dd877b47308db5da6','Wrong historical auditor')
        require(sum(Path(path).parent.name=='requests' for path in report['raw_files_sha256'])==AUTHORITIES[h],'Historical authority coverage differs')
        authorities[h]=report
    for file,h in lock['source_sha256'].items():require(sha(file)==h and sha(root/'source_snapshots'/h)==h,'Recovery execution source changed')
    ledger=root/'authorized_forward_inputs.jsonl';payload=ledger.read_bytes() if ledger.exists() else b''
    require(not payload or payload.endswith(b'\n'),'Recovery ledger has an incomplete last record')
    rows=[json.loads(line) for line in payload.splitlines()];verified=[]
    for row in rows:verified.append(verify_record(root,row,authorities))
    bank={str(path.resolve()):sha(path) for path in (root/'requests').glob('*.pt.gz')}
    expected={str((root/'requests'/Path(row['path']).name).resolve()):row['sha256'] for row in rows}
    require(bank==expected,'Retained recovery request bank differs from guarded forward ledger')
    require(sha(lock_path)==lock_identity,'Recovery candidate changed during audit')
    if ledger.exists():require(sha(ledger)==hashlib.sha256(payload).hexdigest(),'Forward ledger grew during terminal audit')
    report=dict(status='guarded_historical_request_bytes_independently_verified',candidate_lock_sha256=lock_identity,
        historical_authorities=lock['historical_authorities'],forward_attempt_records=len(rows),recovered_unique_containers=len(bank),
        all_forward_inputs_have_byte_exact_prior_authority=True,new_prepared_input_identities_introduced_by_forwarding=0,
        request_files_sha256=bank,verified_forward_inputs=verified,forward_ledger_sha256=hashlib.sha256(payload).hexdigest(),
        auditor_sha256=sha(__file__),primary_corrected_inference_evaluated=False,historical_trajectory_equivalence_claimed=False,
        complete_historical_exposure_coverage=False,primary_paired_stability_measured=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('request_files_sha256','verified_forward_inputs')},indent=2))


if __name__=='__main__':main()

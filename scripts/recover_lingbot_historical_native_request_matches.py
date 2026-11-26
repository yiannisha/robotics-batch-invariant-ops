"""Recover old request containers only when their original SHA-256 matches exactly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from scripts import audit_lingbot_campaign_corpus as legacy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-audit',type=Path,required=True)
    p.add_argument('--current-corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    old=json.loads(a.old_audit.read_text())
    if old['auditor_sha256']!=sha(legacy.__file__):raise ValueError('Historical auditor source differs')
    entries={Path(path).name:(path,identity) for path,identity in old['raw_files_sha256'].items() if Path(path).parent.name=='requests'}
    if len(entries)!=1077:raise ValueError('Expected complete old 1077-request corpus authority')
    candidates=sorted((a.current_corpus/'requests').glob('*.pt.gz'))
    (a.output/'requests').mkdir(parents=True)
    recovered=[]
    for source in candidates:
        if source.name not in entries:continue
        original,expected=entries[source.name]
        if sha(source)!=expected:continue
        target=a.output/'requests'/source.name
        os.link(source,target)
        if sha(target)!=expected:raise ValueError('Recovered container bytes changed')
        recovered.append(dict(original_path=original,original_sha256=expected,
                              source_path=str(source.resolve()),recovered_path=str(target.resolve())))
    report=dict(status='partial_historical_native_request_containers_recovered_by_original_sha256',
        authority_path=str(a.old_audit.resolve()),authority_sha256=sha(a.old_audit),
        historical_auditor_source_sha256=sha(legacy.__file__),producer_sha256=sha(Path(__file__)),
        historical_request_count=1077,current_request_files_examined=len(candidates),
        recovered_request_count=len(recovered),request_files=recovered,
        original_container_bytes_exact=True,neural_inference_performed=False,
        complete_historical_exposure_coverage=False,paired_stability_measured=False)
    (a.output/'recovery_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='request_files'},indent=2))


if __name__=='__main__':main()

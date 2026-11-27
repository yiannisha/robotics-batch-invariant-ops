"""Combine recovery banks only when original historical container hashes match.

Duplicate paths/roles do not create additional unique recovered containers.
No inference is executed and no trajectory/model qualification is implied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

AUTHORITY='24d67091b45b1987e0410993a3daac57404f32c712bc4fff8581b2a8037ed267'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retain_matches(entries,candidates,output):
    output.mkdir(parents=True,exist_ok=False);recovered={}
    for source in candidates:
        if source.name not in entries:continue
        original,expected=entries[source.name]
        if sha(source)!=expected:continue
        target=output/source.name
        if not target.exists():os.link(source,target)
        if sha(target)!=expected:raise ValueError('Recovered original container changed')
        row=recovered.setdefault(source.name,dict(original_path=original,original_sha256=expected,
            source_path=str(source.resolve()),source_paths=[],recovered_path=str(target.resolve())))
        path=str(source.resolve())
        if path not in row['source_paths']:row['source_paths'].append(path)
    return list(recovered.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-audit',type=Path,required=True);p.add_argument('--current-corpus',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sha(a.old_audit)!=AUTHORITY:raise ValueError('Original full200 historical authority required')
    old=json.loads(a.old_audit.read_text())
    if old['auditor_sha256']!='809a9c4087c389e3e16c5f27b5ef88d77573065e70a8f71dd877b47308db5da6':raise ValueError('Historical reconstruction source differs')
    entries={Path(path).name:(path,h) for path,h in old['raw_files_sha256'].items() if Path(path).parent.name=='requests'}
    if len(entries)!=1077:raise ValueError('Historical request authority is incomplete')
    if a.output.exists():raise FileExistsError(a.output)
    candidates=sorted({path for root in a.current_corpus for path in (root/'requests').glob('*.pt.gz')})
    recovered=retain_matches(entries,candidates,a.output/'requests')
    report=dict(status='partial_historical_native_request_containers_recovered_by_original_sha256',
        authority_path=str(a.old_audit.resolve()),authority_sha256=AUTHORITY,producer_sha256=sha(__file__),
        historical_request_count=1077,source_roots=[str(root.resolve()) for root in a.current_corpus],
        current_request_files_examined=len(candidates),recovered_request_count=len(recovered),request_files=recovered,
        original_container_bytes_exact=True,neural_inference_performed=False,duplicate_source_paths_counted_once=True,
        complete_historical_exposure_coverage=False,paired_stability_measured=False)
    (a.output/'recovery_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='request_files'},indent=2))


if __name__=='__main__':main()

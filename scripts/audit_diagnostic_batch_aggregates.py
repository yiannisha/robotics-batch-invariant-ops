"""Recount output batch aggregates from an independently accepted diagnostic.

This deliberately does not import the reporting implementation. Denoising
aggregate tables are outside this auditor's scope.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','diagnostic-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve prior audits; choose a new output')
    report_sha=sha(a.report);audit_sha=sha(a.diagnostic_audit)
    report=json.loads(a.report.read_text());audit=json.loads(a.diagnostic_audit.read_text())
    require(audit['status']=='complete_four_model_diagnostic_independently_verified','Complete diagnostic acceptance required')
    require(len(report['studies'])==1,'Audit one model at a time')
    study=report['studies'][0];raw=Path(study['source']);raw_sha=sha(raw)
    require(raw_sha==audit['diagnostic_sha256'],'Diagnostic bytes differ from accepted audit')
    require(study['runtime_sha256']==audit['runtime_sha256'],'Runtime differs from accepted audit')
    require(study['included_complete_request_noise_pairs']==audit['request_noise_pairs'] and not study['excluded_partial_requests'],'Reporting cohort incomplete')
    groups=defaultdict(list);total=0
    with gzip.open(raw,'rt') as stream:
        for line in stream:
            row=json.loads(line);total+=1
            if row['transformation'] not in ('singleton','batch_position'):continue
            tensor=row['output']['tensors']['output.normalized'];count=math.prod(tensor['shape'])
            require(tensor['nonfinite_count']==0 and count>0,'Invalid source output')
            groups[str(row['task']),row['implementation'],row['batch_size']].append((count,tensor['rmse']**2*count,
                tensor['max_absolute_error'],tensor['unequal_element_count'],tensor['exact'],tensor['dtype']))
    require(total==audit['arrangement_records'],'Raw arrangement count differs')
    reported={(r['task'],r['implementation'],r['batch_size']):r for r in study['normalized_by_task_batch']}
    require(len(reported)==len(study['normalized_by_task_batch']) and reported.keys()==groups.keys(),'Missing or duplicate aggregate groups')
    pooled=defaultdict(list)
    for key,values in groups.items():
        row=reported[key];elements=sum(v[0] for v in values);squares=math.fsum(v[1] for v in values);mse=squares/elements
        require(row['elements']==elements and row['comparisons']==len(values),'Element/comparison count differs')
        require(row['different_elements']==sum(v[3] for v in values) and row['nonexact_comparisons']==sum(not v[4] for v in values),'Exactness count differs')
        require(row['max_absolute_difference']==max(v[2] for v in values),'Maximum difference differs')
        require(row['dtypes']==sorted({v[5] for v in values}),'Source precision differs')
        require(math.isclose(row['mean_squared_singleton_batch_difference'],mse,rel_tol=1e-12,abs_tol=1e-20),'Mean squared deviation differs')
        require(math.isclose(row['rms_singleton_batch_difference'],math.sqrt(mse),rel_tol=1e-12,abs_tol=1e-20),'RMS differs')
        pooled[key[1],key[2]].extend(values)
    rows=[]
    for (mode,batch),values in sorted(pooled.items()):
        elements=sum(v[0] for v in values)
        rows.append(dict(implementation=mode,batch_size=batch,comparisons=len(values),nonexact_comparisons=sum(not v[4] for v in values),
            rms=math.sqrt(math.fsum(v[1] for v in values)/elements),max_absolute_error=max(v[2] for v in values)))
    require(sha(raw)==raw_sha and sha(a.report)==report_sha and sha(a.diagnostic_audit)==audit_sha,'Inputs changed during recount')
    result=dict(status='independent_output_batch_aggregate_recount_passed',scope='Output aggregates only; denoising aggregates excluded',
        report_sha256=report_sha,raw_sha256=raw_sha,source_diagnostic_audit_sha256=audit_sha,auditor_sha256=sha(Path(__file__)),
        verified_task_mode_batch_groups=len(groups),rows=rows,full_campaign_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified',len(groups),'task/mode/batch output aggregates')


if __name__=='__main__':main()

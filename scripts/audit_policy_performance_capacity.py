"""Audit B64 full-policy capacity separately from latency qualification."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_pi05_actual_operators import check_metric


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qa-runtime','diagnostic-audit','records','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve existing audits')
    runtime_path=Path(str(a.records)+'.runtime.json');driver_path=Path(str(a.records)+'.driver.py')
    paths=[a.manifest,a.qa_runtime,a.diagnostic_audit,a.records,runtime_path,driver_path]
    before={str(p.resolve()):sha(p) for p in paths}
    manifest=json.loads(a.manifest.read_text());qa=json.loads(a.qa_runtime.read_text());proof=json.loads(a.diagnostic_audit.read_text());runtime=json.loads(runtime_path.read_text())
    require(proof['status']=='complete_four_model_diagnostic_independently_verified' and proof['runtime_sha256']==sha(a.qa_runtime) and proof['manifest_sha256']==sha(a.manifest),'Accepted diagnostic source required')
    require(runtime['adapter']==qa['adapter'] and runtime['execution_configuration']==qa['execution_configuration']==proof['adapter_execution_configuration'],'Numerical path differs')
    require(runtime['replay_manifest_sha256']==sha(a.manifest) and runtime['split']=='diagnostic' and runtime['batch_sizes']==[64] and runtime['modes']==['native','batch_invariant'] and runtime['numerical_freeze'] is None,'Capacity runtime differs')
    require(runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False,'Capacity precision differs')
    for path,h in qa['source_sha256'].items():require(runtime['source_sha256'].get(path)==h and sha(Path(path))==h,'Diagnostic numerical source differs')
    selected=sorted((r for r in manifest['observations'] if r['split']=='diagnostic'),key=lambda r:(r['observation_index'],str(r['episode_id']),str(r['task'])))[:64]
    require(len(selected)==64 and len({r['replay_index'] for r in selected})==64,'64distinct diagnostic rows required')
    require(len({r['task'] for r in selected})>1,'Diverse selection has one task only')
    rows=[json.loads(line) for line in a.records.read_text().splitlines()];expected={(m,k,p) for m in ('native','batch_invariant') for k in ('duplicate','diverse') for p in ('first','middle','last')};seen=set()
    for r in rows:
        key=(r['implementation'],r['companion_type'],r['target_position']);require(key in expected and key not in seen,'Unexpected/duplicate capacity arrangement');seen.add(key)
        require(r['adapter']==qa['adapter'] and r['adapter_execution_configuration']==qa['execution_configuration'] and r['replay_manifest_sha256']==sha(a.manifest),'Record source differs')
        require(r['batch_size']==64 and r['target_batch_index']=={'first':0,'middle':32,'last':63}[r['target_position']],'Batch extent/target differs')
        require(r['diagnostic_replay_indices']==[v['replay_index'] for v in selected] and r['explicit_noise_id']==0 and r['request_id']==selected[0]['request_ids'][0],'Selected input/noise differs')
        companions=[selected[0]['request_ids'][0]]*63 if r['companion_type']=='duplicate' else [v['request_ids'][0] for v in selected[1:]]
        require(r['companion_ids']==companions and r['driver_sha256']==sha(driver_path),'Frozen companions/driver differ')
        require(r['isolated_performance_qualified'] is False and r['status'] in ('ran','cuda_oom'),'Unsupported capacity claim')
        if r['status']=='ran':
            require(isinstance(r['peak_incremental_bytes'],int) and r['peak_incremental_bytes']>=0,'Invalid memory measurement')
            check_metric(r['output'])
            if r['implementation']=='batch_invariant':require(r['output']['exact'],'Invariant capacity output differs')
        else:require(isinstance(r['error'],str) and r['error'],'Missing OOM evidence')
    require(seen==expected,'Incomplete B64 capacity measurement matrix')
    for path,h in before.items():require(sha(Path(path))==h,'Audit input changed')
    result=dict(status='B64_full_policy_capacity_independently_verified',bound_files_sha256=before,arrangements=len(rows),
        feasible={mode:all(r['status']=='ran' for r in rows if r['implementation']==mode) for mode in ('native','batch_invariant')},
        invariant_ran_arrangements_exact=True,gpu_uuid=runtime['gpu_uuid'],gpu=runtime['gpu'],
        peak_incremental_bytes=max((r.get('peak_incremental_bytes',0) for r in rows),default=0),
        isolated_performance_qualified=False,auditor_sha256=sha(Path(__file__)),full_campaign_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

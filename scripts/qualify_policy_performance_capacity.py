"""Measure B64 full-policy capacity on explicit diagnostic replay inputs.

Capacity is recorded separately from isolated latency qualification. CUDA OOM
is retained as a result; invariant numerical violations fail after recording.
"""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch
from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch,_run_once
from scripts.run_model_replay_invariance import digest,metrics,sample_at,record_runtime_provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',required=True)
    for name in ('manifest','qa-runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads(a.manifest.read_text());identity=digest(a.manifest);qa=json.loads(a.qa_runtime.read_text())
    if qa['adapter']!=a.adapter or qa['replay_manifest_sha256']!=identity:raise ValueError('Wrong accepted diagnostic inputs')
    for path,h in qa['source_sha256'].items():
        if digest(Path(path))!=h:raise ValueError('Pinned numerical source changed: '+path)
    rows=sorted((r for r in manifest['observations'] if r['split']=='diagnostic'),key=lambda r:(r['observation_index'],str(r['episode_id']),str(r['task'])))[:64]
    if len(rows)!=64 or len({r['replay_index'] for r in rows})!=64:raise ValueError('64distinct diagnostic observations required')
    adapter=importlib.import_module(a.adapter).adapter
    if digest(adapter.replay_manifest_path)!=identity:raise ValueError('Adapter replay manifest differs')
    torch.set_num_threads(8);model=adapter.load_model()
    if adapter.execution_configuration()!=qa['execution_configuration']:raise ValueError('Loaded policy differs from diagnostic path')
    samples=[adapter.load_replay_input(r['replay_index'],0) for r in rows];target=samples[0]
    a.split='diagnostic';a.batch_sizes=[64];a.modes=['native','batch_invariant'];a.freeze=None
    a.output.parent.mkdir(parents=True,exist_ok=True);record_runtime_provenance(a,adapter,identity)
    Path(str(a.output)+'.driver.py').write_bytes(Path(__file__).read_bytes());records=[]
    with set_batch_invariant_mode(True):pass
    for mode in a.modes:
        with set_batch_invariant_mode(mode=='batch_invariant'):
            reference=_run_once(adapter,model,_compose_batch(adapter,[target]),0)
            for kind in ('duplicate','diverse'):
                for position,index in (('first',0),('middle',32),('last',63)):
                    companions=[target]*63 if kind=='duplicate' else samples[1:]
                    requests=list(companions);requests.insert(index,target)
                    ids=[rows[0]['request_ids'][0]]*63 if kind=='duplicate' else [r['request_ids'][0] for r in rows[1:]]
                    record=dict(adapter=a.adapter,adapter_execution_configuration=adapter.execution_configuration(),replay_manifest_sha256=identity,
                        implementation=mode,batch_size=64,companion_type=kind,target_position=position,target_batch_index=index,
                        request_id=rows[0]['request_ids'][0],companion_ids=ids,diagnostic_replay_indices=[r['replay_index'] for r in rows],
                        explicit_noise_id=0,driver_sha256=digest(Path(__file__)),isolated_performance_qualified=False)
                    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    try:
                        output=_run_once(adapter,model,_compose_batch(adapter,requests),0);torch.cuda.synchronize()
                        record.update(status='ran',peak_incremental_bytes=max(0,torch.cuda.max_memory_allocated()-baseline),
                            output=metrics(sample_at(reference,0),sample_at(output,index)))
                        del output
                    except torch.cuda.OutOfMemoryError as error:
                        record.update(status='cuda_oom',error=str(error))
                    records.append(record)
                    with a.output.open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
                    print(f'{mode} B64 {kind}/{position}: {record["status"]}',flush=True)
                    if mode=='batch_invariant' and record['status']=='ran' and not record['output']['exact']:raise RuntimeError('Invariant B64 numerical violation retained')
    for path,h in qa['source_sha256'].items():
        if digest(Path(path))!=h:raise ValueError('Pinned numerical source changed during measurement')
    summary=dict(status='performance_capacity_measured_pending_independent_audit',records=len(records),manifest_sha256=identity,
        feasible={mode:all(r['status']=='ran' for r in records if r['implementation']==mode) for mode in a.modes},
        isolated_performance_qualified=False,full_campaign_complete=False)
    Path(str(a.output)+'.summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()

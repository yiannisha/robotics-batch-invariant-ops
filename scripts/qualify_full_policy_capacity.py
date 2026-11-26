"""Measure actual B16/B32 capacity on frozen diagnostic requests and schedules."""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch

if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch,_run_once
from scripts.run_model_replay_invariance import digest,metrics,sample_at,record_runtime_provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.split='diagnostic';args.batch_sizes=[16,32];args.modes=['native','batch_invariant'];args.freeze=None
    manifest=json.loads(args.manifest.read_text());manifest_hash=digest(args.manifest)
    adapter=importlib.import_module(args.adapter).adapter
    if digest(Path(adapter.replay_manifest_path))!=manifest_hash:raise ValueError('Adapter manifest differs')
    model=adapter.load_model()
    if hasattr(model,'eval'):model.eval()
    record_runtime_provenance(args,adapter,manifest_hash)
    driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(str(args.output)+'.driver.py').write_bytes(Path(__file__).read_bytes())
    row=next(r for r in manifest['observations'] if r['split']=='diagnostic')
    target=adapter.load_replay_input(row['replay_index'],0);target_id=row['request_ids'][0]
    indexes=row['companion_replay_indices'][:31]
    if len(indexes)!=31 or len(set(indexes))!=31:raise ValueError('Insufficient distinct real companions')
    other_rows=[manifest['observations'][i] for i in indexes]
    if any(r['split']!='diagnostic' or (r['task'],r['episode_id'])==(row['task'],row['episode_id']) for r in other_rows):
        raise ValueError('Capacity inputs violate split/target-episode isolation')
    others=[adapter.load_replay_input(i,0) for i in indexes]
    records=[]
    with set_batch_invariant_mode(True):pass
    for mode in args.modes:
        with set_batch_invariant_mode(mode=='batch_invariant'):
            reference=_run_once(adapter,model,_compose_batch(adapter,[target]),0)
            for size in args.batch_sizes:
                for kind in ('duplicate','diverse'):
                    companions=[target]*(size-1) if kind=='duplicate' else others[:size-1]
                    ids=[target_id]*(size-1) if kind=='duplicate' else [r['request_ids'][0] for r in other_rows[:size-1]]
                    for position,index in (('first',0),('middle',size//2),('last',size-1)):
                        requests=list(companions);requests.insert(index,target)
                        record=dict(adapter=args.adapter,adapter_execution_configuration=adapter.execution_configuration(),
                            replay_manifest_sha256=manifest_hash,request_id=target_id,companion_ids=ids,batch_size=size,
                            implementation=mode,companion_type=kind,target_position=position,gpu=torch.cuda.get_device_name(),
                            torch=torch.__version__,capacity_runner_sha256=driver_hash)
                        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                        baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        try:
                            output=_run_once(adapter,model,_compose_batch(adapter,requests),0)
                            torch.cuda.synchronize()
                            record.update(status='ran',peak_incremental_bytes=max(0,torch.cuda.max_memory_allocated()-baseline),
                                          output=metrics(sample_at(reference,0),sample_at(output,index)))
                            del output
                        except torch.cuda.OutOfMemoryError as error:
                            gc.collect();torch.cuda.empty_cache()
                            free,total=torch.cuda.mem_get_info()
                            record.update(status='cuda_oom',error=str(error),free_bytes_after_failure=free,total_bytes=total)
                        records.append(record)
                        with args.output.open('a') as out:out.write(json.dumps(record,sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
                        print(f'{mode} B{size} {kind}/{position}: {record["status"]}',flush=True)
    summary=dict(scope='real_diagnostic_complete_policy_capacity',records=len(records),
                 feasible={f'{mode}_B{size}':all(r['status']=='ran' for r in records if r['implementation']==mode and r['batch_size']==size) for mode in args.modes for size in args.batch_sizes},
                 invariant_failed_arrangements=sum(r['implementation']=='batch_invariant' and r['status']=='ran' and not r['output']['exact'] for r in records),
                 source_raw_sha256=digest(args.output))
    Path(str(args.output)+'.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if summary['invariant_failed_arrangements']:raise RuntimeError('Invariant output changed at capacity probe; retain and investigate')


if __name__=='__main__':main()

"""Audit full timing/queue protocols and retained GPU isolation observations."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.audit_full_cross_model_campaign import timing_audit,queue_audit
from scripts.freeze_four_model_numerics import MODELS


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite measurement')
    elif isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,list):
        for item in value:finite(item)


def isolation(report,benchmark,freeze_hash,gpu_name):
    require(report.get('accepted') is True and report.get('status')=='completed' and report.get('child_exit_code')==0,'Unaccepted isolation run')
    require(report['benchmark']==benchmark and report['freeze_sha256']==freeze_hash,'Isolation benchmark/freeze differs')
    require(report['poll_interval_seconds']==1.0,'Wrong inventory sampling protocol')
    child=report['benchmark_pid'];require(isinstance(child,int) and child>0,'Missing benchmark PID')
    observations=report['observations'];require(len(observations)>=2 and observations[0]['processes']==[],'Missing idle prelaunch observation')
    times=[];drivers=set()
    for observed in observations:
        require(observed['gpu_uuid']==report['gpu_uuid'] and observed['gpu_name']==gpu_name,'Observed hardware differs from freeze')
        # Benchmarks use one CUDA process. Requiring its exact PID also avoids
        # trusting historical, no-longer-queryable process-group membership.
        require(all(p['pid']==child for p in observed['processes']),'Foreign compute/graphics process in retained inventory')
        times.append(datetime.fromisoformat(observed['observed_at_utc']));drivers.add(observed['driver_version'])
    require(times==sorted(times) and len(drivers)==1,'Inventory clock/driver changed')
    require(datetime.fromisoformat(report['started_at_utc'])<=datetime.fromisoformat(report['finished_at_utc']),'Invalid run interval')
    return dict(samples=len(times),maximum_observation_gap_seconds=max((b-a).total_seconds() for a,b in zip(times,times[1:])),
                gpu_uuid=report['gpu_uuid'],gpu_name=gpu_name,driver_version=next(iter(drivers)),
                scope='Sampled compute/graphics inventories; no exclusive reservation or between-sample guarantee.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-id',choices=MODELS,required=True)
    for name in ('directory','freeze','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve existing audits')
    freeze=json.loads(a.freeze.read_text());freeze_hash=sha(a.freeze);require(freeze['model_id']==a.model_id,'Wrong frozen model')
    require(freeze['prepared_input_disjointness_verified'] and freeze['invariant_diagnostic_exact'] and freeze['full_traced_operator_qualification_complete'],'Incomplete numerical freeze')
    bound=freeze['content_addressed_freeze'];runtime_paths=[Path(x) for x in bound if x.endswith('invariance.jsonl.gz.runtime.json')]
    require(len(runtime_paths)==1,'Ambiguous frozen diagnostic runtime')
    for path,entry in bound.items():
        file=Path(path);require(file.stat().st_size==entry['bytes'] and sha(file)==entry['sha256'],'Frozen bytes changed: '+path)
    runtime=json.loads(runtime_paths[0].read_text());configuration=freeze['adapter_execution_configuration'];adapter=MODELS[a.model_id][0]
    require(runtime['adapter']==adapter and runtime['execution_configuration']==configuration,'Frozen adapter configuration differs')
    proofs={};summaries={};sources={};latencies=defaultdict(list)
    for benchmark in ('policy','serving','queue'):
        path=a.directory/(benchmark+'.jsonl');proof_path=a.directory/(benchmark+'_isolation.json');report=json.loads(proof_path.read_text())
        proofs[benchmark]=isolation(report,benchmark,freeze_hash,runtime['gpu'])
        require(report['freeze']==str(a.freeze.resolve()) and report['benchmark_output']==str(path.resolve()) and report['benchmark_output_sha256']==sha(path),'Isolation measurement bytes differ')
        archive=Path(report['source_archive'])
        for file,h in report['benchmark_source_sha256'].items():
            require(sha(archive/h)==sha(Path(file))==h,'Benchmark source/archive changed');sources[file]=h
        log=a.directory/(benchmark+'.log');require(sha(log)==report['child_log_sha256'],'Benchmark log changed')
        sources[str(proof_path.resolve())]=sha(proof_path);sources[str(path.resolve())]=sha(path)
        if benchmark=='queue':
            for file in (a.directory/'serving.jsonl',a.directory/'serving_isolation.json'):
                require(report['calibration_input_sha256'].get(str(file.resolve()))==sha(file),'Queue calibration differs from isolated serving bytes')
        with path.open() as stream:
            for line in stream:
                row=json.loads(line);finite(row)
                if row.get('record_type')=='environment':
                    require(row['adapter']==adapter and row['gpu']==runtime['gpu'],'Timing environment differs from freeze')
                    selection=row['input_selection'];require(selection['replay_manifest_path'] in bound and selection['replay_manifest_sha256']==bound[selection['replay_manifest_path']]['sha256'],'Timing input selection is not freeze-bound')
                else:require(row.get('adapter_execution_configuration')==configuration,'Measured execution configuration differs')
                if benchmark=='queue':
                    for key in ('input_selection',):
                        selection=row[key];require(selection['replay_manifest_path'] in bound and selection['replay_manifest_sha256']==bound[selection['replay_manifest_path']]['sha256'],'Queue inputs not freeze-bound')
                elif row.get('record_type')=='timing':
                    name=f"{row['implementation']}|B{row['batch_size']}" if benchmark=='policy' else row['configuration']
                    latencies[(benchmark,name)].append(row['latency_seconds'])
        issues=[]
        if benchmark=='queue':summary=queue_audit(path,issues,logical_batch_size=4,fixed_batch_size=8,serving_timings_path=a.directory/'serving.jsonl')
        else:summary=timing_audit(path,benchmark,issues,required_policy_batches=(1,2,8,64))
        require(summary['complete'] and not issues,'Incomplete performance protocol: '+str(issues[:5]));summaries[benchmark]=summary
    require(len({x['gpu_uuid'] for x in proofs.values()})==1,'Performance stages used different physical GPUs')
    require(sha(a.freeze)==freeze_hash,'Freeze changed during audit')
    result=dict(status='complete_isolated_four_model_performance_independently_verified',model_id=a.model_id,freeze_sha256=freeze_hash,
                isolation=proofs,protocol_counts=summaries,source_evidence_sha256=sources,
                median_latency_seconds={f'{kind}|{name}':statistics.median(v) for (kind,name),v in sorted(latencies.items())},
                audited_raw_timing_calls=sum(map(len,latencies.values())),auditor_sha256=sha(Path(__file__)),
                independent_cardinality_auditor_sha256=sha(Path(__file__).with_name('audit_full_cross_model_campaign.py')),
                full_four_model_campaign_complete=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.output)


if __name__=='__main__':main()

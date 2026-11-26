"""Independently audit measured CogACT linear source layouts and 100-seed coverage."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from scripts.audit_pi05_actual_operators import check_metric


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','records','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args();config=json.loads(a.config.read_text());identity=sha(a.config)
    traces={}
    for path,expected in config['traces'].items():
        require(sha(Path(path))==expected,'Source trace changed');r=json.loads(Path(path).read_text())
        require(r['adapter']==config['adapter'] and r['adapter_execution_configuration']==config['adapter_execution_configuration'],'Wrong source model/protocol')
        traces[r['batch_size']]=[row for row in r['operations'] if row['aten_operation']=='aten::linear']
    require(set(traces)=={1,8},'Missing B1/B8 source trace')
    configs={c['id']:c for c in config['configurations']};require(len(configs)==len(config['configurations']),'Duplicate source configuration')
    used_one=set();used_eight=set()
    for c in configs.values():
        source=c['source'];u=source['b1_index'];v=source['b8_index'];used_one.add(u);used_eight.add(v)
        require(source['b1']==traces[1][u] and source['b8']==traces[8][v],'Config was not observed in source')
        require(source['b1']['arguments'][1:]==source['b8']['arguments'][1:],'Source weight/bias changed across batches')
        x=source['b1']['arguments'][0];y=source['b8']['arguments'][0];axis=c['batch_axis'];shape=list(x['shape'])
        expected_layout='cfg_split_branches' if axis==0 and x['shape'][0]==2 else 'contiguous_requests'
        require(c['coordinate_layout']==expected_layout,'Source CFG request coordinates differ')
        if axis is not None:
            require(c['groups_per_request']==shape[axis] and axis<len(shape)-1,'Wrong logical request grouping')
            shape[axis]*=8
        require(shape==y['shape'],'Wrong B1/B8 source shape relation')
    require(used_one==set(range(len(traces[1]))) and used_eight==set(range(len(traces[8]))),'Source union has holes')
    rows=defaultdict(list);raw_sha=hashlib.sha256();record_count=0
    with a.records.open() as stream:
        for line in stream:
            raw_sha.update(line.encode());r=json.loads(line);record_count+=1
            require(r['configuration_sha256']==identity and r['tf32_matmul'] is False,'Mixed config/precision')
            for path,expected in r['source_sha256'].items():
                require(sha(a.records.parent/'source_linear_qualifier_sources'/expected)==expected,'Qualifier source archive changed')
            name,batch_text,position=r['arrangement'].split('|');batch=int(batch_text[1:]);c=configs[name]
            require(batch in (1,2,4,8,16,31,32,33,64) and position in (('first',) if batch==1 else ('first','last')),'Invalid arrangement')
            require(r['batch_size']==batch and r['target_position']==position and r['batch_axis']==c['batch_axis'] and r['groups_per_request']==c['groups_per_request'],'Wrong request/target metadata')
            require(r['coordinate_layout']==c['coordinate_layout'],'Measured CFG request layout differs')
            index=0 if position=='first' else batch-1
            expected_indices=[index,batch+index] if c['coordinate_layout']=='cfg_split_branches' else None
            require(r['target_axis_indices']==expected_indices,'Wrong conditional/unconditional target coordinates')
            x=c['source']['b1']['arguments'][0];y=c['source']['b8']['arguments'][0];shape=list(x['shape']);axis=c['batch_axis']
            if axis is not None:shape[axis]*=batch
            stride=[]
            for before,after in zip(x['stride'],y['stride']):
                require(after==before or after==8*before,'Uncovered source stride variation')
                stride.append(before if after==before else before*batch)
            require(r['left_shape']==shape and r['left_stride']==stride,'Measured batch input differs from source layout')
            require(r['target_shape']==x['shape'] and r['target_stride']==x['stride'],'Canonical singleton source layout changed')
            weight=c['source']['b1']['arguments'][1]
            require(r['weight_shape']==weight['shape'] and r['weight_stride']==weight['stride'],'Source weight layout changed')
            shape[-1]=weight['shape'][0]
            require(r['output_shape']==shape and r['output_dtype']==x['dtype'],'Output shape/precision differs')
            require(r['seed'] in range(100) and r['generator_seed']==1000000+r['seed'],'Seed protocol differs')
            for field in ('native_batch_invariance','invariant_batch_invariance','singleton_fidelity','reference_batch_invariance'):check_metric(r[field])
            require(r['invariant_batch_invariance']['exact'] and r['reference_batch_invariance']['exact'],'Invariant or per-request coordinate reference failed')
            if r['seed']<3:
                require(r['fp64_reference'] is not None,'Missing FP64 reference')
                for metric in r['fp64_reference'].values():
                    check_metric(metric);require(all(t['dtype']=='torch.float64' for t in metric['tensors'].values()),'False FP64 reference')
            if r['seed']==0:
                profile=r['dispatch_profile'];require(profile and 'aten::linear' in profile['aten_events'],'Missing high-level linear dispatch')
                kernels=profile['cuda_kernels'];require(any('matmul_kernel_persistent' in k for k in kernels),'Invariant CUDA kernel missing')
                require(any(any(token in k.lower() for token in ('gemm','gemv','cutlass','cublas')) for k in kernels),'Native CUDA matrix kernel missing')
            rows[r['arrangement']].append(r)
    arrangements=[]
    for key,items in sorted(rows.items()):
        seeds=[r['seed'] for r in items];require(len(seeds)==len(set(seeds)),'Duplicate measured seed')
        arrangements.append(dict(key=key,measured_seeds=len(seeds),all100seeds=set(seeds)==set(range(100)),
            native_nonexact_seeds=sum(not r['native_batch_invariance']['exact'] for r in items),
            singleton_fidelity_max_absolute_error=max(r['singleton_fidelity']['max_absolute_error'] for r in items)))
    expected={f'{name}|B{b}|{position}' for name in configs for b in (1,2,4,8,16,31,32,33,64) for position in (('first',) if b==1 else ('first','last'))}
    complete={row['key'] for row in arrangements if row['all100seeds']};all_complete=complete==expected
    require(not a.require_complete or all_complete,'Linear source sweep incomplete')
    report=dict(status='source_linear_measured_arrangements_independently_verified',raw_records=record_count,raw_records_sha256=raw_sha.hexdigest(),
        configuration_sha256=identity,auditor_sha256=sha(Path(__file__)),arrangements=arrangements,
        all_linear_arrangements_complete=all_complete,prescribed_arrangements=len(expected),completed_arrangements=len(complete),
        full_operator_qualification_complete=False,shared_gpu_timing_not_qualified=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

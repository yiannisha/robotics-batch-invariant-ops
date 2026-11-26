"""Reconcile RDT grouped source matrix layouts against raw qualification data."""
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
    for name in ('config','records','sources','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());identity=sha(a.config);traces={}
    for path,expected in config['source'].items():
        require(sha(Path(path))==expected,'Source trace changed')
        t=json.loads(Path(path).read_text());traces[t['batch_size']]=t
        require(t['adapter_execution_configuration']==config['adapter_execution_configuration'],'Source execution differs')
    cases={c['id']:c for c in config['configurations']}
    for c in cases.values():
        require(c['operation']=='matmul' and c['layout']=='source_strided_matmul' and c['shared_bystander'] is True,'Wrong shared operation/layout')
        require(c['source_entries']['b1']==c['source_entries']['b8'],'Shared source operation changed with logical batch')
        for batch in (1,8):
            row=c['source_entries']['b'+str(batch)]
            require(row in traces[batch]['operations'] and row['aten_operation']=='aten::'+c['operation'],'Source operation absent')
            require(not row['keyword_arguments'],'Uncovered source kwargs')
            for i,which in enumerate(('left','right')):
                x=row['arguments'][i];shape=c['source_'+which+'_shape']
                require(shape[0]==c['m_per_request'] and x['shape']==shape,'Source grouping differs')
                require(x['stride']==c[f'source_{which}_stride_b{batch}'] and x['dtype']==c['dtype'],'Source layout/precision differs')
    groups=defaultdict(list);raw_hash=hashlib.sha256();sources=set();count=0
    with a.records.open() as stream:
        for line in stream:
            raw_hash.update(line.encode());r=json.loads(line);count+=1
            require(r['configuration_sha256']==identity,'Mixed configuration')
            name,batch,position=r['operator_configuration'].split('|');c=cases[name];b=int(batch[1:])
            require(b in (1,2,4,8,16,31,32,33,64) and position in (('first',) if b==1 else ('first','last')),'Foreign arrangement')
            require(r['batch_size']==b and r['target_position']==position and r['shared_bystander'] is True and r['physical_outer_size']==1,'Logical/physical shared operation metadata differs')
            require(r['singleton_inputs_exact_reconstructed'],'Singleton input values not checked')
            for which in ('left','right'):
                require(r['singleton_'+which+'_stride']==cases[name][f'source_{which}_stride_b1'],'Singleton source layout differs')
            require(r['seed'] in range(100) and r['generator_seed']==1000000+r['seed'],'Seed protocol differs')
            require(not r['tf32_matmul'] and not r['tf32_cudnn'] and not r['cudnn_benchmark'],'Precision protocol differs')
            for which in ('left','right'):
                shape=c['source_'+which+'_shape'];one=c[f'source_{which}_stride_b1'];eight=c[f'source_{which}_stride_b8']
                require(all(x==y or y==8*x for x,y in zip(one,eight)),'Unsupported stride extrapolation')
                require(r[which+'_shape']==shape,'Measured grouping differs')
                require(r[which+'_stride']==[x if x==y else b*x for x,y in zip(one,eight)],'Measured source strides differ')
            h=r['qualifier_source_sha256']
            if h not in sources:
                require(sha(a.sources/(h+'.py'))==h,'Qualifier archive changed');sources.add(h)
            for key in ('native_batch_invariance','invariant_batch_invariance','singleton_fidelity','reference_batch_invariance','reference_singleton_fidelity'):
                check_metric(r[key])
                require(all(v['dtype']==c['dtype'] for v in r[key]['tensors'].values()),'Measured precision differs')
            require(r['invariant_batch_invariance']['exact'] and r['reference_batch_invariance']['exact'],'Invariant/reference mismatch')
            if r['seed']<3:
                require(r['fp64_reference'] is not None,'Missing FP64 comparison')
                for metric in r['fp64_reference'].values():
                    check_metric(metric);require(all(v['dtype']=='torch.float64' for v in metric['tensors'].values()),'Reference not FP64')
            if r['seed']==0:
                profile=r['dispatch_profile'];require(profile and profile['cuda_kernels'] and 'aten::'+c['operation'] in profile['aten_events'],'Missing actual operator dispatch')
            groups[r['operator_configuration']].append(r)
    summaries=[]
    for key,rows in sorted(groups.items()):
        seeds=[r['seed'] for r in rows];require(len(seeds)==len(set(seeds)),'Duplicate seed')
        summaries.append(dict(arrangement=key,measured_seeds=len(seeds),complete=set(seeds)==set(range(100)),native_nonexact_seeds=sum(not r['native_batch_invariance']['exact'] for r in rows)))
    expected={f'{name}|B{b}|{pos}' for name in cases for b in (1,2,4,8,16,31,32,33,64) for pos in (('first',) if b==1 else ('first','last'))}
    complete={r['arrangement'] for r in summaries if r['complete']}
    require(not a.require_complete or complete==expected,'Incomplete matrix sweep')
    report=dict(status='shared_source_matmul_measurements_independently_verified',configuration_sha256=identity,raw_records=count,raw_records_sha256=raw_hash.hexdigest(),auditor_sha256=sha(Path(__file__)),arrangements=summaries,completed_arrangements=len(complete),prescribed_arrangements=len(expected),all_selected_arrangements_complete=complete==expected,full_model_operator_qualification_complete=False,timing_not_qualified=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='arrangements'},indent=2))


if __name__=='__main__':main()

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
        require(c['operation']=='sdpa' and c['layout']=='qkv_source_strided','Wrong attention operation/layout')
        for batch in (1,8):
            row=c['source_entries']['b'+str(batch)]
            require(row in traces[batch]['operations'] and row['aten_operation']=='aten::scaled_dot_product_attention','Source attention absent')
            require(not row['keyword_arguments'] and c['scale']==c['head_dim']**-.5 and not c['enable_gqa'],'Uncovered attention options')
            for i,(which,heads,length,dim) in enumerate((('q','query_heads','query_length','head_dim'),('k','key_heads','key_length','head_dim'),('v','value_heads','key_length','value_dim'))):
                x=row['arguments'][i]
                require(x['shape']==[batch,c[heads],c[length],c[dim]] and x['stride']==c['source_'+which+'_stride'] and x['dtype']==c['dtype'],'Source QKV layout differs')
            if c.get('mask_values') is not None:
                mask=row['arguments'][3]
                require(mask['dtype']=='torch.bool' and mask['shape']==[batch,1,c['query_length'],c['key_length']] and mask['stride']==[c['key_length'],c['key_length'],0,1],'Source mask layout differs')
                import numpy as np
                values=np.asarray(mask['values'],dtype=bool)
                require(np.array_equal(np.broadcast_to(values,mask['shape']),np.broadcast_to(c['mask_values'],mask['shape'])),'Captured mask values differ')
            else:require(len(row['arguments'])==3,'Mask dropped from source')
    groups=defaultdict(list);raw_hash=hashlib.sha256();sources=set();count=0
    with a.records.open() as stream:
        for line in stream:
            raw_hash.update(line.encode());r=json.loads(line);count+=1
            require(r['configuration_sha256']==identity,'Mixed configuration')
            name,batch,position=r['operator_configuration'].split('|');c=cases[name];b=int(batch[1:])
            require(b in (1,2,4,8,16,31,32,33,64) and position in (('first',) if b==1 else ('first','last')),'Foreign arrangement')
            require(r['batch_size']==b and r['target_position']==position,'Arrangement metadata differs')
            require(r['seed'] in range(100) and r['generator_seed']==1000000+r['seed'],'Seed protocol differs')
            require(not r['tf32_matmul'] and not r['tf32_cudnn'] and not r['cudnn_benchmark'],'Precision protocol differs')
            for which,heads,length,dim in (('left','query_heads','query_length','head_dim'),('right','key_heads','key_length','head_dim'),('value','value_heads','key_length','value_dim')):
                stride=c['source_'+{'left':'q','right':'k','value':'v'}[which]+'_stride']
                require(r[which+'_shape']==[b,c[heads],c[length],c[dim]] and r[which+'_stride']==stride,'Measured QKV differs')
            if c.get('reference_contract')=='explicit_per_request_full_source_head_extent':
                index=0 if position=='first' else b-1
                require(c['m_per_request']==1 and r['coordinate_layout']=='contiguous_requests' and r['target_axis_indices']==[index],'Full-head reference target coordinates differ')
                for key in ('q','k','v'):require(r['singleton_'+key+'_stride']==c['source_'+key+'_stride'],'Source singleton strides differ')
                require(r['attention_mask_sha256'] is None and r['singleton_attention_mask_sha256'] is None,'Unmasked reference unexpectedly has a mask')
            mask=r.get('attention_mask')
            if c.get('mask_values') is None:require(mask is None,'Unexpected mask')
            else:
                require(mask['dtype']=='torch.bool' and mask['shape']==[b,1,c['query_length'],c['key_length']] and mask['stride']==[c['key_length'],c['key_length'],0,1],'Measured mask layout differs')
                require(mask['values']==c['mask_values'] and mask['sha256']==hashlib.sha256(bytes(c['mask_values'])).hexdigest(),'Measured mask bytes differ')
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
                profile=r['dispatch_profile'];require(profile and profile['cuda_kernels'] and 'aten::scaled_dot_product_attention' in profile['aten_events'],'Missing actual operator dispatch')
            groups[r['operator_configuration']].append(r)
    summaries=[]
    for key,rows in sorted(groups.items()):
        seeds=[r['seed'] for r in rows];require(len(seeds)==len(set(seeds)),'Duplicate seed')
        summaries.append(dict(arrangement=key,measured_seeds=len(seeds),complete=set(seeds)==set(range(100)),native_nonexact_seeds=sum(not r['native_batch_invariance']['exact'] for r in rows)))
    expected={f'{name}|B{b}|{pos}' for name in cases for b in (1,2,4,8,16,31,32,33,64) for pos in (('first',) if b==1 else ('first','last'))}
    complete={r['arrangement'] for r in summaries if r['complete']}
    require(not a.require_complete or complete==expected,'Incomplete attention sweep')
    report=dict(status='source_attention_measurements_independently_verified',configuration_sha256=identity,raw_records=count,raw_records_sha256=raw_hash.hexdigest(),auditor_sha256=sha(Path(__file__)),arrangements=summaries,completed_arrangements=len(complete),prescribed_arrangements=len(expected),all_selected_arrangements_complete=complete==expected,full_model_operator_qualification_complete=False,timing_not_qualified=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='arrangements'},indent=2))


if __name__=='__main__':main()

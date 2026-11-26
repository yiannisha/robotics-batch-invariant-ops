"""Reconcile RDT grouped source matrix layouts against raw qualification data."""
import argparse
import base64
import zlib
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
        require(c['operation']=='sdpa' and c['layout']=='qkv_source_strided','Wrong attention source operation')
        group=c['m_per_request'];require(group in (1,2),'Unknown logical request grouping')
        require(c['coordinate_layout']==('cfg_split_branches' if group==2 else 'contiguous_requests'),'Wrong CFG source coordinates')
        for batch in (1,8):
            row=c['source_entries']['b'+str(batch)]
            require(row in traces[batch]['operations'] and row['aten_operation']=='aten::scaled_dot_product_attention','Source attention absent')
            require(not row['keyword_arguments'] and c['scale']==c['head_dim']**-.5 and not c['enable_gqa'],'Uncovered attention options')
            for i,(which,heads,length,dim) in enumerate((('q','query_heads','query_length','head_dim'),('k','key_heads','key_length','head_dim'),('v','value_heads','key_length','value_dim'))):
                x=row['arguments'][i]
                require(x['shape']==[batch*group,c[heads],c[length],c[dim]] and x['stride']==c['source_'+which+'_stride'] and x['dtype']==c['dtype'],'Source QKV layout differs')
            capture=c['mask_capture_b1'] if batch==1 else c['mask_capture']
            require(capture==row['attention_mask'],'Source additive mask capture differs')
            if capture:
                require(group==1 and capture['dtype']==c['dtype']=='torch.bfloat16','Unexpected source mask dtype/grouping')
                payload=zlib.decompress(base64.b64decode(capture['data']))
                require(len(payload)==capture['original_dtype_bytes_count'] and hashlib.sha256(payload).hexdigest()==capture['original_dtype_bytes_sha256'],'Captured additive mask bytes changed')
        if c['mask_capture']:
            one=zlib.decompress(base64.b64decode(c['mask_capture_b1']['data']));eight=zlib.decompress(base64.b64decode(c['mask_capture']['data']))
            require(eight[:len(one)]==one,'B1 mask is not B8 request0')
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
            group=c['m_per_request'];index=0 if position=='first' else b-1
            expected_indices=[index,b+index] if group==2 else [index]
            require(r['coordinate_layout']==c['coordinate_layout'] and r['target_axis_indices']==expected_indices,'Measured CFG coordinates differ')
            for which,source,heads,length,dim in (('left','q','query_heads','query_length','head_dim'),('right','k','key_heads','key_length','head_dim'),('value','v','value_heads','key_length','value_dim')):
                stride=c['source_'+source+'_stride']
                require(r[which+'_shape']==[b*group,c[heads],c[length],c[dim]] and r[which+'_stride']==stride,'Measured QKV differs')
                require(r['singleton_'+source+'_stride']==stride,'Singleton source strides differ')
            capture=c['mask_capture']
            if capture is None:require(r['attention_mask_sha256'] is None and r['singleton_attention_mask_sha256'] is None,'Unexpected additive mask')
            else:
                payload=zlib.decompress(base64.b64decode(capture['data']));row_bytes=len(payload)//8
                chunks=[payload[i*row_bytes:(i+1)*row_bytes] for i in range(8)]
                full=b''.join(chunks[i%8] for i in range(b))
                require(r['attention_mask_sha256']==hashlib.sha256(full).hexdigest() and r['singleton_attention_mask_sha256']==hashlib.sha256(chunks[index%8]).hexdigest(),'Measured full or selected singleton mask bytes differ')
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

"""Independently reconstruct source layouts and audit retained measurements."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from scripts.audit_pi05_actual_operators import check_metric


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_layout(case,index,batch):
    one=case['source_entries']['b1']['arguments'][index]
    eight=case['source_entries']['b8']['arguments'][index]
    shape=list(one['shape']);axis=case['batch_axis']
    if axis is not None:shape[axis]*=batch
    strides=[]
    for i,(s,t) in enumerate(zip(one['stride'],eight['stride'])):
        if s==t:strides.append(s)
        elif i==1 and axis==1 and one['shape'][i]==1 and t==0:strides.append(s if batch==1 else 0)
        else:
            require(t==8*s,'Uncovered stride relation');strides.append(s*batch)
    return shape,strides


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','records','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());identity=sha(a.config)
    require(config['adapter']=='scripts.model_invariance.adapters.lingbot_robotwin_exact','Wrong adapter')
    traces={}
    for path,h in config['traces'].items():
        require(sha(Path(path))==h,'Changed source trace');t=json.loads(Path(path).read_text())
        require(t['adapter']==config['adapter'] and t['adapter_execution_configuration']==config['adapter_execution_configuration'],'Foreign source protocol')
        traces[t['batch_size']]=t
    require(set(traces)=={1,8},'Missing source batches')
    cases={c['id']:c for c in config['configurations']}
    require(len(cases)==3,'Expected three remaining patterns')
    for c in cases.values():
        for b in (1,8):require(c['source_entries']['b'+str(b)] in traces[b]['operations'],'Unobserved source case')
        args=c['source_entries']['b1']['arguments']
        if c['operation']=='aten::mean':
            require(c['batch_axis'] is None and args[0]['shape']==[8,32,2560] and args[1:]==[[1]],'Wrong shared pooling')
        else:
            require(c['operation']=='aten::matmul' and c['batch_axis']==1 and len(args)==2,'Wrong rotary operation')
            require(args[0]['shape']==[3,1,64,1] and args[1]['shape'] in ([3,1,1,286],[3,1,1,51]),'Wrong rotary operands')
        for b in (1,8):
            for i in range(len(args) if c['operation']=='aten::matmul' else 1):
                shape,stride=expected_layout(c,i,b);observed=c['source_entries']['b'+str(b)]['arguments'][i]
                require(shape==observed['shape'] and stride==observed['stride'],'Reconstructed layout differs from trace')
    sizes=(1,2,4,8,16,31,32,33,64)
    expected={f'{name}|B{b}|{pos}' for name in cases for b in sizes for pos in (('first',) if b==1 else ('first','last'))}
    rows=defaultdict(set);raw=hashlib.sha256();count=0
    with a.records.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'),'Truncated record');raw.update(line);r=json.loads(line);count+=1
            require(r['configuration_sha256']==identity and r['tf32_matmul'] is False,'Changed configuration or precision')
            for path,h in r['source_sha256'].items():require(sha(a.records.parent/'remaining_qualifier_sources'/h)==h,'Changed archived source')
            key=r['arrangement'];require(key in expected,'Unprescribed arrangement');name,btext,pos=key.split('|');batch=int(btext[1:]);c=cases[name]
            require(r['batch_size']==batch and r['target_position']==pos and r['batch_axis']==c['batch_axis'],'Wrong request metadata')
            require(r['seed'] in range(100) and r['generator_seed']==1000000+r['seed'] and r['seed'] not in rows[key],'Wrong or duplicate seed')
            n=2 if c['operation']=='aten::matmul' else 1
            layouts=[expected_layout(c,i,batch) for i in range(n)]
            require(r['input_shapes']==[x[0] for x in layouts] and r['input_strides']==[x[1] for x in layouts],'Wrong measured source layout')
            singles=[expected_layout(c,i,1) for i in range(n)]
            require(r['singleton_shapes']==[x[0] for x in singles] and r['singleton_strides']==[x[1] for x in singles],'Wrong singleton layout')
            out=[8,2560] if n==1 else [3,batch,64,c['source_entries']['b1']['arguments'][1]['shape'][-1]]
            require(r['output_shape']==out and r['output_dtype']=='torch.float32','Wrong output shape/dtype')
            for field in ('native_batch_invariance','invariant_batch_invariance','singleton_fidelity','reference_batch_invariance'):check_metric(r[field])
            require(r['invariant_batch_invariance']['exact'] and r['reference_batch_invariance']['exact'],'Invariant/reference disagreement')
            if r['seed']<3:
                require(r['fp64_reference'] is not None,'Missing FP64 reference')
                for metric in r['fp64_reference'].values():
                    check_metric(metric);require(all(x['dtype']=='torch.float64' for x in metric['tensors'].values()),'False FP64 reference')
            if r['seed']==0:
                prof=r['dispatch_profile'];require(prof and prof['cuda_kernels'] and c['operation'] in prof['aten_events'],'Missing source CUDA dispatch')
            rows[key].add(r['seed'])
    complete={k for k,v in rows.items() if v==set(range(100))};full=complete==expected
    require(not a.require_complete or full,'Remaining operator measurements incomplete')
    report=dict(status='remaining_source_operator_measurements_independently_verified',configuration_sha256=identity,
                raw_records=count,raw_records_sha256=raw.hexdigest(),auditor_sha256=sha(Path(__file__)),
                completed_arrangements=len(complete),prescribed_arrangements=len(expected),all_selected_arrangements_complete=full,
                full_operator_qualification_complete=False,performance_qualified=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

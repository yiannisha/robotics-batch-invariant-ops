"""Audit selected real-source convolution/reduction runs without claiming all ops."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from scripts.audit_pi05_actual_operators import check_metric,check_operands,check_source_config


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','records','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--qualifier-sources',type=Path,required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());identity=sha(a.config);traces={};source=config['source']
    if 'trace' in source:
        path=Path(source['trace']);require(sha(path)==source['sha256'],'Source trace changed')
        trace=json.loads(path.read_text());require(source['observed_operation'] in trace['operations'],'Selected source operation absent')
        traces[trace['batch_size']]=trace
    else:
        for path,expected in source.items():
            require(sha(Path(path))==expected,'Source trace changed');trace=json.loads(Path(path).read_text());traces[trace['batch_size']]=trace
    require(config['adapter_execution_configuration']['tf32'] is False,'Expected TF32-disabled model path')
    for trace in traces.values():require(trace['adapter_execution_configuration']==config['adapter_execution_configuration'],'Source/model inference configuration differs')
    configurations={c['id']:c for c in config['configurations']}
    for c in configurations.values():
        if c['operation']=='softmax' and 'observed_operation' in source:
            entry=source['observed_operation'];x=entry['arguments'][0];batch=next(iter(traces))
            require(entry['aten_operation']=='aten::softmax' and entry['arguments'][1]==c['dim']==-1,'Wrong source softmax dimension')
            require(x['shape']==[batch*c['m_per_request'],*c['input_per_unit']] and x['dtype'].removeprefix('torch.')==c['dtype'].removeprefix('torch.'),'Source softmax shape/dtype differs')
            require(entry['arguments'][2].removeprefix('torch.')==c['output_dtype'].removeprefix('torch.'),'Source softmax output precision lost')
        else:
            require(c['operation'] in ('mean','conv2d','softmax'),'Unsupported selected source operation')
            require(c['source_entries']['b1'] in traces[1]['operations'] and c['source_entries']['b8'] in traces[8]['operations'],'Selected source cases absent')
            one=c['source_entries']['b1'];eight=c['source_entries']['b8'];args=one['arguments'];x=args[0]
            require(not one['keyword_arguments'] and not eight['keyword_arguments'],'Uncovered source kwargs')
            from scripts.audit_pi05_actual_operators import contiguous
            require(x['stride']==contiguous(x['shape']) and eight['arguments'][0]['stride']==contiguous(eight['arguments'][0]['shape']),'Source input layout differs')
            require(x['dtype'].removeprefix('torch.')==c['dtype'].removeprefix('torch.'),'Source precision differs')
            if c['operation']=='mean':
                one=c['source_entries']['b1'];eight=c['source_entries']['b8'];args=one['arguments'];x=args[0]
                require(one['aten_operation']==eight['aten_operation']=='aten::mean','Wrong mean source operation')
                require(x['shape']==[c['m_per_request'],*c['input_per_unit']] and eight['arguments'][0]['shape']==[8*c['m_per_request'],*c['input_per_unit']],'Mean source grouping differs')
                require(x['dtype'].removeprefix('torch.')==c['dtype'].removeprefix('torch.'),'Mean source precision differs')
                require(args[1]==[c['dim']] and c['dim'] not in (0,-len(x['shape'])),'Mean dimension differs or reduces request axis')
                require((args[2] if len(args)>2 else False)==c['keepdim'] and not one['keyword_arguments'] and not eight['keyword_arguments'],'Uncovered mean options')
                from scripts.audit_pi05_actual_operators import contiguous
                require(x['stride']==contiguous(x['shape']) and eight['arguments'][0]['stride']==contiguous(eight['arguments'][0]['shape']),'Mean source layout differs')
            elif c['operation']=='conv2d':
                require(one['aten_operation']==eight['aten_operation']=='aten::conv2d' and len(args)==4,'Wrong source convolution')
                require(x['shape']==[c['m_per_request'],*c['input_per_image']] and eight['arguments'][0]['shape']==[8*c['m_per_request'],*c['input_per_image']],'Convolution source grouping differs')
                require(c['weight_shape']==args[1]['shape'] and args[1]['stride']==contiguous(args[1]['shape']) and args[1]['dtype']==x['dtype'],'Source convolution weight differs')
                require(c['has_bias']==(args[2] is not None) and (args[2] is None or args[2]['shape']==[args[1]['shape'][0]] and args[2]['dtype']==x['dtype']),'Source convolution bias differs')
                require(c['stride']==args[3] and c['padding']==[0,0] and c['dilation']==[1,1] and c['groups']==1 and c['layout']=='contiguous','Convolution source options differ')
            else:
                require(one['aten_operation']==eight['aten_operation']=='aten::softmax' and len(args) in (2,3),'Wrong source softmax')
                require(x['shape']==[c['m_per_request'],*c['input_per_unit']] and eight['arguments'][0]['shape']==[8*c['m_per_request'],*c['input_per_unit']],'Softmax source grouping differs')
                require(args[1]==c['dim']==-1,'Source softmax dimension differs')
                source_dtype=args[2] if len(args)==3 and args[2] is not None else x['dtype']
                require(source_dtype.removeprefix('torch.')==c.get('output_dtype',c['dtype']).removeprefix('torch.'),'Source softmax default/explicit output precision lost')
            require(c['source_entries']['b8']['arguments'][1:]==c['source_entries']['b1']['arguments'][1:],'Source operator parameters differ')
    rows=defaultdict(list);raw_hash=hashlib.sha256();archives=set();records=0
    with a.records.open() as stream:
        for line in stream:
            raw_hash.update(line.encode());r=json.loads(line);records+=1
            require(r['configuration_sha256']==identity,'Mixed config raw records');key=r['operator_configuration'];name,batch,position=key.split('|');c=configurations[name]
            require(r['batch_size']==int(batch[1:]) and r['target_position']==position,'Wrong batch/position metadata')
            require(r['batch_size'] in c.get('batch_sizes',(1,2,4,8,16,31,32,33,64)) and position in (('first',) if r['batch_size']==1 else ('first','last')),'Unprescribed arrangement')
            require(r['dtype'].removeprefix('torch.')==c['dtype'].removeprefix('torch.'),'Measured input dtype differs')
            require(r['seed'] in range(100) and r['generator_seed']==1000000+r['seed'],'Wrong seed protocol')
            require(r['tf32_matmul'] is False and r['tf32_cudnn'] is False and r['cudnn_benchmark'] is False,'Changed precision/algorithm protocol')
            h=r['qualifier_source_sha256']
            if h not in archives:
                path=a.qualifier_sources/(h+'.py');require(sha(path)==h,'Qualifier source archive differs');archives.add(h)
            check_operands(r,c)
            expected_dtype='torch.'+c.get('output_dtype',c['dtype']).removeprefix('torch.')
            for field in ('native_batch_invariance','invariant_batch_invariance','singleton_fidelity','reference_batch_invariance','reference_singleton_fidelity'):
                check_metric(r[field]);require(all(t['dtype']==expected_dtype for t in r[field]['tensors'].values()),'Measured output precision differs from source')
            require(r['invariant_batch_invariance']['exact'] and r['reference_batch_invariance']['exact'],'Invariant or explicit-reference batch check failed')
            if r['seed']<3:
                require(r['fp64_reference'] is not None,'Missing FP64 reference')
                for m in r['fp64_reference'].values():
                    check_metric(m);require(all(t['dtype']=='torch.float64' for t in m['tensors'].values()),'False FP64 reference')
            if r['seed']==0:
                profile=r['dispatch_profile'];require(profile and profile['cuda_kernels'],'Missing CUDA dispatch evidence')
                expected={'conv2d':'aten::convolution','mean':'aten::mean','softmax':'aten::_softmax'}[c['operation']]
                require(expected in profile['aten_events'],'Wrong profiled operation')
            rows[key].append(r)
    summaries=[]
    for key,items in sorted(rows.items()):
        seeds=[r['seed'] for r in items];require(len(seeds)==len(set(seeds)),'Duplicate seed')
        summaries.append(dict(arrangement=key,measured_seeds=len(seeds),complete100seeds=set(seeds)==set(range(100)),
            native_nonexact_seeds=sum(not r['native_batch_invariance']['exact'] for r in items),
            singleton_fidelity_max_absolute_error=max(r['singleton_fidelity']['max_absolute_error'] for r in items)))
    expected={f'{c["id"]}|B{b}|{position}' for c in configurations.values() for b in c.get('batch_sizes',(1,2,4,8,16,31,32,33,64))
        for position in (('first',) if b==1 else ('first','last'))}
    complete={r['arrangement'] for r in summaries if r['complete100seeds']};all_complete=complete==expected
    require(not a.require_complete or all_complete,'Selected source sweep incomplete')
    report=dict(status='selected_source_operator_measurements_independently_verified',configuration_sha256=identity,
        raw_records=records,raw_records_sha256=raw_hash.hexdigest(),auditor_sha256=sha(Path(__file__)),
        arrangements=summaries,completed_arrangements=len(complete),prescribed_arrangements=len(expected),
        all_selected_arrangements_complete=all_complete,full_model_operator_qualification_complete=False,
        timing_not_qualified=True,scope=config.get('scope',config.get('coverage_limit')))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Audit all observed PI05 operators, separating source layouts from diagnostics.

The original flattened-head BMM tests and TF32-enabled convolution measurements
remain supplemental. Actual matmul source layouts and TF32-disabled convolution
must independently complete before coverage can pass.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

SIZES=(1,2,4,8,16,31,32,33,64)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def key(operation,arguments):return json.dumps([operation,arguments],sort_keys=True)


def contiguous(shape):
    result=[1]*len(shape)
    for i in range(len(shape)-2,-1,-1):result[i]=result[i+1]*shape[i+1]
    return result


def check_metric(metric):
    require(isinstance(metric.get('exact'),bool),'Missing exactness measurement')
    require(metric['nonfinite_count']==0,'Nonfinite operator measurement')
    require(metric['exact']==(metric['unequal_element_count']==0 and metric['nonfinite_count']==0),'Inconsistent byte equality metadata')
    for name in ('max_absolute_error','mean_absolute_error','rmse','relative_l2_error'):
        require(math.isfinite(metric[name]) and metric[name]>=0,'Invalid numerical error')
    tensors=metric.get('tensors',{})
    require(tensors and sum(t['unequal_element_count'] for t in tensors.values())==metric['unequal_element_count'],'Tensor mismatch counts differ')
    for tensor in tensors.values():
        for field in ('reference_sha256','candidate_sha256'):
            require(isinstance(tensor.get(field),str) and len(tensor[field])==64,'Missing original-dtype byte fingerprint')
        if metric['exact']:require(tensor['reference_sha256']==tensor['candidate_sha256'] and tensor['exact'],'False exactness claim')
        require(tensor['nonfinite_count']==0 and tensor['exact']==(tensor['unequal_element_count']==0),'Inconsistent tensor exactness/count')
        require((tensor['reference_sha256']==tensor['candidate_sha256'])==(tensor['unequal_element_count']==0),'Byte fingerprints contradict mismatch count')


def check_source_config(c,entry):
    args=entry['arguments'];x=args[0];op=c['operation']
    require(c['dtype']==x['dtype'] and c['source_observed_count']==entry['count'],'Source dtype/call count differs')
    if op in ('mm','addmm'):
        require(entry['aten_operation']=='aten::linear' and c['m_per_request']==math.prod(x['shape'][:-1])
                and c['k']==x['shape'][-1]==args[1]['shape'][1] and c['n']==args[1]['shape'][0]
                and (op=='addmm')==(len(args)>2 and args[2] is not None),'Linear dimensions/bias differ from source')
        require(c['layout']=='right_transposed' and x['stride']==contiguous(x['shape']) and args[1]['stride']==contiguous(args[1]['shape']),'Unqualified source linear layout')
    elif op=='matmul':
        require(entry['aten_operation']=='aten::matmul' and c['layout']=='source_strided_matmul' and c['m_per_request']==1,'Source matmul collapsed into shape-only diagnostic')
        require(c['source_left_shape']==x['shape'] and c['source_right_shape']==args[1]['shape']
                and c['source_left_stride_b1']==x['stride'] and c['source_right_stride_b1']==args[1]['stride'],'Source matmul ranks/strides differ')
    elif op in ('mean','softmax'):
        require(entry['aten_operation']==f'aten::{op}' and c['m_per_request']==1 and c['input_per_unit']==x['shape'][1:]
                and c['dim']==-1 and c['layout']=='contiguous' and x['stride']==contiguous(x['shape']),'Reduction source differs')
        if op=='mean':require(args[1]==[-1] and c['keepdim']==args[2],'Mean dimension/keepdim differs')
        else:require(args[1]==-1 and args[2]==x['dtype'],'Softmax dimension/dtype differs')
    elif op=='conv2d':
        require(entry['aten_operation']=='aten::conv2d' and c['m_per_request']==1 and c['input_per_image']==x['shape'][1:]
                and c['weight_shape']==args[1]['shape'] and c['stride']==args[3] and c['padding']==[0,0]
                and c['dilation']==[1,1] and c['groups']==1 and c['has_bias'] and c['layout']=='contiguous','Convolution source differs')
    elif op=='sdpa':
        q,k,v=args[:3]
        require(entry['aten_operation']=='aten::scaled_dot_product_attention' and c['m_per_request']==1
                and [c['query_heads'],c['query_length'],c['head_dim']]==q['shape'][1:]
                and [c['key_heads'],c['key_length'],c['head_dim']]==k['shape'][1:]
                and [c['value_heads'],c['key_length'],c['value_dim']]==v['shape'][1:]
                and c['source_q_stride']==q['stride'] and c['source_k_stride']==k['stride'] and c['source_v_stride']==v['stride']
                and c['scale']==entry['keyword_arguments']['scale'],'SDPA source shape/scale/strides differ')
    else:raise ValueError(f'Unsupported primary qualification {op}')


def check_operands(row,c):
    b=row['batch_size'];op=c['operation'];m=b*c['m_per_request'];layout=c['layout']
    if op in ('mm','addmm'):
        left=[m,c['k']];right=[c['k'],c['n']]
        ls=[c['k']*2,2] if layout=='left_strided' else [c['k'],1]
        rs=[1,c['k']] if layout=='right_transposed' else [c['n'],1]
    elif op=='bmm':
        left=[m,c['m'],c['k']];right=[m,c['k'],c['n']]
        ls=[c['m']*c['k']*2,c['k']*2,2] if layout=='left_strided' else contiguous(left)
        rs=[c['n']*c['k'],1,c['k']] if layout=='right_transposed' else contiguous(right)
    elif op=='matmul':
        left=[b,*c['source_left_shape'][1:]];right=[b,*c['source_right_shape'][1:]]
        ls=c['source_left_stride_b1' if b==1 else 'source_left_stride_b8'];rs=c['source_right_stride_b1' if b==1 else 'source_right_stride_b8']
    elif op in ('mean','softmax'):
        left=[m,*c['input_per_unit']];right=None;ls=contiguous(left);rs=None
    elif op=='conv2d':
        left=[m,*c['input_per_image']];right=c['weight_shape'];ls=contiguous(left);rs=contiguous(right)
    elif op=='sdpa':
        left=[m,c['query_heads'],c['query_length'],c['head_dim']];right=[m,c['key_heads'],c['key_length'],c['head_dim']]
        ls=c['source_q_stride'];rs=c['source_k_stride']
        require(row['value_shape']==[m,c['value_heads'],c['key_length'],c['value_dim']] and row['value_stride']==c['source_v_stride'],'SDPA value source differs')
    else:raise ValueError(op)
    require(row['left_shape']==left and row['right_shape']==right and row['left_stride']==ls and row['right_stride']==rs,'Measured operand shape/stride differs')


def audit_group(directory,config_name,raw_name,source,configuration,primary,new_runtime):
    path=directory/config_name;raw=directory/raw_name
    cdata=json.loads(path.read_text());require(cdata['adapter_execution_configuration']==configuration,'Operator/model configuration differs')
    for evidence in cdata['source_evidence']:require(sha(Path(evidence['path']))==evidence['sha256'],'Source trace changed')
    trace_pair=[json.loads(Path(e['path']).read_text()) for e in cdata['source_evidence']]
    require([t['batch_size'] for t in trace_pair]==[1,8],'Missing B1/B8 source traces')
    require(len(trace_pair[0]['operations'])==len(trace_pair[1]['operations']),'B1/B8 operation coverage differs')
    for small,large in zip(*(t['operations'] for t in trace_pair)):
        require(small['aten_operation']==large['aten_operation'] and small['count']==large['count'] and small['keyword_arguments']==large['keyword_arguments'],'B1/B8 operator semantics differ')
        for i,(a,b) in enumerate(zip(small['arguments'],large['arguments'])):
            if isinstance(a,dict) and 'shape' in a:
                batched=i==0 or small['aten_operation']=='aten::matmul' or (small['aten_operation']=='aten::scaled_dot_product_attention' and i<3)
                wanted_shape=[8*a['shape'][0],*a['shape'][1:]] if batched else a['shape']
                require(b['shape']==wanted_shape and a['dtype']==b['dtype'],'B1/B8 tensor scaling/dtype differs')
            else:require(a==b,'B1/B8 scalar arguments differ')
    configs={c['id']:c for c in cdata['configurations']};require(len(configs)==len(cdata['configurations']),'Duplicate configuration IDs')
    wanted={f'{identifier}|B{b}|{position}' for identifier in configs for b in SIZES for position in (('first',) if b==1 else ('first','last'))}
    seeds=defaultdict(set);profiles=set();count=0;native_failures=0
    if raw.exists():
        for line in raw.open():
            row=json.loads(line);name=row['operator_configuration'];identifier,batch,position=name.split('|');c=configs[identifier]
            require(name in wanted and row['batch_size']==int(batch[1:]) and row['target_position']==position,'Unplanned arrangement')
            seed=row['seed'];require(isinstance(seed,int) and 0<=seed<100 and seed not in seeds[name] and row['generator_seed']==1000000+seed,'Missing/duplicate/wrong seed')
            seeds[name].add(seed);count+=1
            require(row['configuration_sha256']==sha(path) and row['source_operation']==c['source_operation'] and row['layout']==c['layout'] and row['dtype']==c['dtype'],'Raw source/configuration identity differs')
            check_operands(row,c)
            expected_dispatch='aten::convolution' if c['operation']=='conv2d' else 'aten::scaled_dot_product_attention' if c['operation']=='sdpa' else 'aten::_softmax' if c['operation']=='softmax' else f'aten::{c["operation"]}'
            require(row['dispatched_aten_operation']==expected_dispatch,'Wrong measured ATen operation')
            for field in ('native_batch_invariance','invariant_batch_invariance','singleton_fidelity','reference_batch_invariance','reference_singleton_fidelity'):check_metric(row[field])
            require(row['invariant_batch_invariance']['exact'] and row['invariant_batch_invariance']['nonfinite_count']==0,'Invariant operator qualification failed')
            native_failures+=not row['native_batch_invariance']['exact']
            for field in ('native_latency_seconds','invariant_latency_seconds','reference_latency_seconds'):
                require(math.isfinite(row[field]) and row[field]>=0,'Invalid timing evidence')
            if seed<3:
                require(row['fp64_reference'] is not None,'Missing selected FP64 reference')
                for metric in row['fp64_reference'].values():check_metric(metric)
            if seed==0:
                profile=row['dispatch_profile'];require(profile and expected_dispatch in profile['aten_events'] and profile['cuda_kernels'],'Missing measured CUDA dispatch')
                profiles.add(name)
            if new_runtime:
                require(row['tf32_matmul'] is False and row['tf32_cudnn'] is False and row['cudnn_benchmark'] is False,'Precision does not match model')
                require(row['gpu'] and row['gpu_uuid']!='unavailable' and row['torch'],'Missing runtime identity')
                helper=directory/'qualifier_sources'/f'{row["qualifier_source_sha256"]}.py'
                require(helper.is_file() and sha(helper)==row['qualifier_source_sha256'],'Measured qualifier source is not retained')
    complete_ids=[];covered=set()
    for identifier,c in configs.items():
        done=all(seeds[name]==set(range(100)) and name in profiles for name in wanted if name.startswith(identifier+'|'))
        if done:complete_ids.append(identifier)
        if primary(c):
            source_key=key(c['source_operation'],c['source_observed_arguments'])
            require(source_key in source,'Qualification has no observed source operation')
            check_source_config(c,source[source_key])
            if c['operation']=='matmul':
                large=trace_pair[1]['operations'][c['source_trace_index']]
                require(large['aten_operation']=='aten::matmul' and large['arguments'][0]['shape']==[8,*c['source_left_shape'][1:]]
                        and large['arguments'][1]['shape']==[8,*c['source_right_shape'][1:]]
                        and c['source_left_stride_b8']==large['arguments'][0]['stride']
                        and c['source_right_stride_b8']==large['arguments'][1]['stride'],'Expanded source matmul layout differs')
            if done:covered.add(source_key)
    return dict(configuration_file=str(path),configuration_sha256=sha(path),raw_file=str(raw),raw_sha256=sha(raw) if raw.exists() else None,
        records=count,complete_configurations=complete_ids,total_configurations=len(configs),native_failed_measurements=native_failures,
        all_invariant_measurements_exact=True,complete=len(complete_ids)==len(configs)),covered


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--base-manifest',type=Path,required=True)
    p.add_argument('--expanded-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args();d=args.directory
    expanded=[json.loads((d/f'aten_all_trace_v2_B{b}.json').read_text()) for b in (1,8)]
    original=[json.loads((d/f'aten_complete_trace_B{b}.json').read_text()) for b in (1,8)]
    config=expanded[0]['adapter_execution_configuration']
    require(config['checkpoint_revision']=='7152f13e34c6c115d808aea0e5d08aca74eac14b' and config['tf32'] is False,'Unexpected checkpoint/precision')
    for traces,manifest_path in ((expanded,args.expanded_manifest),(original,args.base_manifest)):
        m=json.loads(manifest_path.read_text());allowed={request for row in m['observations'] if row['split']=='diagnostic' for request in row['request_ids']}
        for b,trace in zip((1,8),traces):
            require(trace['batch_size']==b and len(trace['request_ids'])==b and all(i in allowed for i in trace['request_ids']),'Operator source inputs are not diagnostic')
            require(trace['replay_manifest_sha256']==sha(manifest_path) and trace['adapter_execution_configuration']==config,'Trace binding/configuration differs')
    for old,new in zip(original,expanded):
        require(old['request_ids']==new['request_ids'],'Expanded trace changed selected diagnostic requests')
        filtered=[entry for entry in new['operations'] if entry['aten_operation']!='aten::softmax']
        require(old['operations']==filtered,'Previously observed operators changed')
    source={key(entry['aten_operation'],entry['arguments']):entry for entry in expanded[0]['operations']}
    require(len(source)==29,'Expected all29 distinct observed source patterns')
    groups=[];covered=set()
    for cfile,raw,primary,new_runtime in (
        ('actual_shape_config.json','actual_shapes_100seeds.jsonl',lambda c:c['operation'] not in ('bmm','conv2d'),False),
        ('actual_precision_supplement_config.json','actual_precision_supplement_100seeds.jsonl',lambda c:True,True),
        ('actual_matmul_source_layout_config.json','actual_matmul_source_layout_100seeds.jsonl',lambda c:True,True)):
        report,keys=audit_group(d,cfile,raw,source,config,primary,new_runtime);groups.append(report);covered.update(keys)
    complete=covered==set(source)
    report=dict(status='actual_operator_coverage_independently_verified' if complete else 'operator_qualification_incomplete',complete=complete,
        observed_source_patterns=29,completed_source_patterns=len(covered),groups=groups,
        excluded_from_primary_coverage=['Legacy flattened-head BMM layouts','Legacy TF32True native convolution'],
        legacy_precision_context=json.loads((d/'original_precision_context.json').read_text()),
        model_performance_qualified=False,full_campaign_complete=False,auditor_sha256=sha(Path(__file__)))
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not complete and not args.allow_incomplete:raise SystemExit(1)


if __name__=='__main__':main()

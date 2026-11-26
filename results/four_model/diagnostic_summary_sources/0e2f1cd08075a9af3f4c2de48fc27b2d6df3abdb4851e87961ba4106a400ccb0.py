"""Summarize fixed-input singleton/batch errors from complete diagnostic pairs.

Mean squared singleton/batch difference is a paired deviation measure, not a
centered sample variance. Output and each denoising stage retain separate units.
Incomplete trailing requests in a live gzip stream are excluded explicitly.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_arrangements(sizes):
    keys={('singleton',1,'first','none')}
    for size in sizes:
        if size==1:continue
        for companions in ('duplicate','diverse'):
            for position in (('first','last') if size==2 else ('first','middle','last')):
                keys.add(('batch_position',size,position,companions))
        keys.add(('permutation',size,'last','diverse'))
        keys.add(('partition',size,'first','diverse'))
    return keys


def accumulator():
    return dict(comparisons=0,nonexact_comparisons=0,elements=0,different_elements=0,
        squared_difference_sum=0.,max_absolute_difference=0.,dtypes=set())


def add(bucket,tensor):
    require(tensor['nonfinite_count']==0,'Nonfinite diagnostic tensor')
    count=math.prod(tensor['shape']);error=tensor['rmse'];maximum=tensor['max_absolute_error']
    require(count>0 and math.isfinite(error) and math.isfinite(maximum) and error>=0 and maximum>=0,'Invalid diagnostic error')
    different=tensor['unequal_element_count']
    require(0<=different<=count and tensor['exact']==(different==0),'Inconsistent diagnostic exactness')
    require((tensor['reference_sha256']==tensor['candidate_sha256'])==tensor['exact'],'Byte hashes contradict exactness')
    bucket['comparisons']+=1;bucket['nonexact_comparisons']+=not tensor['exact'];bucket['elements']+=count
    bucket['different_elements']+=different;bucket['squared_difference_sum']+=error*error*count
    bucket['max_absolute_difference']=max(bucket['max_absolute_difference'],maximum);bucket['dtypes'].add(tensor['dtype'])


def table(buckets,fields):
    rows=[]
    for key,bucket in sorted(buckets.items()):
        squared=bucket['squared_difference_sum']/bucket['elements']
        rows.append(dict(zip(fields,key),**{k:v for k,v in bucket.items() if k not in ('dtypes','squared_difference_sum')},
            dtypes=sorted(bucket['dtypes']),mean_squared_singleton_batch_difference=squared,
            rms_singleton_batch_difference=math.sqrt(squared)))
    return rows


def summarize(path):
    runtime_path=Path(str(path)+'.runtime.json');runtime=json.loads(runtime_path.read_text())
    require(runtime['split']=='diagnostic','This report accepts diagnostic data only')
    expected=expected_arrangements(runtime['batch_sizes']);modes=set(runtime['modes'])
    required={(mode,*key) for mode in modes for key in expected}
    normalized=defaultdict(accumulator);arrangements=defaultdict(accumulator);flow=defaultdict(accumulator)
    included=hashlib.sha256();completed=[];seen=set();excluded=[]
    def consume(items):
        if not items:return
        request=items[0][0]['request_id']
        keys=[(r['implementation'],r['transformation'],r['logical_batch_size'],r['target_batch_position'],r['companion_type']) for r,_ in items]
        require(len(keys)==len(set(keys)),'Duplicate diagnostic arrangement')
        require(set(keys).issubset(required),'Unexpected diagnostic arrangement')
        if set(keys)!=required:excluded.append(dict(request_id=request,records=len(items),required=len(required)));return
        require(request not in seen,'Noncontiguous/duplicate completed request');seen.add(request)
        task=items[0][0]['task'];noise=items[0][0]['request_noise_sha256']
        for r,line in items:
            require(r['request_id']==request and r['task']==task and r['request_noise_sha256']==noise,'Diagnostic request/input/noise identity differs')
            require(r['adapter']==runtime['adapter'] and r['replay_manifest_sha256']==runtime['replay_manifest_sha256'],'Foreign diagnostic source')
            require(r['adapter_execution_configuration']==runtime['execution_configuration'],'Mixed inference configuration')
            require(r['tf32_matmul']==runtime['tf32_matmul'] and r['tf32_cudnn']==runtime['tf32_cudnn'],'Mixed precision protocol')
            included.update(line.encode());mode=r['implementation'];logical=r['logical_batch_size'];physical=r['batch_size']
            output=r['output']['tensors']['output.normalized']
            if r['transformation'] in ('singleton','batch_position'):
                add(normalized[(str(task),mode,physical)],output)
            add(arrangements[(mode,r['transformation'],logical,physical,r['companion_type'],r['target_batch_position'])],output)
            for stage,metric in r['flow_steps'].items():
                if stage.startswith('flow.step_') and r['transformation'] in ('singleton','batch_position'):
                    require(set(metric['tensors'])=={'output'},'Unexpected denoising tensor schema')
                    add(flow[(str(task),mode,physical,stage)],metric['tensors']['output'])
        completed.append(dict(request_id=request,task=task,noise_sha256=noise))
    current=[];current_id=None;stream_complete=True
    try:
        with gzip.open(path,'rt') as stream:
            for line in stream:
                # A live member can end inside a JSON line; it is uncommitted.
                if not line.endswith('\n'):stream_complete=False;break
                r=json.loads(line)
                if current_id is not None and r['request_id']!=current_id:consume(current);current=[]
                current.append((r,line));current_id=r['request_id']
    except EOFError:stream_complete=False
    consume(current)
    return dict(source=str(path.resolve()),runtime_snapshot=runtime,runtime_sha256=sha(runtime_path),
        included_json_records_sha256=included.hexdigest(),included_complete_request_noise_pairs=len(completed),
        included_request_noise_pairs=completed,excluded_partial_requests=excluded,gzip_clean_at_observation=stream_complete,
        normalized_by_task_batch=table(normalized,('task','implementation','batch_size')),
        normalized_by_batch_arrangement=table(arrangements,('implementation','transformation','logical_batch_size','physical_batch_size','companion_type','target_position')),
        denoising_by_task_batch_stage=table(flow,('task','implementation','batch_size','stage')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',nargs=2,action='append',required=True,metavar=('MODEL','RECORDS'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=dict(primary_endpoint='fixed_input_singleton_vs_batched_stability',scope='complete_diagnostic_request_noise_pairs_only',
        provisional=True,full_campaign_complete=False,source_sha256=sha(Path(__file__)),
        metric_definition='Element-weighted mean squared deviation from the same-implementation singleton; not centered sample variance. No mixing physical action units.',studies=[])
    for model,path in args.study:
        result=summarize(Path(path));result['model']=model;report['studies'].append(result)
        print(f'{model}: {result["included_complete_request_noise_pairs"]} complete request/noise pairs',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(report,indent=2)+'\n').encode();temporary=args.output.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(args.output)
    archive=args.output.parent/'diagnostic_summary_snapshots';archive.mkdir(exist_ok=True)
    (archive/(hashlib.sha256(payload).hexdigest()+'.json')).write_bytes(payload)


if __name__=='__main__':main()

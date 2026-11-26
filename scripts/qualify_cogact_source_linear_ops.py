"""Qualify actual CogACT linear shapes, strides and logical request axes.

Measures 100 seeds per selected first/last arrangement. The full prescribed
sweep is B1,2,4,8,16,31,32,33,64; an arrangement filter makes partial coverage
explicit. Shared-GPU timings are retained but never qualified performance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from batch_invariant_ops import set_batch_invariant_mode
from scripts.run_model_replay_invariance import metrics
from scripts.qualify_real_shape_matrix_ops import time_call

SIZES=(1,2,4,8,16,31,32,33,64)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def input_layout(config,batch_size):
    a=config['source']['b1']['arguments'][0];b=config['source']['b8']['arguments'][0]
    shape=list(a['shape']);axis=config['batch_axis']
    if axis is not None:shape[axis]*=batch_size
    expected=list(a['shape'])
    if axis is not None:expected[axis]*=8
    require(expected==b['shape'],'Source logical batch axis differs')
    stride=[]
    for s,t in zip(a['stride'],b['stride']):
        require(s>=0 and (t==s or t==8*s),'Uncovered source stride rule')
        stride.append(s if t==s else s*batch_size)
    return shape,stride


def allocate(shape,stride,dtype,device,generator=None):
    require(len(shape)==len(stride) and all(n>0 for n in shape) and all(s>=0 for s in stride),'Invalid source layout')
    storage=torch.empty(1+sum((n-1)*s for n,s in zip(shape,stride)),dtype=dtype,device=device)
    if generator is not None:storage.normal_(generator=generator)
    view=storage.as_strided(shape,stride)
    require(torch._debug_has_internal_overlap(view)!=1,'Overlapping source linear input')
    return view


def operands(config,batch_size,seed,device='cuda'):
    args=config['source']['b1']['arguments'];dtype=getattr(torch,args[0]['dtype'].removeprefix('torch.'))
    generator=torch.Generator(device=device).manual_seed(seed)
    shape,stride=input_layout(config,batch_size);left=allocate(shape,stride,dtype,device,generator)
    weight=allocate(args[1]['shape'],args[1]['stride'],dtype,device,generator)
    bias=allocate(args[2]['shape'],args[2]['stride'],dtype,device,generator) if len(args)>2 and args[2] is not None else None
    return left,weight,bias


def select_request(config,value,position):
    axis=config['batch_axis'];groups=config['groups_per_request']
    if axis is None:return value
    if config.get('coordinate_layout')=='cfg_split_branches':
        require(axis==0 and groups==2 and value.shape[0]%2==0,'Invalid CFG layout')
        batch=value.shape[0]//2
        return torch.index_select(value,0,torch.tensor([position,batch+position],device=value.device))
    return value.narrow(axis,position*groups,groups)


def singleton(config,left,position):
    axis=config['batch_axis'];groups=config['groups_per_request']
    selected=select_request(config,left,position)
    shape,stride=input_layout(config,1)
    target=allocate(shape,stride,left.dtype,left.device)
    target.copy_(selected);return target


def call(left,weight,bias,invariant=False):
    with torch.inference_mode(),set_batch_invariant_mode(invariant):return F.linear(left,weight,bias)


def explicit_reference(config,left,weight,bias,batch_size):
    if config['batch_axis'] is None:return call(singleton(config,left,0),weight,bias)
    outputs=[call(singleton(config,left,i),weight,bias) for i in range(batch_size)]
    if config.get('coordinate_layout')=='cfg_split_branches':
        return torch.cat([torch.cat([v[branch:branch+1] for v in outputs],dim=0) for branch in (0,1)],dim=0)
    return torch.cat(outputs,dim=config['batch_axis'])


def profile(left,weight,bias):
    for attempt in range(1,4):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
            call(left,weight,bias);call(left,weight,bias,True);torch.cuda.synchronize()
        aten=sorted({e.key for e in trace.key_averages() if e.key.startswith('aten::')})
        kernels=sorted({e.name for e in trace.events() if getattr(getattr(e,'device_type',None),'name','')=='CUDA'})
        require('aten::linear' in aten,'High-level linear was not exercised')
        if kernels:return dict(aten_events=aten,cuda_kernels=kernels,capture_attempt=attempt)
    raise RuntimeError('CUPTI captured no CUDA kernels after three attempts')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arrangement-keys',nargs='+');p.add_argument('--resume',action='store_true');a=p.parse_args()
    require(not a.output.exists() or a.resume,'Output exists; explicit resume required')
    config=json.loads(a.config.read_text());config_sha=sha(a.config)
    require(config['adapter']=='scripts.model_invariance.adapters.cogact_simpler_exact','Foreign source adapter')
    for path,expected in config['traces'].items():require(sha(Path(path))==expected,'Source trace changed')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    files=[Path(__file__),Path(__file__).with_name('run_model_replay_invariance.py'),Path(__file__).with_name('qualify_real_shape_matrix_ops.py')]
    files+=list((Path(__file__).resolve().parents[1]/'batch_invariant_ops').rglob('*.py'))
    sources={str(path.resolve()):sha(path) for path in files};archive=a.output.parent/'source_linear_qualifier_sources';archive.mkdir(parents=True,exist_ok=True)
    for path,identity in sources.items():
        saved=archive/identity
        if not saved.exists():saved.write_bytes(Path(path).read_bytes())
        require(sha(saved)==identity,'Archived qualifier source differs')
    valid={f'{c["id"]}|B{b}|{pos}' for c in config['configurations'] for b in SIZES for pos in (('first',) if b==1 else ('first','last'))}
    if a.arrangement_keys:require(set(a.arrangement_keys).issubset(valid),'Unknown arrangement key')
    done=set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            r=json.loads(line);require(r['configuration_sha256']==config_sha and r['source_sha256']==sources,'Cannot mix source/config versions')
            key=(r['arrangement'],r['seed']);require(key not in done,'Duplicate raw record');done.add(key)
    gpu=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    for c in config['configurations']:
        for batch in SIZES:
            for position in (('first',) if batch==1 else ('first','last')):
                key=f'{c["id"]}|B{batch}|{position}'
                if a.arrangement_keys and key not in a.arrangement_keys:continue
                for seed in range(100):
                    if (key,seed) in done:continue
                    require(all(sha(Path(path))==identity for path,identity in sources.items()),'Source changed during qualification')
                    left,weight,bias=operands(c,batch,1000000+seed);index=0 if position=='first' else batch-1
                    target=singleton(c,left,index);axis=c['batch_axis'];groups=c['groups_per_request']
                    native_one=call(target,weight,bias);native_batch=call(left,weight,bias)
                    invariant_one=call(target,weight,bias,True);invariant_batch=call(left,weight,bias,True)
                    select=lambda value:select_request(c,value,index)
                    reference=explicit_reference(c,left,weight,bias,batch)
                    comparisons=dict(native_batch_invariance=metrics(native_one,select(native_batch)),
                        invariant_batch_invariance=metrics(invariant_one,select(invariant_batch)),
                        singleton_fidelity=metrics(native_one,invariant_one),reference_batch_invariance=metrics(native_one,select(reference)))
                    double=None
                    if seed<3:
                        expected=F.linear(target.double(),weight.double(),bias.double() if bias is not None else None)
                        double=dict(native=metrics(expected,native_one.double()),invariant=metrics(expected,invariant_one.double()))
                    record=dict(schema_version=1,arrangement=key,seed=seed,generator_seed=1000000+seed,
                        configuration_sha256=config_sha,source_sha256=sources,gpu_uuid=gpu,torch=torch.__version__,
                        batch_size=batch,target_position=position,batch_axis=axis,groups_per_request=groups,
                        coordinate_layout=c['coordinate_layout'],target_axis_indices=([index,batch+index] if c['coordinate_layout']=='cfg_split_branches' else None),
                        left_shape=list(left.shape),left_stride=list(left.stride()),target_shape=list(target.shape),target_stride=list(target.stride()),
                        output_shape=list(invariant_batch.shape),output_dtype=str(invariant_batch.dtype),weight_shape=list(weight.shape),weight_stride=list(weight.stride()),
                        fp64_reference=double,dispatch_profile=profile(left,weight,bias) if seed==0 else None,
                        native_latency_seconds=time_call(lambda:call(left,weight,bias)),invariant_latency_seconds=time_call(lambda:call(left,weight,bias,True)),
                        shared_gpu_timing_not_qualified=True,tf32_matmul=False,**comparisons)
                    with a.output.open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
                    require(comparisons['invariant_batch_invariance']['exact'],'Actual source linear violates batch invariance')
                    done.add((key,seed))
                print(f'qualified {key}: 100 seeds',flush=True)


if __name__=='__main__':main()

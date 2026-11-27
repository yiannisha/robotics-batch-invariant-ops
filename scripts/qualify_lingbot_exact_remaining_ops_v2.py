"""Corrected-adapter component protocol; legacy source remains unchanged.

Measure LingBot shared pooling and rotary matmul with captured layouts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from batch_invariant_ops import set_batch_invariant_mode
from scripts.run_model_replay_invariance import metrics

SIZES = (1,2,4,8,16,31,32,33,64)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layout(case, operand, batch):
    a = case['source_entries']['b1']['arguments'][operand]
    b = case['source_entries']['b8']['arguments'][operand]
    shape = list(a['shape']); axis = case['batch_axis']
    if axis is not None:
        shape[axis] *= batch
    stride = []
    for i,(s,t) in enumerate(zip(a['stride'],b['stride'])):
        if s == t:
            stride.append(s)
        elif i == axis and t == 0 and a['shape'][i] == 1:
            stride.append(s if batch == 1 else 0)
        elif t == 8*s:
            stride.append(s*batch)
        else:
            raise ValueError('Unsupported source stride scaling')
    return shape,stride


def allocate(shape,stride,dtype,device,generator=None):
    storage = torch.empty(1+sum((n-1)*s for n,s in zip(shape,stride)),dtype=dtype,device=device)
    if generator is not None:
        storage.normal_(generator=generator)
    return storage.as_strided(shape,stride)


def select(case,value,index):
    return value if case['batch_axis'] is None else value.narrow(case['batch_axis'],index,1)


def target(case,values,index):
    result=[]
    for i,value in enumerate(values):
        shape,stride=layout(case,i,1)
        output=allocate(shape,stride,value.dtype,value.device)
        selected=select(case,value,index)
        # Copy each physical value once before exposing the captured broadcasts.
        physical=tuple(slice(0,1) if s==0 else slice(None) for s in stride)
        output[physical].copy_(selected[physical]);result.append(output)
    return result


def call(case,values,invariant=False):
    with torch.inference_mode(),set_batch_invariant_mode(invariant):
        if case['operation']=='aten::matmul':
            return torch.matmul(*values)
        args=case['source_entries']['b1']['arguments']
        return torch.mean(values[0],dim=args[1],keepdim=args[2] if len(args)>2 else False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arrangement-keys',nargs='+');p.add_argument('--resume',action='store_true')
    a=p.parse_args();config=json.loads(a.config.read_text());identity=sha(a.config)
    if config['adapter']!='scripts.model_invariance.adapters.lingbot_robotwin_exact_v2':
        raise ValueError('Foreign adapter')
    for path,h in config['traces'].items():
        if sha(Path(path))!=h:raise ValueError('Source trace changed')
    if a.output.exists() and not a.resume:raise FileExistsError(a.output)
    files=[Path(__file__),Path(__file__).with_name('run_model_replay_invariance.py')]
    files+=list((Path(__file__).resolve().parents[1]/'batch_invariant_ops').rglob('*.py'))
    sources={str(x.resolve()):sha(x) for x in files}
    archive=a.output.parent/'remaining_qualifier_sources';archive.mkdir(exist_ok=True)
    for path,h in sources.items():
        dest=archive/h
        if not dest.exists():dest.write_bytes(Path(path).read_bytes())
        if sha(dest)!=h:raise ValueError('Source archive differs')
    done=set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            row=json.loads(line)
            if row['configuration_sha256']!=identity or row['source_sha256']!=sources:raise ValueError('Mixed source/configuration')
            key=(row['arrangement'],row['seed'])
            if key in done:raise ValueError('Duplicate seed')
            done.add(key)
    valid={f'{c["id"]}|B{b}|{pos}' for c in config['configurations'] for b in SIZES for pos in (('first',) if b==1 else ('first','last'))}
    if a.arrangement_keys and not set(a.arrangement_keys)<=valid:raise ValueError('Unknown arrangement')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for c in config['configurations']:
        for b in SIZES:
            for pos in (('first',) if b==1 else ('first','last')):
                key=f'{c["id"]}|B{b}|{pos}'
                if a.arrangement_keys and key not in a.arrangement_keys:continue
                for seed in range(100):
                    if (key,seed) in done:continue
                    if any(sha(Path(path))!=h for path,h in sources.items()):raise ValueError('Source changed')
                    generator=torch.Generator(device='cuda').manual_seed(1000000+seed)
                    count=2 if c['operation']=='aten::matmul' else 1
                    values=[allocate(*layout(c,i,b),torch.float32,'cuda',generator) for i in range(count)]
                    index=0 if pos=='first' else b-1;one=target(c,values,index)
                    native_one=call(c,one);native_batch=call(c,values)
                    invariant_one=call(c,one,True);invariant_batch=call(c,values,True)
                    reference=call(c,one) if c['batch_axis'] is None else torch.cat([call(c,target(c,values,i)) for i in range(b)],dim=c['batch_axis'])
                    comparisons=dict(native_batch_invariance=metrics(native_one,select(c,native_batch,index)),
                        invariant_batch_invariance=metrics(invariant_one,select(c,invariant_batch,index)),
                        singleton_fidelity=metrics(native_one,invariant_one),reference_batch_invariance=metrics(native_one,select(c,reference,index)))
                    profile=None
                    if seed==0:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
                            call(c,values);call(c,values,True);torch.cuda.synchronize()
                        profile=dict(aten_events=sorted({e.key for e in trace.key_averages() if e.key.startswith('aten::')}),cuda_kernels=sorted({e.name for e in trace.events() if getattr(getattr(e,'device_type',None),'name','')=='CUDA'}))
                        if not profile['cuda_kernels']:raise RuntimeError('CUPTI captured no CUDA kernels')
                    double=None
                    if seed<3:
                        expected=call(c,[x.double() for x in one])
                        double=dict(native=metrics(expected,native_one.double()),invariant=metrics(expected,invariant_one.double()))
                    row=dict(configuration_sha256=identity,source_sha256=sources,arrangement=key,seed=seed,generator_seed=1000000+seed,
                        batch_size=b,target_position=pos,batch_axis=c['batch_axis'],input_shapes=[list(x.shape) for x in values],input_strides=[list(x.stride()) for x in values],
                        singleton_shapes=[list(x.shape) for x in one],singleton_strides=[list(x.stride()) for x in one],output_shape=list(invariant_batch.shape),output_dtype=str(invariant_batch.dtype),
                        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),torch=torch.__version__,tf32_matmul=False,performance_qualified=False,dispatch_profile=profile,fp64_reference=double,**comparisons)
                    with a.output.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
                    if not comparisons['invariant_batch_invariance']['exact'] or not comparisons['reference_batch_invariance']['exact']:raise RuntimeError('Invariant or request-coordinate reference failed')
                    done.add((key,seed))
                print(key+': 100 seeds',flush=True)


if __name__=='__main__':
    main()

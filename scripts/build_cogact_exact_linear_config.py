"""Retain the union of actual CogACT B1/B8 high-level linear layouts."""
import argparse
import hashlib
import json
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(one,eight):
    adapter='scripts.model_invariance.adapters.cogact_simpler_exact'
    require(one['adapter']==eight['adapter']==adapter,'Wrong CogACT adapter')
    require(one['batch_size']==1 and eight['batch_size']==8,'Need B1 and B8 traces')
    require(one['adapter_execution_configuration']==eight['adapter_execution_configuration'],'Different CogACT inference protocols')
    require(one['replay_manifest_sha256']==eight['replay_manifest_sha256'],'Different real-input manifests')
    a=[r for r in one['operations'] if r['aten_operation']=='aten::linear']
    b=[r for r in eight['operations'] if r['aten_operation']=='aten::linear']
    configs=[];used=set()
    for i,left in enumerate(a):
        matches=[]
        for j,right in enumerate(b):
            x=left['arguments'][0];y=right['arguments'][0]
            if left['arguments'][1:]!=right['arguments'][1:] or x['dtype']!=y['dtype']:continue
            if len(x['shape'])!=len(y['shape']):continue
            changed=[k for k,(p,q) in enumerate(zip(x['shape'],y['shape'])) if p!=q]
            if len(changed)>1 or changed and y['shape'][changed[0]]!=8*x['shape'][changed[0]]:continue
            axis=changed[0] if changed else None
            require(axis is None or axis<len(x['shape'])-1,'Linear features cannot be logical batch axis')
            require(all(s>=0 for s in x['stride']+y['stride']),'Negative source strides unsupported')
            require(all(t==s or t==8*s for s,t in zip(x['stride'],y['stride'])),'Nonlinear source stride rule requires explicit coverage')
            source=dict(b1=left,b8=right,b1_index=i,b8_index=j)
            signature=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:16]
            configs.append(dict(id=f'cogact_source_linear_axis{axis}_{signature}',operation='linear',batch_axis=axis,
                groups_per_request=x['shape'][axis] if axis is not None else None,source=source,coordinate_layout='cfg_split_branches' if axis==0 and x['shape'][0]==2 else 'contiguous_requests'))
            matches.append(j);used.add(j)
        require(matches,'Unmatched B1 linear source pattern')
    require(used==set(range(len(b))),'Unmatched B8 linear source pattern')
    # A single B1 shape can serve a shared global embedding and a batched
    # projection at B8. Keep both source-observed paths, rather than zipping.
    require(len({c['source']['b8_index'] for c in configs})==len(configs),'Ambiguous B8 source mapping')
    return dict(schema_version=1,adapter=adapter,adapter_execution_configuration=one['adapter_execution_configuration'],
        source_linear_patterns_b1=len(a),source_linear_patterns_b8=len(b),configurations=configs,
        layout_contract='Only source-observed constant or batch-linear strides; canonical singleton copies retain the source B1 strides.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--b1',type=Path,required=True);p.add_argument('--b8',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=build(json.loads(a.b1.read_text()),json.loads(a.b8.read_text()))
    result['traces']={str(path.resolve()):sha(path) for path in (a.b1,a.b8)};result['builder_source_sha256']=sha(Path(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(result["configurations"])} high-level linear layout configurations')


if __name__=='__main__':main()

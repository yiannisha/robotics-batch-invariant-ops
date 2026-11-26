"""Select actual LingBot exact-path request-scaled mean and softmax source cases."""
import argparse
import hashlib
import json
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('b1','b8','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();one=json.loads(a.b1.read_text());eight=json.loads(a.b8.read_text())
    require(not a.output.exists(),'Output already exists')
    require(one['adapter']==eight['adapter']=='scripts.model_invariance.adapters.lingbot_robotwin_exact','Wrong LingBot adapter')
    require(one['batch_size']==1 and eight['batch_size']==8,'Wrong source batch sizes')
    require(one['adapter_execution_configuration']==eight['adapter_execution_configuration'],'Different inference paths')
    require(one['adapter_execution_configuration']['language_tokens']==72 and one['adapter_execution_configuration']['tf32'] is False,'Wrong controlled protocol')
    cases=[];covered=set()
    for row in one['operations']:
        operation=row['aten_operation']
        if operation not in ('aten::softmax','aten::mean'):continue
        args=row['arguments'];x=args[0]
        if x['shape'][0] != 1:continue
        shape=[8,*x['shape'][1:]]
        matches=[r for r in eight['operations'] if r['aten_operation']==operation and r['arguments'][1:]==args[1:]
            and r['arguments'][0]['shape']==shape and r['arguments'][0]['stride']==x['stride'] and r['keyword_arguments']==row['keyword_arguments']]
        require(len(matches)==1,'Missing/ambiguous source batch counterpart');covered.add(json.dumps(matches[0],sort_keys=True))
        signature=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:12]
        c=dict(id='lingbot_exact_source_'+operation.split('::')[1]+'_'+signature,source_operation=operation,layout='contiguous',dtype=x['dtype'],
            m_per_request=1,source_observed_count=row['count'],source_entries={'b1':row,'b8':matches[0]})
        if operation=='aten::softmax':
            require(len(args)==2 and args[1]==-1 and not row['keyword_arguments'],'Uncovered softmax options')
            c.update(operation='softmax',input_per_unit=x['shape'][1:],dim=-1,output_dtype=x['dtype'])
        else:
            require(args[1] in ([-1],[1]) and not row['keyword_arguments'],'Uncovered mean options')
            c.update(operation='mean',input_per_unit=x['shape'][1:],dim=args[1][0],keepdim=args[2] if len(args)>2 else False)
        cases.append(c)
    require(cases and len(covered)==sum(r['aten_operation'] in ('aten::softmax','aten::mean') and r['arguments'][0]['shape'][0]==8 and r['arguments'][0]['shape'] not in [s['arguments'][0]['shape'] for s in one['operations'] if s['aten_operation']==r['aten_operation']] for r in eight['operations']),'Source coverage gap')
    report=dict(adapter=one['adapter'],adapter_execution_configuration=one['adapter_execution_configuration'],
        source={str(path.resolve()):sha(path) for path in (a.b1,a.b8)},builder_source_sha256=sha(Path(__file__)),
        scope='Request-scaled mean/softmax only. Shared vision-pooling mean and rotary matmul remain separate; measurements not yet complete.',configurations=cases)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(cases)} source request-scaled reduction cases')


if __name__=='__main__':main()

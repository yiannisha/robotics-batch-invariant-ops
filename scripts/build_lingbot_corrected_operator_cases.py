"""Build new operator cases directly from corrected TF32-off B1/B8 traces.

Preserve the actual corrected adapter identity at every component layer. This
constructs cases only; it does not measure arithmetic or certify coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.build_lingbot_exact_linear_cases import pair_cases
from scripts.build_lingbot_exact_remaining_cases import build as remaining_cases

ADAPTER='scripts.model_invariance.adapters.lingbot_robotwin_exact_v2'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_traces(one,eight):
    for trace,batch in [(one,1),(eight,8)]:
        require(trace['adapter']==ADAPTER and trace['batch_size']==batch and trace['implementation']=='native',
                'Actual corrected native B1/B8 source traces required')
        require(trace.get('tf32_matmul') is False and trace.get('tf32_cudnn') is False,'Actual trace precision must be TF32-off')
        config=trace['adapter_execution_configuration']
        require(config.get('tf32') is False and config.get('language_tokens')==72
                and config.get('checkpoint_revision')=='0451855729ec904f970600e0aec8b84661423afe','Wrong controlled LingBot path')
    for key in ('adapter_execution_configuration','replay_manifest_sha256','gpu','torch'):
        require(one[key]==eight[key],'Different corrected source provenance: '+key)


def reduction_cases(one,eight):
    cases=[];covered=set()
    for row in one['operations']:
        operation=row['aten_operation']
        if operation not in ('aten::softmax','aten::mean'):continue
        args=row['arguments'];x=args[0]
        if x['shape'][0]!=1:continue
        shape=[8,*x['shape'][1:]]
        matches=[r for r in eight['operations'] if r['aten_operation']==operation and r['arguments'][1:]==args[1:]
                 and r['arguments'][0]['shape']==shape and r['arguments'][0]['stride']==x['stride'] and r['keyword_arguments']==row['keyword_arguments']]
        require(len(matches)==1,'Missing/ambiguous corrected source batch counterpart');covered.add(json.dumps(matches[0],sort_keys=True))
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
    expected=sum(r['aten_operation'] in ('aten::softmax','aten::mean') and r['arguments'][0]['shape'][0]==8
                 and r['arguments'][0]['shape'] not in [s['arguments'][0]['shape'] for s in one['operations'] if s['aten_operation']==r['aten_operation']]
                 for r in eight['operations'])
    require(cases and len(covered)==expected,'Corrected source reduction coverage gap')
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('b1','b8','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve prior case sets')
    one,eight=[json.loads(path.read_text()) for path in (a.b1,a.b8)];validate_traces(one,eight)
    groups={'linear':pair_cases(one,eight),'reduction':reduction_cases(one,eight),'remaining':remaining_cases(one,eight)}
    a.output.mkdir(parents=True)
    source={str(path.resolve()):sha(path) for path in (a.b1,a.b8)}
    helpers=[Path(__file__),Path(__file__).with_name('build_lingbot_exact_linear_cases.py'),Path(__file__).with_name('build_lingbot_exact_remaining_cases.py')]
    for name,cases in groups.items():
        report=dict(adapter=ADAPTER,adapter_execution_configuration=one['adapter_execution_configuration'],
                    configurations=cases,measurements_complete=False,corrected_actual_trace_precision_verified=True,
                    builder_sources_sha256={str(path.resolve()):sha(path) for path in helpers})
        report['source' if name=='reduction' else 'traces']=source
        (a.output/(name+'_cases.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(case_counts={key:len(value) for key,value in groups.items()},measurements_complete=False)))


if __name__=='__main__':main()

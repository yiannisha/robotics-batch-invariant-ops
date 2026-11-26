"""Retain shared vision pooling and request-axis-1 rotary matmul layouts."""
import argparse
import hashlib
import json
from pathlib import Path


def build(one, eight):
    result = []
    for row in one['operations']:
        operation = row['aten_operation']
        if operation == 'aten::mean' and row['arguments'][0]['shape'][0] == 8:
            matches = [x for x in eight['operations'] if all(x[k] == row[k] for k in ('aten_operation','arguments','keyword_arguments'))]
            axis = None
        elif operation == 'aten::matmul':
            matches = []
            axis = 1
            for other in eight['operations']:
                if other['aten_operation'] != operation or other['keyword_arguments'] != row['keyword_arguments']:
                    continue
                if len(other['arguments']) != len(row['arguments']):
                    continue
                valid = True
                for a, b in zip(row['arguments'], other['arguments']):
                    shape = list(a['shape']); shape[1] *= 8
                    valid &= b['shape'] == shape and a['dtype'] == b['dtype']
                    valid &= all(t == s or t == 8*s or (i == axis and a['shape'][i] == 1 and t == 0)
                                 for i, (s, t) in enumerate(zip(a['stride'], b['stride'])))
                if valid:
                    matches.append(other)
        else:
            continue
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous remaining source pattern')
        identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
        result.append(dict(id='lingbot_exact_remaining_'+identity, source_entries={'b1':row,'b8':matches[0]},
                           operation=operation, batch_axis=axis,
                           coordinate_layout='shared_vision_pooling' if axis is None else 'rotary_request_axis_1'))
    if len(result) != 3:
        raise ValueError('Expected one shared pooling and two rotary matmul patterns')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('b1','b8','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    traces = [json.loads(x.read_text()) for x in (a.b1,a.b8)]
    for t,batch in zip(traces,(1,8)):
        if t['adapter'] != 'scripts.model_invariance.adapters.lingbot_robotwin_exact' or t['batch_size'] != batch or t['implementation'] != 'native':
            raise ValueError('Foreign trace')
    for key in ('adapter_execution_configuration','replay_manifest_sha256','gpu','torch'):
        if traces[0][key] != traces[1][key]:
            raise ValueError('Different trace protocol: '+key)
    d=dict(adapter=traces[0]['adapter'],adapter_execution_configuration=traces[0]['adapter_execution_configuration'],
           traces={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.b1,a.b8)},
           configurations=build(*traces),measurements_complete=False)
    a.output.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'remaining_source_patterns':len(d['configurations']),'measurements_complete':False}))


if __name__ == '__main__':
    main()

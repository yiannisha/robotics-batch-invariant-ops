"""Pair actual LingBot exact-path linear layouts without relying on trace order.

This constructs measurement inputs only; it does not certify qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path


def pair_cases(one, eight):
    cases = []
    used = set()
    linear_one = [r for r in one['operations'] if r['aten_operation'] == 'aten::linear']
    linear_eight = [r for r in eight['operations'] if r['aten_operation'] == 'aten::linear']
    for one_index, row in enumerate(linear_one):
        if row['aten_operation'] != 'aten::linear':
            continue
        args = row['arguments']
        matches = []
        for index, other in enumerate(linear_eight):
            if other['aten_operation'] != 'aten::linear':
                continue
            target = other['arguments']
            if args[1:] != target[1:] or row['keyword_arguments'] != other['keyword_arguments']:
                continue
            a, b = args[0], target[0]
            if a['dtype'] != b['dtype'] or len(a['shape']) != len(b['shape']):
                continue
            axes = [i for i, (s, t) in enumerate(zip(a['shape'], b['shape'])) if s != t]
            if len(axes) > 1 or (axes and b['shape'][axes[0]] != 8 * a['shape'][axes[0]]):
                continue
            if any(t not in (s, 8*s) for s, t in zip(a['stride'], b['stride'])):
                continue
            matches.append((index, other, axes[0] if axes else None))
        if len(matches) != 1:
            raise ValueError(f'Linear source has {len(matches)} possible B8 counterparts: {args}')
        index, other, axis = matches[0]
        if index in used:
            raise ValueError('Two B1 patterns map to the same B8 pattern')
        used.add(index)
        identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
        cases.append(dict(id='lingbot_exact_linear_'+identity, source={'b1': row, 'b8': other, 'b1_index': one_index, 'b8_index': index},
                          batch_axis=axis, groups_per_request=args[0]['shape'][axis] if axis is not None else None,
                          coordinate_layout='contiguous_requests'))
    expected = sum(r['aten_operation'] == 'aten::linear' for r in eight['operations'])
    if not cases or len(used) != expected:
        raise ValueError('Incomplete B1/B8 linear inventory mapping')
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('b1', 'b8', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    traces = [json.loads(x.read_text()) for x in (a.b1, a.b8)]
    for t, batch in zip(traces, (1, 8)):
        if t['adapter'] != 'scripts.model_invariance.adapters.lingbot_robotwin_exact' or t['batch_size'] != batch or t['implementation'] != 'native':
            raise ValueError('Foreign source trace')
    for key in ('adapter_execution_configuration', 'replay_manifest_sha256', 'gpu', 'torch'):
        if traces[0][key] != traces[1][key]:
            raise ValueError('Source traces differ: '+key)
    report = dict(adapter=traces[0]['adapter'], adapter_execution_configuration=traces[0]['adapter_execution_configuration'],
                  traces={str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.b1, a.b8)},
                  configurations=pair_cases(*traces), measurements_complete=False)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'mapped_linear_patterns': len(report['configurations']), 'measurements_complete': False}))


if __name__ == '__main__':
    main()

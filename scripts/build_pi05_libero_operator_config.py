"""Derive operator shape/dtype qualification from the fine-tuned runtime traces."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-b1', type=Path, required=True)
    parser.add_argument('--trace-b8', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operations', nargs='+', help='qualify only these observed ATen operations in a supplemental manifest')
    parser.add_argument('--preserve-matmul-source-layouts', action='store_true')
    args = parser.parse_args()
    traces = [json.loads(p.read_text()) for p in (args.trace_b1, args.trace_b8)]
    for batch, trace in zip((1, 8), traces):
        if trace['adapter'] != 'scripts.model_invariance.adapters.pi05_libero_exact' or trace['batch_size'] != batch:
            raise ValueError('Foreign adapter or batch size')
    for key in ('adapter_execution_configuration', 'replay_manifest_sha256', 'gpu', 'torch'):
        if traces[0][key] != traces[1][key]:
            raise ValueError(f'Trace identity differs: {key}')
    if len(traces[0]['operations']) != len(traces[1]['operations']):
        raise ValueError('B1/B8 operation coverage differs')
    configs = []
    for index, (entry, large) in enumerate(zip(*(t['operations'] for t in traces))):
        op = entry['aten_operation']
        if op != large['aten_operation'] or entry['count'] != large['count']:
            raise ValueError('B1/B8 operation order/count differs')
        if args.operations and op not in args.operations:
            continue
        operands = entry['arguments']
        for small_arg, big_arg in zip(operands, large['arguments']):
            if isinstance(small_arg, dict) and 'shape' in small_arg:
                small, big = small_arg['shape'], big_arg['shape']
                if small_arg['dtype'] != big_arg['dtype'] or not (small == big or (small[1:] == big[1:] and big[0] == small[0] * 8)):
                    raise ValueError('Unexpected operand scaling')
        x = operands[0]
        shape = x['shape']
        c = dict(id=f'actual_{index:02d}_{op.split("::")[1]}', source_operation=op,
                 source_trace_index=index, source_observed_arguments=operands,
                 source_observed_count=entry['count'], dtype=x['dtype'], layout='contiguous')
        if op == 'aten::linear':
            weight = operands[1]['shape']
            c.update(operation='addmm' if len(operands) > 2 and operands[2] is not None else 'mm',
                     m_per_request=math.prod(shape[:-1]), k=shape[-1], n=weight[0], layout='right_transposed')
        elif op == 'aten::matmul':
            other = operands[1]['shape']
            if args.preserve_matmul_source_layouts:
                c.update(operation='matmul',m_per_request=1,layout='source_strided_matmul',
                    source_left_shape=shape,source_right_shape=other,
                    source_left_stride_b1=operands[0]['stride'],source_right_stride_b1=operands[1]['stride'],
                    source_left_stride_b8=large['arguments'][0]['stride'],source_right_stride_b8=large['arguments'][1]['stride'],
                    layout_scope='actual high-level matmul ranks, broadcast and transpose strides; B1/B8 source layouts preserved')
            else:
                c.update(operation='bmm', m_per_request=math.prod(shape[:-2]), m=shape[-2], k=shape[-1], n=other[-1],
                         layout='right_transposed' if operands[1]['stride'][-2] == 1 else 'contiguous',
                         layout_scope='shape-equivalent flattened-head diagnostic; original broadcast strides retained in source evidence')
        elif op == 'aten::mean':
            if operands[1] != [-1]:
                raise ValueError('Unsupported reduction dimensions')
            c.update(operation='mean', m_per_request=shape[0], input_per_unit=shape[1:], dim=-1, keepdim=operands[2])
        elif op == 'aten::softmax':
            if operands[1]!=-1 or (len(operands)>2 and operands[2] not in (None,x['dtype'])):
                raise ValueError('Unsupported softmax dimension or dtype conversion')
            c.update(operation='softmax',m_per_request=shape[0],input_per_unit=shape[1:],dim=-1)
        elif op == 'aten::conv2d':
            c.update(operation='conv2d', m_per_request=shape[0], input_per_image=shape[1:], weight_shape=operands[1]['shape'],
                     stride=operands[3], padding=[0, 0], dilation=[1, 1], groups=1, has_bias=True)
        elif op == 'aten::scaled_dot_product_attention':
            q, k, v = operands[:3]
            c.update(operation='sdpa', m_per_request=shape[0], query_heads=shape[1], key_heads=k['shape'][1], value_heads=v['shape'][1],
                     query_length=shape[2], key_length=k['shape'][2], head_dim=shape[3], value_dim=v['shape'][3],
                     source_q_stride=q['stride'], source_k_stride=k['stride'], source_v_stride=v['stride'],
                     scale=entry['keyword_arguments']['scale'], layout='qkv_source_strided')
        else:
            raise ValueError(f'Unmapped observed operation: {op}')
        configs.append(c)
    if not configs:raise ValueError('No selected observed configurations')
    report = dict(schema_version=1, selected_operations=args.operations, adapter=traces[0]['adapter'],
                  adapter_execution_configuration=traces[0]['adapter_execution_configuration'],
                  replay_manifest_sha256=traces[0]['replay_manifest_sha256'],
                  source_evidence=[dict(path=str(p.resolve()), sha256=digest(p)) for p in (args.trace_b1, args.trace_b8)],
                  configurations=configs)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(configurations=len(configs), output=str(args.output))))


if __name__ == '__main__':
    main()

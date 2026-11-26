"""Bind operator timing cases to qualified native B1/B8 policy traces.

No model execution or timing occurs. The complete operator timing stage remains
pending until isolated execution and independent raw-timing acceptance.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import zlib

SUPPORTED = {'aten::linear', 'aten::matmul', 'aten::bmm', 'aten::mm', 'aten::addmm',
             'aten::conv2d', 'aten::mean', 'aten::softmax', 'aten::scaled_dot_product_attention'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def capture_bytes(capture):
    if capture['encoding'] != 'zlib_base64_original_dtype_bytes_v1':
        raise ValueError('Unknown mask encoding')
    payload = zlib.decompress(base64.b64decode(capture['data'], validate=True))
    if (len(payload) != capture['original_dtype_bytes_count']
            or hashlib.sha256(payload).hexdigest() != capture['original_dtype_bytes_sha256']):
        raise ValueError('Captured mask bytes changed')
    return payload


def attention_signature(entry):
    return [(a['dtype'], a['shape'][1:], a['stride'][1:]) for a in entry['arguments'][:3]]


def mask_binding(model, trace, entry, peers):
    if entry['aten_operation'] != 'aten::scaled_dot_product_attention':
        return None
    args = entry['arguments']
    if len(args) < 4 or args[3] is None:
        if entry.get('attention_mask') is not None:
            raise ValueError('Mask capture without mask argument')
        return None
    if not isinstance(args[3], dict):
        raise ValueError('Unsupported mask argument')
    capture = entry.get('attention_mask')
    if capture is not None:
        capture_bytes(capture)
        if capture['shape'] != args[3]['shape'] or capture['dtype'] != args[3]['dtype']:
            raise ValueError('Mask descriptor/capture differs')
        return dict(kind='direct_lossless_source_capture', capture=capture, selector=None)
    # The original RDT B1 trace predates lossless mask capture. Its target request
    # is also request zero of the qualified B8 mixed-task trace. Derive only that
    # exact request's mask; all matching B8 captures must agree on its bytes.
    if model != 'rdt' or trace['batch_size'] != 1 or args[3]['dtype'] != 'torch.bool':
        raise ValueError('Missing lossless source mask')
    matches = []
    for peer_path, peer in peers:
        if (peer['batch_size'] != 8 or peer['request_ids'][0] != trace['request_ids'][0]
                or peer['adapter_execution_configuration'] != trace['adapter_execution_configuration']):
            continue
        for index, other in enumerate(peer['operations']):
            cap = other.get('attention_mask')
            if (other['aten_operation'] != entry['aten_operation'] or cap is None
                    or attention_signature(other) != attention_signature(entry)
                    or other['keyword_arguments'] != entry['keyword_arguments']):
                continue
            if (cap['dtype'] != 'torch.bool' or cap['shape'][0] != 8
                    or cap['shape'][1:] != args[3]['shape'][1:] or cap['values_shape'][0] != 8):
                continue
            payload = capture_bytes(cap)
            if len(payload) % 8:
                raise ValueError('Invalid request mask storage')
            target = payload[:len(payload) // 8]
            matches.append((target, peer_path, index, cap))
    if not matches or any(m[0] != matches[0][0] for m in matches):
        raise ValueError('B1 source mask derivation is missing or ambiguous')
    target, peer_path, index, cap = matches[0]
    return dict(kind='same_request_zero_from_lossless_B8_capture', capture=cap, selector=0,
                source_trace=str(peer_path), source_trace_sha256=sha(peer_path),
                source_operation_index=index, target_request_id=trace['request_ids'][0],
                target_compact_mask_sha256=hashlib.sha256(target).hexdigest(),
                all_matching_B8_target_masks_agree=True)


def build(base, model, operator_directory=None):
    folders = dict(pi05='pi05_libero', rdt='rdt_maniskill_exact',
                   cogact='cogact_simpler_exact', lingbot='lingbot_robotwin_exact')
    if operator_directory is not None and model != 'lingbot':
        raise ValueError('Corrected qualification override is LingBot-only')
    directory = (Path(operator_directory) if operator_directory is not None
                 else base / 'campaign' / folders[model] / 'full/operator')
    coverage = directory / ('coverage_audit.json' if model == 'pi05' else 'coverage_complete_audit.json')
    evidence = read(coverage)
    if model == 'pi05':
        if evidence['status'] != 'actual_operator_coverage_independently_verified' or not evidence['complete']:
            raise ValueError('Complete PI operator qualification required')
        group = next(g for g in evidence['groups'] if Path(g['configuration_file']).name == 'actual_matmul_source_layout_config.json')
        config = Path(group['configuration_file'])
        if sha(config) != group['configuration_sha256']:
            raise ValueError('Qualified PI configuration changed')
        paths = {Path(e['path']): e['sha256'] for e in read(config)['source_evidence']}
        bound_sources = {str(config): sha(config)}
    else:
        if (evidence['status'] != 'source_operator_inventory_reconciled'
                or not evidence['full_traced_operator_qualification_complete']):
            raise ValueError('Complete source operator qualification required')
        if operator_directory is not None and evidence.get('corrected_actual_trace_precision_verified') is not True:
            raise ValueError('Corrected trace precision audit required')
        paths = {Path(p): h for p, h in evidence['source_traces'].items()}
        bound_sources = {}
    bound_sources[str(coverage)] = sha(coverage)
    peers = []
    for path, expected in paths.items():
        if sha(path) != expected:
            raise ValueError('Qualified operator trace changed')
        trace = read(path)
        if trace['batch_size'] not in (1, 8) or trace['implementation'] != 'native':
            raise ValueError('Expected native B1/B8 source trace')
        if operator_directory is not None:
            if (trace['adapter'] != 'scripts.model_invariance.adapters.lingbot_robotwin_exact_v2'
                    or trace.get('tf32_matmul') is not False or trace.get('tf32_cudnn') is not False):
                raise ValueError('Corrected LingBot adapter and TF32-off traces required')
        bound_sources[str(path)] = expected
        peers.append((path, trace))
    if {t['batch_size'] for _, t in peers} != {1, 8}:
        raise ValueError('Both B1 and B8 source traces required')
    cases = []
    for path, trace in peers:
        for index, entry in enumerate(trace['operations']):
            if entry['aten_operation'] not in SUPPORTED:
                raise ValueError('Unimplemented source operator: ' + entry['aten_operation'])
            modes = ['native', 'batch_invariant']
            if entry['aten_operation'] in ('aten::matmul', 'aten::bmm', 'aten::scaled_dot_product_attention'):
                modes.append('explicit_invariant_per_matrix')
            cases.append(dict(id=f"B{trace['batch_size']}_source_{index:03d}",
                              source_trace=str(path), source_trace_sha256=sha(path),
                              source_operation_index=index, logical_batch_size=trace['batch_size'],
                              entry=copy.deepcopy(entry), modes=modes,
                              variants=(['whole', 'attention_qk', 'attention_pv']
                                        if entry['aten_operation'] == 'aten::scaled_dot_product_attention' else ['whole']),
                              mask_binding=mask_binding(model, trace, entry, peers)))
    return dict(schema_version=1, model=model, adapter=peers[0][1]['adapter'],
                adapter_execution_configuration=peers[0][1]['adapter_execution_configuration'],
                source_sha256=bound_sources, cases=cases,
                timing_protocol=dict(sessions=5, warmup_calls=20, timed_calls=200),
                operand_contract='Retain exact traced dtype, shape and strides. Generate physical storage once, preserving broadcast strides. Restore captured masks losslessly.',
                reference_contract='Attention matrix products: explicit invariant per-matrix MM versus batched persistent invariant path; native measured separately.',
                attention_component_contract='QK and PV are standalone native/invariant BMM versus invariant per-matrix MM. Flattening/setup is excluded. PV uses regenerated source-shaped P operands, not captured probabilities.',
                excludes='Tracing, profiler, allocation, input generation, compilation and initialization from timed regions.',
                status='operator_timing_plan_only_execution_and_independent_acceptance_pending',
                full_campaign_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-base', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model', choices=('pi05', 'rdt', 'cogact', 'lingbot'))
    parser.add_argument('--operator-directory', type=Path)
    args = parser.parse_args()
    if args.operator_directory is not None and args.model != 'lingbot':
        parser.error('--operator-directory requires --model lingbot')
    models = (args.model,) if args.model else ('pi05', 'rdt', 'cogact', 'lingbot')
    plans = {model: build(args.campaign_base, model, args.operator_directory) for model in models}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for model, plan in plans.items():
        plan['builder_sha256'] = sha(__file__)
        (args.output_dir / f'{model}.json').write_text(json.dumps(plan, indent=2) + '\n')
        print(model, len(plan['cases']), 'qualified-trace timing cases', flush=True)


if __name__ == '__main__':
    main()

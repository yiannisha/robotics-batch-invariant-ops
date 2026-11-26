"""Recount isolated operator timing records without importing the timing worker.

Checks complete case/session/mode coverage, original trace layouts and masks,
raw timing arithmetic, source fingerprints and sampled GPU isolation. It does
not re-execute operators or claim original-model operand values were captured.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import zlib

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def layout(signature, descriptor):
    if isinstance(descriptor, dict) and {'shape', 'stride', 'dtype'} <= descriptor.keys():
        require(all(signature[k] == descriptor[k] for k in ('shape', 'stride', 'dtype')), 'Operand layout differs')
        require(len(signature['logical_bytes_sha256']) == 64, 'Missing operand byte fingerprint')
    elif isinstance(descriptor, list):
        require(isinstance(signature, list) and len(signature) == len(descriptor), 'Argument count differs')
        for a, b in zip(signature, descriptor):
            layout(a, b)
    elif isinstance(descriptor, dict):
        require(signature.keys() == descriptor.keys(), 'Keyword arguments differ')
        for key in descriptor:
            layout(signature[key], descriptor[key])
    else:
        require(signature == descriptor, 'Scalar argument differs')


def mask_fingerprint(case):
    binding = case['mask_binding']
    if binding is None:
        return None
    cap = binding['capture']
    raw = zlib.decompress(base64.b64decode(cap['data'], validate=True))
    require(len(raw) == cap['original_dtype_bytes_count']
            and hashlib.sha256(raw).hexdigest() == cap['original_dtype_bytes_sha256'], 'Mask payload changed')
    kinds = {'torch.bool': np.uint8, 'torch.bfloat16': np.uint16,
             'torch.float16': np.uint16, 'torch.float32': np.uint32}
    value = np.frombuffer(raw, dtype=kinds[cap['dtype']]).reshape(cap['values_shape'])
    if binding['selector'] is not None:
        source = json.loads(Path(binding['source_trace']).read_text())
        require(sha(binding['source_trace']) == binding['source_trace_sha256'], 'Derived mask source changed')
        other = source['operations'][binding['source_operation_index']]
        require(other['attention_mask'] == cap and binding['selector'] == 0, 'Foreign derived mask')
        own = json.loads(Path(case['source_trace']).read_text())
        require(own['request_ids'][0] == source['request_ids'][0] == binding['target_request_id'], 'Foreign mask request')
        value = value[:1]
        require(hashlib.sha256(value.tobytes()).hexdigest() == binding['target_compact_mask_sha256'], 'Derived target mask changed')
    else:
        require(case['entry'].get('attention_mask') == cap, 'Mask is not the source capture')
    shape = case['entry']['arguments'][3]['shape']
    return hashlib.sha256(np.broadcast_to(value, shape).copy().tobytes()).hexdigest()


def component_layout(case, variant):
    # Metadata-only reshape semantics: no GPU context, allocation or operand
    # regeneration. This auditor does not import the worker's derivation code.
    import torch
    tensors = [torch.empty_strided(tuple(a['shape']), tuple(a['stride']), device='meta')
               for a in case['entry']['arguments'][:3]]
    q, k, v = tensors
    if case['entry']['keyword_arguments'].get('enable_gqa', False):
        repeats = q.shape[-3] // k.shape[-3]
        k, v = k.repeat_interleave(repeats, -3), v.repeat_interleave(repeats, -3)
    qflat = q.reshape(-1, q.shape[-2], q.shape[-1])
    kflat = k.reshape(-1, k.shape[-2], k.shape[-1])
    if variant == 'attention_qk':
        pair = [qflat, kflat.transpose(-1, -2)]
    elif variant == 'attention_pv':
        pair = [torch.empty((qflat.shape[0], q.shape[-2], k.shape[-2]), device='meta'),
                v.reshape(-1, v.shape[-2], v.shape[-1])]
    else:
        raise ValueError('Unknown attention component')
    return [dict(shape=list(t.shape), stride=list(t.stride()),
                 dtype=case['entry']['arguments'][0]['dtype']) for t in pair]


def audit(plan_path, directory):
    plan_hash = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    isolation_path = directory / 'isolation.json'
    isolation = json.loads(isolation_path.read_text())
    require(isolation['accepted'] is True and isolation['child_exit_code'] == 0
            and isolation['status'] == 'completed_operator_timing_isolation_verified_independent_audit_pending',
            'Accepted completed isolation run required')
    require(isolation['plan_sha256'] == plan_hash, 'Isolation uses another plan')
    raw, journal = Path(isolation['raw']), Path(isolation['observation_journal'])
    runtime_path = Path(str(raw) + '.runtime.json')
    require(sha(raw) == isolation['raw_sha256'] and sha(runtime_path) == isolation['runtime_sha256']
            and sha(journal) == isolation['observation_journal_sha256'], 'Timing/isolation bytes changed')
    runtime = json.loads(runtime_path.read_text())
    require(runtime['status'] == 'completed_raw_operator_timings_independent_acceptance_pending'
            and runtime['raw_sha256'] == isolation['raw_sha256'] and runtime['plan_sha256'] == plan_hash,
            'Incomplete or foreign operator runtime')
    require(runtime['model'] == plan['model'] and runtime['gpu_uuid'] == isolation['gpu_uuid'], 'Model/GPU differs')
    require(runtime['adapter_execution_configuration'] == plan['adapter_execution_configuration']
            and runtime['tf32'] is False and runtime['autocast'] is False, 'Numerical execution configuration differs')
    protocol = dict(sessions=5, warmup_calls=20, timed_calls=200)
    require(runtime['timing_protocol'] == plan['timing_protocol'] == protocol, 'Wrong timing protocol')
    for proof in (isolation, runtime):
        for source, expected in proof['source_sha256'].items():
            require(sha(source) == expected and sha(Path(proof['source_archive']) / expected) == expected,
                    'Measured source or source archive changed')
    scripts = Path(__file__).resolve().parent
    common = {str(scripts / n) for n in ('benchmark_qualified_operator_cases.py',
                                       'operator_performance_operands.py', 'build_operator_performance_plans.py')}
    require(str(scripts / 'benchmark_qualified_operator_cases_v2.py') in runtime['source_sha256']
            and runtime.get('cuda_events_initialized_before_timing') is True, 'CUDA event initialization exclusion unproven')
    require(str(scripts / 'run_isolated_operator_benchmark_v2.py') in isolation['source_sha256'], 'Versioned timing isolation source missing')
    require(common <= runtime['source_sha256'].keys(), 'Missing measured worker source pins')
    require({str(scripts.parent / 'batch_invariant_ops' / n) for n in ('__init__.py', 'batch_invariant_ops.py')}
            <= runtime['source_sha256'].keys(), 'Missing invariant kernel source pins')
    require(common | {str(scripts / n) for n in ('run_isolated_operator_benchmark.py',
                                                'gpu_isolation.py', 'run_isolated_frozen_benchmark.py')}
            <= isolation['source_sha256'].keys(), 'Missing isolation source pins')
    for source, expected in plan['source_sha256'].items():
        require(sha(source) == expected, 'Qualified source evidence changed')
    freeze = json.loads(Path(isolation['freeze']).read_text())
    require(sha(isolation['freeze']) == isolation['freeze_sha256'], 'Numerical freeze changed')
    require(freeze['adapter_execution_configuration'] == plan['adapter_execution_configuration'], 'Frozen configuration differs')
    frozen_runtimes = [Path(p) for p in freeze['content_addressed_freeze']
                       if p.endswith('invariance.jsonl.gz.runtime.json')]
    require(len(frozen_runtimes) == 1, 'Ambiguous frozen diagnostic runtime')
    diagnostic = json.loads(frozen_runtimes[0].read_text())
    require(sha(frozen_runtimes[0]) == freeze['content_addressed_freeze'][str(frozen_runtimes[0])]['sha256'],
            'Frozen diagnostic runtime changed')
    require(runtime['torch'] == diagnostic['torch'] and runtime['cuda'] == diagnostic['cuda']
            and runtime['gpu_name'] == diagnostic['gpu'] and diagnostic['adapter'] == plan['adapter'],
            'Operator timing runtime differs from frozen model runtime')
    for filename, expected in freeze['content_addressed_freeze'].items():
        if filename.endswith('.py'):
            require(sha(filename) == expected['sha256'], 'Frozen numerical source changed')
    observed = []
    with journal.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row['gpu_uuid'] == isolation['gpu_uuid'] and row['gpu_name'] == runtime['gpu_name'], 'Observed GPU differs')
            require(all(p['pid'] == isolation['benchmark_pid'] for p in row['processes']), 'Foreign GPU process')
            observed.append(row)
    require(len(observed) == isolation['observation_count'] and len(observed) >= 3
            and not observed[0]['processes']
            and any(row['processes'] for row in observed), 'Missing pre/during/post isolation evidence')
    cases = {c['id']: c for c in plan['cases']}
    require(len(cases) == len(plan['cases']) and cases, 'Duplicate or empty case roster')
    coverage = []
    for name in plan['source_sha256']:
        value = json.loads(Path(name).read_text())
        if value.get('status') in ('actual_operator_coverage_independently_verified', 'source_operator_inventory_reconciled'):
            coverage.append((name, value))
    require(len(coverage) == 1, 'Missing or ambiguous qualified operator coverage')
    coverage_path, qualification = coverage[0]
    require(coverage_path in freeze['content_addressed_freeze']
            and sha(coverage_path) == freeze['content_addressed_freeze'][coverage_path]['sha256'],
            'Operator qualification is not frozen')
    if plan['model'] == 'pi05':
        require(qualification['complete'] is True, 'Incomplete PI operator qualification')
        configs = [Path(g['configuration_file']) for g in qualification['groups']
                   if Path(g['configuration_file']).name == 'actual_matmul_source_layout_config.json']
        require(len(configs) == 1 and str(configs[0]) in freeze['content_addressed_freeze'], 'Missing bound PI trace configuration')
        require(sha(configs[0]) == freeze['content_addressed_freeze'][str(configs[0])]['sha256'], 'PI trace configuration changed')
        evidence = json.loads(configs[0].read_text())['source_evidence']
        require(all(sha(x['path']) == x['sha256'] == plan['source_sha256'].get(x['path']) for x in evidence),
                'PI qualified trace fingerprint differs')
        trace_paths = {x['path'] for x in evidence}
    else:
        require(qualification['full_traced_operator_qualification_complete'] is True, 'Incomplete source operator qualification')
        trace_paths = set(qualification['source_traces'])
        require(all(sha(p) == h == plan['source_sha256'].get(p)
                    for p, h in qualification['source_traces'].items()), 'Qualified trace fingerprint differs')
    require({c['source_trace'] for c in cases.values()} == trace_paths, 'Missing qualified source trace')
    expected_sources = {(p, i) for p in trace_paths
                        for i, _ in enumerate(json.loads(Path(p).read_text())['operations'])}
    actual_sources = [(c['source_trace'], c['source_operation_index']) for c in cases.values()]
    require(set(actual_sources) == expected_sources and len(actual_sources) == len(expected_sources),
            'Missing or duplicate qualified trace case')
    require({json.loads(Path(p).read_text())['batch_size'] for p in trace_paths} == {1, 8}, 'Missing B1/B8 source cases')
    masks = {}
    for case in cases.values():
        source = json.loads(Path(case['source_trace']).read_text())
        require(sha(case['source_trace']) == case['source_trace_sha256']
                and source['operations'][case['source_operation_index']] == case['entry'], 'Operator case differs from source trace')
        require(source['adapter'] == plan['adapter'] and source['batch_size'] == case['logical_batch_size']
                and source['adapter_execution_configuration'] == plan['adapter_execution_configuration'],
                'Foreign operator trace')
        modes = ['native', 'batch_invariant']
        if case['entry']['aten_operation'] in ('aten::matmul', 'aten::bmm', 'aten::scaled_dot_product_attention'):
            modes.append('explicit_invariant_per_matrix')
        require(case['modes'] == modes, 'Required attention reference missing')
        variants = (['whole', 'attention_qk', 'attention_pv']
                    if case['entry']['aten_operation'] == 'aten::scaled_dot_product_attention' else ['whole'])
        require(case['variants'] == variants, 'Required native-BMM components missing')
        masks[case['id']] = mask_fingerprint(case)
    expected = {(c['id'], session, variant, mode) for c in cases.values() for session in range(5)
                for variant in c['variants'] for mode in c['modes']}
    seen, shared_inputs, source_inputs, statuses = set(), {}, {}, Counter()
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['case_id'], row['session'], row['variant'], row['mode']
            require(key in expected and key not in seen, 'Duplicate or foreign timing slot')
            seen.add(key)
            case = cases[row['case_id']]
            require(row['plan_sha256'] == plan_hash and row['model'] == plan['model']
                    and row['gpu_uuid'] == isolation['gpu_uuid'] and row['timing_protocol'] == protocol,
                    'Timing provenance differs')
            require(row['source_trace_sha256'] == case['source_trace_sha256']
                    and row['source_operation_index'] == case['source_operation_index']
                    and row['operation'] == case['entry']['aten_operation']
                    and row['logical_batch_size'] == case['logical_batch_size'], 'Timing source case differs')
            require(row['measured_operation'] == (case['entry']['aten_operation'] if row['variant'] == 'whole' else 'aten::bmm'),
                    'Measured kernel family differs')
            seed = int.from_bytes(hashlib.sha256(f"{plan['model']}/{case['id']}/{row['session']}/20261109".encode()).digest()[:8], 'big') % (2 ** 63)
            require(row['seed'] == seed, 'Operand seed differs')
            require(row['probabilities_seed'] == ((seed + 1) % (2 ** 63) if row['variant'] == 'attention_pv' else None),
                    'PV operand seed differs')
            index = next(i for i, c in enumerate(plan['cases']) if c['id'] == case['id'])
            shift = (index + row['session'] + case['variants'].index(row['variant'])) % len(case['modes'])
            order = case['modes'][shift:] + case['modes'][:shift]
            require(order[row['mode_position']] == row['mode'], 'Counterbalanced mode order differs')
            statuses[row['status']] += 1
            if row['status'] == 'cuda_capacity_failure':
                require(row['phase'] in ('operand_allocation', 'operator_execution') and row['error'], 'Invalid capacity record')
                require('cuda_seconds' not in row, 'Capacity failure has fabricated timing samples')
                continue
            require(row['status'] == 'timed' and row['same_case_output_repeat_exact'] is True, 'Incomplete operator timing')
            output = row['output_signature']
            require(output['shape'] and len(output['shape']) == len(output['stride'])
                    and all(n > 0 for n in output['shape']) and all(s >= 0 for s in output['stride'])
                    and len(output['logical_bytes_sha256']) == 64
                    and set(output['logical_bytes_sha256']) <= set('0123456789abcdef'),
                    'Missing output repeat fingerprint')
            source_signature = row['source_operand_signatures']
            layout(source_signature[0], case['entry']['arguments'])
            layout(source_signature[1], case['entry']['keyword_arguments'])
            if masks[case['id']] is not None:
                require(source_signature[0][3]['logical_bytes_sha256'] == masks[case['id']], 'Measured source mask differs')
            source_key = row['case_id'], row['session']
            if source_key in source_inputs:
                require(source_inputs[source_key] == source_signature, 'Components used different source operands')
            else:
                source_inputs[source_key] = source_signature
            signature = row['operand_signatures']
            if row['variant'] == 'whole':
                require(signature == source_signature, 'Whole operator operands differ')
            else:
                layout(signature[0], component_layout(case, row['variant']))
                require(signature[1] == {}, 'Unexpected matrix-component keyword argument')
            cohort = row['case_id'], row['session'], row['variant']
            if cohort in shared_inputs:
                require(shared_inputs[cohort] == signature, 'Modes timed different operands')
            else:
                shared_inputs[cohort] = signature
            for samples_key, median_key, tail_key in (
                    ('cuda_seconds', 'median_cuda_seconds', 'p95_cuda_seconds'),
                    ('synchronized_wall_seconds', 'median_synchronized_wall_seconds', 'p95_synchronized_wall_seconds')):
                values = row[samples_key]
                require(len(values) == 200 and all(math.isfinite(x) and x > 0 for x in values), 'Invalid raw timing samples')
                require(math.isclose(statistics.median(values), row[median_key], rel_tol=1e-12, abs_tol=1e-20)
                        and math.isclose(statistics.quantiles(values, n=100, method='inclusive')[94], row[tail_key], rel_tol=1e-12, abs_tol=1e-20),
                        'Incorrect timing summary')
            require(math.isclose(row['whole_operation_calls_per_second'], 1 / row['median_cuda_seconds'], rel_tol=1e-12), 'Incorrect whole-operation throughput')
            require(row['kernel_launch_count'] is None and row['peak_incremental_allocated_bytes'] >= 0, 'Invented launch count or negative memory')
    require(seen == expected and runtime['raw_records'] == len(seen), 'Incomplete operator timing factorial')
    require(sha(raw) == isolation['raw_sha256'] and sha(plan_path) == plan_hash, 'Measurements changed during audit')
    return dict(status='complete_operator_timing_records_independently_recounted', model=plan['model'],
                plan_sha256=plan_hash, raw_sha256=sha(raw), isolation_sha256=sha(isolation_path),
                runtime_sha256=sha(runtime_path), case_count=len(cases), timing_slots=len(seen),
                statuses=dict(statuses), all_planned_modes_timed=statuses['cuda_capacity_failure'] == 0,
                generated_numerical_operand_bytes_independently_rederived=False,
                source_attention_masks_independently_rederived=True,
                timing_and_isolation_scope='Retained raw samples, trace layouts, captured masks and sampled compute/graphics inventory.',
                auditor_sha256=sha(__file__), full_campaign_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.plan, args.directory)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

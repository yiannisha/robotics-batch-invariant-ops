"""Verify complete causal measurement contracts and recount descriptive results.

This audits retained measurement records, fingerprints and source provenance.
It does not reconstruct missing tensor arrays or independently re-execute the
model. Numerical tensor errors remain measured by the pinned replay runner.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

MODES = {'native', 'native_deterministic', 'existing_invariant_ops',
         'operators_plus_patch_projection', 'explicit_per_matrix_reference', 'full_invariant'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_metrics(metric):
    tensors = metric['tensors']
    require(tensors, 'Empty tensor metrics')
    elements = unequal = 0
    squares = []
    maximum = 0.
    for tensor in tensors.values():
        n = math.prod(tensor['shape'])
        require(n > 0 and tensor['nonfinite_count'] == 0, 'Invalid tensor shape or nonfinite values')
        bad = tensor['unequal_element_count']
        require(isinstance(bad, int) and 0 <= bad <= n, 'Invalid unequal-element count')
        require(tensor['exact'] == (bad == 0), 'Contradictory exactness count')
        left, right = tensor['reference_sha256'], tensor['candidate_sha256']
        require(all(isinstance(h, str) and len(h) == 64
                    and set(h) <= set('0123456789abcdef') for h in (left, right)), 'Invalid tensor hash')
        require(tensor['exact'] == (left == right), 'Contradictory byte fingerprints')
        rmse, peak = tensor['rmse'], tensor['max_absolute_error']
        require(all(math.isfinite(x) and x >= 0 for x in (rmse, peak)), 'Invalid numerical error')
        require(rmse <= peak + max(1e-20, peak * 1e-12), 'RMSE exceeds maximum')
        require(not tensor['exact'] or (rmse == 0 and peak == 0), 'Nonzero exact error')
        elements += n
        unequal += bad
        squares.append(n * rmse * rmse)
        maximum = max(maximum, peak)
    require(metric['nonfinite_count'] == 0 and metric['unequal_element_count'] == unequal,
            'Aggregate count differs')
    require(metric['exact'] == (unequal == 0), 'Aggregate exactness differs')
    require(metric['max_absolute_error'] == maximum, 'Aggregate maximum differs')
    require(math.isclose(metric['rmse'], math.sqrt(math.fsum(squares) / elements),
                         rel_tol=1e-11, abs_tol=1e-20), 'Aggregate RMS differs')


def audit(raw, diagnostic_audit):
    raw_hash = sha(raw)
    sidecar_path = Path(str(raw) + '.provenance.json')
    sidecar_hash = sha(sidecar_path)
    sidecar = json.loads(sidecar_path.read_text())
    require(sidecar['status'] == 'measurements_completed_independent_audit_pending',
            'Completed wrapper execution required')
    require(sidecar['output_sha256'] == raw_hash, 'Measurements changed')
    require(sidecar.get('tf32_matmul') is False and sidecar.get('tf32_cudnn') is False,'Actual causal runtime precision required')
    plan_path = Path(sidecar['plan'])
    require(sha(plan_path) == sidecar['plan_sha256'], 'Plan changed')
    plan = json.loads(plan_path.read_text())
    adapters = dict(pi05='pi05_libero_exact', rdt='rdt_maniskill_exact',
                    cogact='cogact_simpler_exact', lingbot='lingbot_robotwin_exact_v2')
    require(plan['model'] in adapters and plan['adapter'] ==
            'scripts.model_invariance.adapters.' + adapters[plan['model']], 'Foreign model plan')
    repo = Path(__file__).resolve().parents[1]
    required_sources = [repo / 'scripts/run_exact_model_causal_ablation_v4.py',
                        repo / 'scripts/run_model_causal_ablation_v3.py',
                        repo / 'batch_invariant_ops/batch_invariant_ops.py',
                        repo / ('scripts/model_invariance/adapters/' + adapters[plan['model']] + '.py')]
    require(all(str(p) in plan['source_sha256'] for p in required_sources), 'Missing experiment source pins')
    require(plan['source_sha256'] == sidecar['source_sha256'], 'Source pin inventory differs')
    for path, value in plan['source_sha256'].items():
        require(sha(path) == value, 'Source changed: ' + path)
    manifest_path = Path(plan['manifest'])
    require(sha(manifest_path) == plan['manifest_sha256'] == sidecar['manifest_sha256'], 'Manifest changed')
    manifest = json.loads(manifest_path.read_text())
    acceptance_hash = sha(diagnostic_audit)
    acceptance = json.loads(diagnostic_audit.read_text())
    require(acceptance['status'] == 'complete_four_model_diagnostic_independently_verified'
            and acceptance['request_noise_pairs'] == 600, 'Complete diagnostic acceptance required')
    require(acceptance['manifest_sha256'] == plan['manifest_sha256'], 'Foreign diagnostic acceptance')
    labels = set(acceptance['required_denoising_labels'])
    rows = manifest['observations']
    selected = [r for r in rows if r['split'] == 'diagnostic']
    expected = {request: (r, noise) for r in selected for noise, request in enumerate(r['request_ids'])}
    require(len(selected) == 200 and len(expected) == 600, 'Incomplete diagnostic roster')
    seen = set()
    groups = defaultdict(list)
    boundaries = defaultdict(Counter)
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            mode, request = row['mode'], row['request_id']
            require(mode in MODES and request in expected, 'Foreign mode or request')
            require((mode, request) not in seen, 'Duplicate mode/request')
            seen.add((mode, request))
            source, noise = expected[request]
            require(row['task'] == source['task'] and row['episode_id'] == source['episode_id']
                    and row['observation_index'] == source['observation_index'] and row['noise_id'] == noise,
                    'Source identity differs')
            require(row['request_noise_sha256'] == source['noise'][noise]['sha256'], 'Noise fingerprint differs')
            require(row['replay_manifest_sha256'] == plan['manifest_sha256'] and row['adapter'] == plan['adapter'],
                    'Foreign model or manifest')
            require(row['adapter_execution_configuration'] == acceptance['adapter_execution_configuration'],
                    'Loaded inference configuration differs')
            require(row['batch_size'] == 2, 'Causal comparison must use B2')
            require(row['deterministic_algorithms'] == (mode == 'native_deterministic'), 'Deterministic mode differs')
            companion_index = source['companion_replay_indices'][0]
            companion = rows[companion_index]
            require(companion['split'] == 'diagnostic'
                    and (companion['task'], companion['episode_id']) != (source['task'], source['episode_id']),
                    'Heldout or same-episode companion')
            require(row['companion_replay_index'] == companion_index
                    and row['companion_request_id'] == companion['request_ids'][noise], 'Companion identity differs')
            require(labels <= row['flow_steps'].keys(), 'Missing required denoising stages')
            validate_metrics(row['output'])
            require(set(row['output']['tensors']) == {'output.normalized', 'output.physical'},
                    'Missing normalized or physical returned output')
            for metric in row['flow_steps'].values():
                validate_metrics(metric)
            order = sorted(k for k in row['flow_steps'] if k.startswith('flow.step_'))
            order += sorted(k for k in row['flow_steps'] if k.startswith('actions.'))
            first = next((k for k in order if not row['flow_steps'][k]['exact']), None)
            first = first or ('returned_output' if not row['output']['exact'] else None)
            attribution = row['first_divergence']
            require((attribution is None) == (first is None), 'Divergence attribution missing or spurious')
            if first is not None:
                require(attribution['observed_boundary'] == first, 'Wrong first recorded divergence')
                metric = row['output'] if first == 'returned_output' else row['flow_steps'][first]
                require(attribution['metrics'] == metric, 'Boundary metrics differ')
                boundaries[mode][first] += 1
            counts = row['override_invocations']
            require(all(isinstance(v, int) and v >= 0 for v in counts.values()), 'Invalid override counts')
            inapplicable = row['inapplicable_operations']
            allowed = sidecar['inapplicable_operations']
            for op, contract in inapplicable.items():
                require(op in allowed and all(contract.get(k) == v for k, v in allowed[op].items()),
                        'Unpinned inapplicability claim')
                require(sha(contract['evidence_path']) == contract['evidence_sha256'], 'Changed inapplicability proof')
            if mode == 'existing_invariant_ops':
                require(sum(counts.get(k, 0) for k in ('mm', 'addmm', 'bmm')) > 0, 'Matrix intervention bypassed')
            if mode in ('operators_plus_patch_projection', 'explicit_per_matrix_reference'):
                require('convolution' in inapplicable or counts.get('convolution', 0) > 0, 'Convolution bypassed')
            if mode == 'explicit_per_matrix_reference':
                require(counts.get('explicit_sdpa', 0) > 0 if 'explicit_sdpa' not in inapplicable
                        else counts.get('explicit_bmm', 0) > 0, 'Attention intervention bypassed')
            if mode == 'full_invariant':
                require(first is None and row['output']['exact'], 'Complete invariant path diverged')
            groups[mode].append(row)
    require(seen == {(mode, request) for mode in MODES for request in expected}, 'Incomplete six-mode factorial')
    summaries = []
    for mode, values in sorted(groups.items()):
        summaries.append(dict(mode=mode, requests=len(values),
                              nonexact_outputs=sum(not r['output']['exact'] for r in values),
                              any_recorded_boundary_divergence=sum(r['first_divergence'] is not None for r in values),
                              maximum_output_error=max(r['output']['max_absolute_error'] for r in values),
                              first_recorded_boundaries=dict(boundaries[mode])))
    require(sha(raw) == raw_hash and sha(sidecar_path) == sidecar_hash
            and sha(diagnostic_audit) == acceptance_hash, 'Evidence changed during audit')
    require(sha(plan_path) == sidecar['plan_sha256'] and sha(manifest_path) == plan['manifest_sha256']
            and all(sha(path) == value for path, value in plan['source_sha256'].items()),
            'Pinned input changed during audit')
    return dict(status='complete_six_mode_causal_measurement_contracts_verified', model=plan['model'],
                measurement_records=len(seen), request_noise_pairs=600, rows=summaries,
                raw_sha256=raw_hash, provenance_sha256=sidecar_hash, plan_sha256=sha(plan_path),
                diagnostic_acceptance_sha256=acceptance_hash, auditor_sha256=sha(Path(__file__)),
                scope='Retained measurement contract, byte fingerprints, intervention coverage and first recorded boundary',
                tensor_errors_independently_rederived=False, local_operator_reexecution_verified=False,
                full_campaign_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'diagnostic-audit', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing audit')
    result = audit(args.raw, args.diagnostic_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Verified complete 3600-record causal measurement contracts', flush=True)


if __name__ == '__main__':
    main()

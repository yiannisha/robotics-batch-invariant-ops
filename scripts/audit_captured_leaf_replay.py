"""Bind captured operands to a diagnostic trace and reproduce local operations.

This proves local reproduction and reports explicit causal prerequisites. It
never promotes a first comparable leaf to a globally first operator claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.captured_leaf_operator import replay_operator
from scripts.captured_tensor_layout import restore, byte_hash


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def select_target(reference, candidate, batch, blocks):
    if (reference.ndim < 1 or candidate.ndim != reference.ndim
            or reference.shape[1:] != candidate.shape[1:] or reference.shape[0] % blocks
            or candidate.shape[0] != reference.shape[0] * batch):
        return None
    grouped = candidate.reshape(blocks, batch, reference.shape[0] // blocks, *reference.shape[1:])
    return grouped[:, 0].reshape(reference.shape)


def validate_records(records, trace, manifest):
    names = {mode + '_' + batch for mode in ('native', 'batch_invariant') for batch in ('singleton', 'batched')}
    check(records.keys() == names, 'Complete four-artifact native/invariant capture required')
    check(set(trace['modes']) == {'native', 'batch_invariant'}, 'Both implementations required')
    target = manifest['observations'][trace['target_replay_index']]
    noise = trace['noise_id']
    check(noise in (0, 1, 2) and target['split'] == 'diagnostic', 'Diagnostic input/noise required')
    check(all(manifest['observations'][i]['split'] == 'diagnostic' for i in trace['companion_replay_indices']),
          'Heldout companion exposure')
    check(trace['companion_replay_indices'] == target['companion_replay_indices'][:trace['batch_size'] - 1],
          'Companion schedule differs')
    check(all((manifest['observations'][i]['task'], manifest['observations'][i]['episode_id'])
              != (target['task'], target['episode_id']) for i in trace['companion_replay_indices']),
          'Same-episode companion')
    first = trace['host_leaf_scan']['first_divergent_leaf']
    config = None
    for name, record in records.items():
        check(record['operator'] == {'Linear': 'linear', 'Conv2d': 'conv2d'}.get(first['module_type']),
              'Captured operator type differs')
        mode = 'batch_invariant' if name.startswith('batch_invariant_') else 'native'
        batch = trace['batch_size'] if name.endswith('_batched') else 1
        p = record['provenance']
        check(record['runtime']['invariant_mode'] == (mode == 'batch_invariant'), 'Captured implementation flag differs')
        check(p['adapter'] == trace['adapter'] and p['mode'] == mode and p['logical_batch_size'] == batch,
              'Captured adapter/mode/batch differs')
        check(p['replay_manifest_sha256'] == trace['replay_manifest_sha256']
              and p['target_replay_index'] == trace['target_replay_index'] and p['noise_id'] == noise
              and p['request_noise_sha256'] == target['noise'][noise]['sha256']
              and p['target_request_id'] == target['request_ids'][noise], 'Captured request/noise differs')
        check(p['companion_replay_indices'] == trace['companion_replay_indices']
              and p['seed'] == trace['seed']
              and p['module_name'] == first['name'] and p['module_occurrence'] == first['occurrence'],
              'Captured boundary differs')
        check(p['projection_contract'] == trace['request_projection_contract'], 'Projection contract differs')
        check(p['execution_configuration'] and 'checkpoint_revision' in p['execution_configuration'],
              'Missing loaded checkpoint configuration')
        config = p['execution_configuration'] if config is None else config
        check(p['execution_configuration'] == config, 'Mixed loaded execution configurations')
    for parameter in ('weight', 'bias'):
        values = [record[parameter] for record in records.values()]
        check(all(v is None for v in values) or all(v is not None for v in values), 'Mixed parameter presence')
        if values[0] is not None:
            check(len({(v['dtype'], tuple(v['shape']), v['logical_sha256']) for v in values}) == 1,
                  'Captured weights differ across batch or implementation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-report', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    check(not args.output.exists(), 'Preserve existing audit')
    torch.set_num_threads(8)
    trace_hash, manifest_hash = sha(args.trace_report), sha(args.manifest)
    trace = json.loads(args.trace_report.read_text())
    check(trace['replay_manifest_sha256'] == manifest_hash, 'Wrong replay manifest')
    source_hashes = trace['request_projection_contract']['source_sha256']
    check(source_hashes and all(sha(p) == h for p, h in source_hashes.items()), 'Trace source changed')
    scan_path = Path(trace['leaf_scan_file'])
    check(sha(scan_path) == trace['leaf_scan_sha256'], 'Leaf scan changed')
    scan = json.loads(scan_path.read_text())
    index = trace['host_leaf_scan']['first_divergent_index']
    check(isinstance(index, int) and 0 <= index < len(scan['candidate']), 'No comparable divergent leaf')
    observed = [i for i, r in enumerate(scan['candidate']) if r.get('comparable') and not r.get('exact')]
    check(observed and observed[0] == index, 'First comparable leaf index differs')
    check(trace['earlier_unresolved_leaf_count'] == sum(not r.get('comparable') for r in scan['candidate'][:index]),
          'Unresolved prefix differs')
    artifact_directory = Path(trace['captured_operator_directory']).resolve()
    check(artifact_directory == Path(str(args.trace_report) + '.operators').resolve(), 'Artifact directory differs')
    names = [mode + '_' + batch for mode in ('native', 'batch_invariant') for batch in ('singleton', 'batched')]
    files = {name: artifact_directory / (name + '.pt') for name in names}
    expected_paths = {str(path) for path in files.values()}
    check(set(trace['captured_operator_artifact_sha256']) == expected_paths, 'Incomplete or unsupported operator capture')
    records = {}
    for name, path in files.items():
        check(sha(path) == trace['captured_operator_artifact_sha256'][str(path)], 'Captured artifact changed')
        records[name] = torch.load(path, map_location='cpu', weights_only=False)
    validate_records(records, trace, json.loads(args.manifest.read_text()))
    first = trace['host_leaf_scan']['first_divergent_leaf']
    blocks = 2 if trace['adapter'].endswith('.cogact_simpler_exact') and 'action_model.net' in first['name'] else 1
    conditions, inputs = {}, {}
    results = {name: replay_operator(record, args.device) for name, record in records.items()}
    for mode in ('native', 'batch_invariant'):
        a, b = records[mode + '_singleton'], records[mode + '_batched']
        x = restore(a['input'], 'cpu', require_pointer_alignment=False)
        y = restore(b['input'], 'cpu', require_pointer_alignment=False)
        target_input = select_target(x, y, trace['batch_size'], blocks)
        input_exact = target_input is not None and byte_hash(x) == byte_hash(target_input)
        u = restore(a['output'], 'cpu', require_pointer_alignment=False)
        v = restore(b['output'], 'cpu', require_pointer_alignment=False)
        target_output = select_target(u, v, trace['batch_size'], blocks)
        check(target_output is not None, 'Unresolved captured output projection')
        hashes = byte_hash(u), byte_hash(target_output)
        measured = trace['modes'][mode]['first_leaf']
        check(len(measured['tensors']) == 1 and measured['tensors'][0]['reference_hash'] == hashes[0]
              and measured['tensors'][0]['candidate_hash'] == hashes[1]
              and measured['exact'] == (hashes[0] == hashes[1]), 'Trace/captured leaf output differs')
        if mode == 'native':
            check(scan['reference'][index]['hash'] == hashes[0]
                  and scan['candidate'][index]['target_hash'] == hashes[1], 'Repeated native leaf capture differs from scan')
        conditions[mode] = dict(target_input_comparable=target_input is not None,
                                target_input_exact=input_exact, target_output_exact=hashes[0] == hashes[1])
        inputs[mode] = a['input']['logical_sha256'], b['input']['logical_sha256']
    check(trace['modes']['batch_invariant']['final_output']['exact'], 'Invariant final model output differs')
    common = inputs['native'] == inputs['batch_invariant']
    native_dependence = conditions['native']['target_input_exact'] and not conditions['native']['target_output_exact']
    invariant_stable = conditions['batch_invariant']['target_input_exact'] and conditions['batch_invariant']['target_output_exact']
    check(sha(args.trace_report) == trace_hash and sha(args.manifest) == manifest_hash
          and sha(scan_path) == trace['leaf_scan_sha256']
          and all(sha(p) == h for p, h in source_hashes.items())
          and all(sha(path) == trace['captured_operator_artifact_sha256'][str(path)] for path in files.values()),
          'Evidence changed during replay')
    proof = dict(status='captured_four_condition_leaf_outputs_independently_replayed',
                 trace_report_sha256=trace_hash, manifest_sha256=manifest_hash,
                 captured_artifact_sha256=trace['captured_operator_artifact_sha256'], local_replays=results,
                 comparisons=conditions, same_operands_across_implementations=common,
                 native_leaf_batch_dependence_demonstrated=native_dependence,
                 invariant_leaf_bitwise_stability_demonstrated=invariant_stable,
                 same_operand_repair_demonstrated=common and native_dependence and invariant_stable,
                 earlier_unresolved_leaf_count=trace['earlier_unresolved_leaf_count'],
                 scope='Supported first comparable leaf, not globally first operator',
                 globally_first_operator_attribution_complete=False, auditor_sha256=sha(Path(__file__)),
                 full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(proof, stream, indent=2)
        stream.write('\n')
    print('Replayed four local leaf captures; causal prerequisite flags retained explicitly', flush=True)


if __name__ == '__main__':
    main()

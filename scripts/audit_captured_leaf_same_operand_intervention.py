"""Replay native captured operands under both operator implementations.

This is a counterfactual local intervention, not a full-model rerun or a
claim about the globally first divergent operation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts.captured_leaf_operator import replay_operator, recorded_runtime
from scripts.captured_tensor_layout import restore, byte_hash
from scripts.audit_captured_leaf_replay import select_target


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'local-audit', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Preserve previous interventions')
    torch.set_num_threads(8)
    trace, audit = json.loads(a.trace.read_text()), json.loads(a.local_audit.read_text())
    require(audit['status'] == 'captured_four_condition_leaf_outputs_independently_replayed', 'Accepted local replay required')
    require(sha(a.trace) == audit['trace_report_sha256'], 'Trace binding differs')
    require(sha(Path(__file__).with_name('audit_captured_leaf_replay.py')) == audit['auditor_sha256'], 'Local auditor changed')
    paths = [a.trace, a.local_audit, Path(__file__), *[Path(__file__).with_name(n) for n in
             ('captured_leaf_operator.py', 'captured_tensor_layout.py', 'audit_captured_leaf_replay.py')]]
    paths += [Path(v) for v in audit['captured_artifact_sha256']]
    before = {str(v.resolve()): sha(v) for v in paths}
    for path, identity in audit['captured_artifact_sha256'].items():
        require(sha(Path(path)) == identity, 'Captured bytes changed')
    directory = Path(trace['captured_operator_directory'])
    records = {name: torch.load(directory / ('native_' + name + '.pt'), map_location='cpu', weights_only=False)
               for name in ('singleton', 'batched')}
    require(all(r['runtime']['invariant_mode'] is False for r in records.values()), 'Native operands required')
    require(records['singleton']['operator'] == records['batched']['operator'], 'Operator differs')
    require(records['singleton']['configuration'] == records['batched']['configuration'], 'Configuration differs')
    require(records['singleton']['runtime'] == records['batched']['runtime'], 'Native runtime differs')
    for parameter in ('weight', 'bias'):
        x, y = (r[parameter] for r in records.values())
        require((x is None and y is None) or (x is not None and y is not None and
                x['logical_sha256'] == y['logical_sha256']), 'Parameters differ')
    first = trace['host_leaf_scan']['first_divergent_leaf']
    blocks = 2 if trace['adapter'].endswith('.cogact_simpler_exact') and 'action_model.net' in first['name'] else 1
    cpu_inputs = {name: restore(r['input'], 'cpu', require_pointer_alignment=False) for name, r in records.items()}
    target = select_target(cpu_inputs['singleton'], cpu_inputs['batched'], trace['batch_size'], blocks)
    require(target is not None and byte_hash(target) == byte_hash(cpu_inputs['singleton']), 'Target inputs differ')
    originals = {name: replay_operator(r, 'cuda') for name, r in records.items()}
    results, hashes = {}, {}
    for invariant in (False, True):
        mode = 'batch_invariant' if invariant else 'native'
        outputs = {}
        for name, record in records.items():
            runtime = copy.deepcopy(record['runtime'])
            runtime['invariant_mode'] = invariant
            repeats = []
            with recorded_runtime(runtime):
                for _ in range(2):
                    operands = {k: restore(record[k], 'cuda', require_pointer_alignment=True)
                                if record[k] is not None else None for k in ('input', 'weight', 'bias')}
                    require(all(byte_hash(operands[k]) == record[k]['logical_sha256'] for k in operands if operands[k] is not None), 'Intervention operand bytes differ')
                    if record['operator'] == 'linear':
                        out = F.linear(operands['input'], operands['weight'], operands['bias'])
                    else:
                        out = F.conv2d(operands['input'], operands['weight'], operands['bias'], **record['configuration'])
                    require(torch.isfinite(out).all().item(), 'Nonfinite intervention output')
                    repeats.append(out.detach().cpu().clone())
            require(byte_hash(repeats[0]) == byte_hash(repeats[1]), 'Local repeats differ')
            if not invariant:
                require(byte_hash(repeats[0]) == originals[name]['output_sha256'], 'Native capture reproduction changed')
            outputs[name] = repeats[0]
            hashes[mode + '_' + name] = byte_hash(repeats[0])
        target = select_target(outputs['singleton'], outputs['batched'], trace['batch_size'], blocks)
        require(target is not None, 'Output projection unresolved')
        error = (outputs['singleton'].double() - target.double()).abs()
        results[mode] = dict(target_output_exact=byte_hash(outputs['singleton']) == byte_hash(target),
                             maximum_absolute_error=error.max().item(), repeats_exact=True)
    for path, identity in before.items():
        require(sha(Path(path)) == identity, 'Intervention source or input changed')
    report = dict(status='captured_native_operands_cross_implementation_intervention_completed',
                  input_and_source_sha256=before, adapter=trace['adapter'],
                  target_replay_index=trace['target_replay_index'], noise_id=trace['noise_id'],
                  first_comparable_leaf=first, same_native_operand_bytes_used_for_both_implementations=True,
                  comparisons=results, output_sha256=hashes,
                  same_operand_local_repair_demonstrated=(not results['native']['target_output_exact'] and results['batch_invariant']['target_output_exact']),
                  scope='Counterfactual supported local operator intervention on captured native operands; original full-model paths unchanged',
                  globally_first_operator_attribution_complete=False, full_campaign_complete=False)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()

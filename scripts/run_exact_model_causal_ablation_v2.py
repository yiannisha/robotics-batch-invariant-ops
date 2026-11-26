"""Run the existing six-mode experiment through pinned exact model adapters.

Only experiment metadata is attached to the adapter instance. Numerical model,
adapter and operator sources are neither edited nor monkeypatched. This wrapper
adds a retained provenance sidecar; it does not independently accept results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


ADAPTERS = {
    'pi05': 'pi05_libero_exact', 'rdt': 'rdt_maniskill_exact',
    'cogact': 'cogact_simpler_exact', 'lingbot': 'lingbot_robotwin_exact',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def preflight(plan):
    require(plan['model'] in ADAPTERS, 'Only the four selected models are supported')
    adapter = 'scripts.model_invariance.adapters.' + ADAPTERS[plan['model']]
    require(plan['adapter'] == adapter, 'Foreign adapter')
    require(plan['source_sha256'], 'Missing source pins')
    repo = Path(__file__).resolve().parents[1]
    required_sources = [
        Path(__file__).resolve(), repo / 'scripts/run_model_causal_ablation.py',
        repo / 'scripts/run_model_replay_invariance.py',
        repo / 'scripts/model_invariance/harness.py',
        repo / ('scripts/model_invariance/adapters/' + ADAPTERS[plan['model']] + '.py'),
        repo / 'batch_invariant_ops/__init__.py',
        repo / 'batch_invariant_ops/batch_invariant_ops.py',
    ]
    dependencies = {
        'pi05': ['scripts/verify_pi05_libero_checkpoint.py'],
        'rdt': ['scripts/model_invariance/adapters/rdt_maniskill.py', 'scripts/run_rdt_paired_server.py'],
        'cogact': ['scripts/model_invariance/adapters/cogact_small.py'],
        'lingbot': ['scripts/model_invariance/adapters/lingbot_vla2.py', 'batch_invariant_ops/moe.py'],
    }
    required_sources.extend(repo / p for p in dependencies[plan['model']])
    require(all(str(p) in plan['source_sha256'] for p in required_sources),
            'Missing required experiment source pin')
    for path, expected in plan['source_sha256'].items():
        require(sha(path) == expected, 'Pinned source changed: ' + path)
    manifest = json.loads(Path(plan['manifest']).read_text())
    require(sha(plan['manifest']) == plan['manifest_sha256'], 'Replay manifest changed')
    selected = [r for r in manifest['observations'] if r['split'] == 'diagnostic']
    require(len(selected) == 200 and all(len(r['request_ids']) == 3 for r in selected),
            'Requires complete 200-observation / 600-input-noise diagnostic cohort')
    require(len({i for r in selected for i in r['request_ids']}) == 600, 'Duplicate diagnostic request')
    observations = manifest['observations']
    for row in selected:
        companion_index = row['companion_replay_indices'][0]
        require(isinstance(companion_index, int) and 0 <= companion_index < len(observations),
                'Invalid companion index')
        companion = observations[companion_index]
        require(companion['split'] == 'diagnostic', 'Held-out companion exposure is forbidden')
        require((companion['task'], companion['episode_id']) != (row['task'], row['episode_id']),
                'Companion must come from another episode')
    trace = json.loads(Path(plan['trace']).read_text())
    require(sha(plan['trace']) == plan['trace_sha256'], 'Trace changed')
    require(trace['adapter'] == adapter and trace['batch_size'] == 8
            and trace['implementation'] == 'native', 'Wrong trace boundary')
    operations = {r['aten_operation'] for r in trace['operations']}
    convolution = any('conv' in op for op in operations)
    sdpa = any('scaled_dot_product' in op for op in operations)
    contracts = {}
    if plan['model'] != 'lingbot':
        require(convolution and sdpa, 'Expected convolution and SDPA interventions not observed')
    else:
        require(trace['replay_manifest_sha256'] == plan['manifest_sha256'],
                'Inapplicability requires the actual selected trace manifest')
        require(not convolution and not sdpa, 'LingBot intervention assumptions changed')
        source = Path(plan['eager_attention_source'])
        require(sha(source) == plan['source_sha256'].get(str(source)), 'Unpinned eager attention source')
        text = source.read_text()
        require('torch.einsum' in text and 'scaled_dot_product_attention' not in text,
                'Expected eager attention implementation changed')
        contracts = {
            'convolution': {
                'reason': 'No convolution dispatch in the pinned exact diagnostic execution trace; the patch intervention is inapplicable to this observed path.',
                'evidence_path': plan['trace'],
            },
            'explicit_sdpa': {
                'reason': 'Joint attention uses eager einsum; the per-matrix bmm reference remains enabled, while SDPA substitution is inapplicable.',
                'evidence_path': str(source),
            },
        }
    return adapter, trace['adapter_execution_configuration'], contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    plan_hash = sha(args.plan)
    plan = json.loads(args.plan.read_text())
    adapter_module, configuration, contracts = preflight(plan)
    if args.preflight_only:
        print(json.dumps(dict(status='preflight_passed_no_inference', adapter=adapter_module,
                              inapplicable_operations=contracts)), flush=True)
        return
    require(not args.output.exists(), 'Preserve existing measurements; choose a fresh output')
    sidecar = Path(str(args.output) + '.provenance.json')
    require(not sidecar.exists(), 'Preserve existing provenance')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status='starting', plan=str(args.plan.resolve()), plan_sha256=plan_hash,
                  source_sha256=plan['source_sha256'], manifest_sha256=plan['manifest_sha256'],
                  trace_sha256=plan['trace_sha256'],
                  trace_is_same_manifest=plan['manifest_sha256'] == json.loads(Path(plan['trace']).read_text())['replay_manifest_sha256'],
                  inapplicable_operations=contracts, independent_acceptance=False)
    def save(status, **extra):
        record.update(status=status, **extra)
        sidecar.write_text(json.dumps(record, indent=2) + '\n')
    save('starting')
    try:
        if __package__ is None or __package__ == '':
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        adapter = importlib.import_module(adapter_module).adapter
        require(Path(adapter.replay_manifest_path).resolve() == Path(plan['manifest']).resolve(),
                'Adapter environment selects another manifest')
        adapter.causal_ablation_inapplicable_operations = contracts
        import torch
        save('running_six_modes', torch_version=torch.__version__, cuda_version=torch.version.cuda,
             gpu_name=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        from scripts.run_model_causal_ablation import main as run
        sys.argv = ['run_model_causal_ablation', '--adapter', adapter_module,
                    '--manifest', plan['manifest'], '--output', str(args.output),
                    '--seed', str(plan['seed'])]
        run()
        with args.output.open() as stream:
            for line in stream:
                measurement = json.loads(line)
                require(measurement['adapter_execution_configuration'] == configuration,
                        'Recorded loaded inference configuration changed')
        preflight(plan)
        require(sha(args.plan) == plan_hash, 'Plan changed during execution')
        save('measurements_completed_independent_audit_pending', output_sha256=sha(args.output))
    except BaseException as error:
        save('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()

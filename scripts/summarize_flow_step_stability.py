"""Plot-ready denoising propagation from independently accepted diagnostics.

Select the target in the first slot of diverse-companion batches. Summaries
are task-specific and descriptive: noise replicas and multiple observations
are not treated as independent episodes for confidence intervals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(raw, audit_path):
    audit = json.loads(audit_path.read_text())
    require(audit['status'] == 'complete_four_model_diagnostic_independently_verified',
            'Complete independent diagnostic audit required')
    hashes = {str(raw): sha(raw), str(audit_path): sha(audit_path)}
    runtime_path = Path(str(raw) + '.runtime.json')
    hashes[str(runtime_path)] = sha(runtime_path)
    require(hashes[str(raw)] == audit['diagnostic_sha256'], 'Changed diagnostic')
    require(hashes[str(runtime_path)] == audit['runtime_sha256'], 'Changed runtime')
    runtime = json.loads(runtime_path.read_text())
    require(runtime['split'] == 'diagnostic', 'Diagnostic split required')
    labels = set(audit['required_denoising_labels'])
    require(labels, 'No required denoising labels')
    sizes = {int(x) for x in runtime['batch_sizes'] if int(x) > 1}
    modes = set(runtime['modes'])
    groups = defaultdict(list)
    roster = {}
    selected = set()
    records = 0
    with gzip.open(raw, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            records += 1
            require(row['replay_manifest_sha256'] == audit['manifest_sha256'],
                    'Foreign replay manifest')
            require(row['adapter_execution_configuration'] == audit['adapter_execution_configuration'],
                    'Foreign inference configuration')
            if not (row['transformation'] == 'batch_position'
                    and row['companion_type'] == 'diverse'
                    and row['target_batch_position'] == 'first'):
                continue
            mode, batch, request = row['implementation'], row['batch_size'], row['request_id']
            require(batch == row['logical_batch_size'] and batch in sizes and mode in modes,
                    'Foreign selected arrangement')
            identity = (str(row['task']), row['episode_id'], row['observation_index'],
                        row['noise_id'], row['request_noise_sha256'])
            require(request not in roster or roster[request] == identity, 'Request identity changed')
            roster[request] = identity
            key = (mode, batch, request)
            require(key not in selected, 'Duplicate selected arrangement')
            selected.add(key)
            require(labels <= row['flow_steps'].keys(), 'Missing denoising stage')
            for label in sorted(labels):
                metric = row['flow_steps'][label]
                require(set(metric['tensors']) == {'output'}, 'Unexpected stage tensor schema')
                tensor = metric['tensors']['output']
                count = math.prod(tensor['shape'])
                error, maximum = tensor['rmse'], tensor['max_absolute_error']
                require(count > 0 and tensor['nonfinite_count'] == 0, 'Invalid tensor')
                require(all(math.isfinite(v) and v >= 0 for v in (error, maximum)), 'Invalid error')
                exact = tensor['exact']
                require(exact == (tensor['unequal_element_count'] == 0), 'Inconsistent exactness')
                require(exact == (tensor['reference_sha256'] == tensor['candidate_sha256']),
                        'Inconsistent tensor hashes')
                require(not exact or (error == 0 and maximum == 0), 'Nonzero exact error')
                if mode == 'batch_invariant':
                    require(exact, 'Invariant propagation diverged')
                groups[str(row['task']), mode, batch, label].append(
                    (request, row['episode_id'], count, error, maximum, exact))
    require(records == audit['arrangement_records'], 'Incomplete arrangement stream')
    require(len(roster) == audit['request_noise_pairs'], 'Incomplete request coverage')
    require(selected == {(mode, batch, request) for mode in modes for batch in sizes
                         for request in roster}, 'Incomplete selected factorial')
    rows = []
    for (task, mode, batch, stage), values in sorted(groups.items()):
        errors = [v[3] for v in values]
        elements = sum(v[2] for v in values)
        mse = math.fsum(v[2] * v[3] ** 2 for v in values) / elements
        rows.append(dict(task=task, implementation=mode, batch_size=batch, stage=stage,
                         comparisons=len(values), episodes=len({v[1] for v in values}),
                         exact_comparisons=sum(v[5] for v in values),
                         exact_rate=sum(v[5] for v in values) / len(values),
                         median_comparison_rmse=quantile(errors, .5),
                         p95_comparison_rmse=quantile(errors, .95),
                         maximum_comparison_rmse=max(errors),
                         maximum_absolute_error=max(v[4] for v in values),
                         mean_squared_singleton_batch_difference=mse,
                         pooled_rms_singleton_batch_difference=math.sqrt(mse)))
    require(all(sha(Path(path)) == value for path, value in hashes.items()),
            'Source changed during extraction')
    return dict(status='descriptive_flow_propagation_extracted_independent_recount_pending',
                model_id=audit['model_id'], source_hashes=hashes,
                reporter_sha256=sha(Path(__file__)), request_noise_pairs=len(roster),
                selected_arrangements=len(selected), required_labels=sorted(labels),
                selection='diverse companions; target first; each non-singleton batch size',
                quantiles='linear interpolation at (n-1)*p over per-comparison RMSE',
                uncertainty='Descriptive only; no independent-episode confidence intervals claimed',
                metric='Paired deviation from singleton; not centered sample variance',
                full_campaign_complete=False, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'diagnostic-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new output; retain existing reports')
    result = summarize(args.raw, args.diagnostic_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f"Extracted {len(result['rows'])} stage groups; independent recount pending", flush=True)


if __name__ == '__main__':
    main()

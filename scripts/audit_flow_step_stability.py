"""Independently recount descriptive stage distributions from retained records."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Preserve existing audit')
    report_hash = digest(args.report)
    report = json.loads(args.report.read_text())
    sources = {Path(p): value for p, value in report['source_hashes'].items()}
    check(len(sources) == 3 and all(digest(p) == value for p, value in sources.items()),
          'Source evidence changed')
    raw = next(p for p in sources if p.name.endswith('.jsonl.gz'))
    audit_path = next(p for p in sources if not p.name.endswith(('.jsonl.gz', '.runtime.json')))
    audit = json.loads(audit_path.read_text())
    runtime = json.loads(Path(str(raw) + '.runtime.json').read_text())
    check(audit['status'] == 'complete_four_model_diagnostic_independently_verified', 'Unaccepted diagnostic')
    check(digest(raw) == audit['diagnostic_sha256'], 'Wrong raw input')
    check(digest(Path(str(raw) + '.runtime.json')) == audit['runtime_sha256'], 'Wrong runtime')
    labels = set(audit['required_denoising_labels'])
    buckets = defaultdict(list)
    identities = set()
    requests = set()
    records = 0
    with gzip.open(raw, 'rt') as stream:
        for line in stream:
            item = json.loads(line)
            records += 1
            if (item['transformation'], item['companion_type'], item['target_batch_position']) != (
                    'batch_position', 'diverse', 'first'):
                continue
            identity = (item['implementation'], item['batch_size'], item['request_id'])
            check(identity not in identities, 'Duplicate arrangement')
            identities.add(identity)
            requests.add(item['request_id'])
            check(labels <= item['flow_steps'].keys(), 'Missing stage')
            for label in labels:
                tensor = item['flow_steps'][label]['tensors']['output']
                check(tensor['nonfinite_count'] == 0, 'Nonfinite stage')
                check(tensor['exact'] == (tensor['unequal_element_count'] == 0)
                      == (tensor['reference_sha256'] == tensor['candidate_sha256']), 'Invalid exactness')
                key = (str(item['task']), item['implementation'], item['batch_size'], label)
                buckets[key].append((tensor, item['episode_id']))
    check(records == audit['arrangement_records'], 'Incomplete raw stream')
    check(len(requests) == audit['request_noise_pairs'] == report['request_noise_pairs'], 'Incomplete requests')
    expected = {(mode, b, r) for mode in runtime['modes'] for b in runtime['batch_sizes'] if b > 1 for r in requests}
    check(identities == expected and len(identities) == report['selected_arrangements'], 'Incomplete factorial')
    reported = {(r['task'], r['implementation'], r['batch_size'], r['stage']): r for r in report['rows']}
    check(len(reported) == len(report['rows']) and reported.keys() == buckets.keys(), 'Missing/duplicate table rows')
    for key, values in buckets.items():
        row = reported[key]
        tensors = [v[0] for v in values]
        errors = [v['rmse'] for v in tensors]
        check(all(math.isfinite(x) and x >= 0 for x in errors), 'Invalid RMSE')
        counts = [math.prod(t['shape']) for t in tensors]
        mse = sum(n * x * x for n, x in zip(counts, errors)) / sum(counts)
        exact = sum(t['exact'] for t in tensors)
        expected_values = dict(comparisons=len(values), episodes=len({v[1] for v in values}),
                               exact_comparisons=exact, exact_rate=exact / len(values),
                               median_comparison_rmse=statistics.median(errors),
                               p95_comparison_rmse=statistics.quantiles(errors, n=100, method='inclusive')[94],
                               maximum_comparison_rmse=max(errors),
                               maximum_absolute_error=max(t['max_absolute_error'] for t in tensors),
                               mean_squared_singleton_batch_difference=mse,
                               pooled_rms_singleton_batch_difference=math.sqrt(mse))
        for name, value in expected_values.items():
            check(math.isclose(row[name], value, rel_tol=1e-11, abs_tol=1e-20), f'{key}: {name} differs')
        if key[1] == 'batch_invariant':
            check(exact == len(values) and max(errors) == 0 and row['maximum_absolute_error'] == 0,
                  'Invariant stage drift')
    check(digest(args.report) == report_hash and all(digest(p) == h for p, h in sources.items()), 'Evidence changed')
    result = dict(status='descriptive_flow_stage_distributions_independently_recounted',
                  model_id=report['model_id'], report_sha256=report_hash, source_hashes=report['source_hashes'],
                  auditor_sha256=digest(Path(__file__)), verified_stage_groups=len(buckets),
                  request_noise_pairs=len(requests), scope='Descriptive task/mode/batch/stage distributions only',
                  causal_ablations_complete=False, confidence_intervals_verified=False, full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f'Independently recounted {len(buckets)} stage distributions', flush=True)


if __name__ == '__main__':
    main()

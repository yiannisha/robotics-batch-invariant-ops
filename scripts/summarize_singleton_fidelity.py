"""Descriptive BI-singleton/native-singleton fidelity, separate from batching."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def aggregate(items):
    values = [row['output']['tensors']['output.normalized'] for row in items]
    elements = [math.prod(value['shape']) for value in values]
    errors = [value['rmse'] for value in values]
    require(all(n > 0 for n in elements), 'Empty normalized output')
    for value in values:
        require(value['nonfinite_count'] == 0 and value['exact'] == (value['unequal_element_count'] == 0)
                and value['exact'] == (value['reference_sha256'] == value['candidate_sha256']), 'Invalid normalized exactness')
        require(all(math.isfinite(value[k]) and value[k] >= 0 for k in ('rmse', 'max_absolute_error')), 'Invalid fidelity error')
    mse = math.fsum(n * error * error for n, error in zip(elements, errors)) / sum(elements)
    return dict(comparisons=len(items), normalized_nonexact=sum(not v['exact'] for v in values),
                returned_output_nonexact=sum(not row['output']['exact'] for row in items),
                physical_output_nonexact=sum(not row['output']['tensors']['output.physical']['exact'] for row in items),
                normalized_elements=sum(elements), pooled_normalized_mse=mse, pooled_normalized_rms=math.sqrt(mse),
                median_normalized_comparison_rmse=statistics.median(errors),
                p95_normalized_comparison_rmse=statistics.quantiles(errors, n=100, method='inclusive')[94] if len(errors) > 1 else errors[0],
                maximum_normalized_absolute_error=max(v['max_absolute_error'] for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'diagnostic-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve prior report')
    raw_hash, audit_hash = sha(args.raw), sha(args.diagnostic_audit)
    audit = json.loads(args.diagnostic_audit.read_text())
    require(audit['status'] == 'complete_four_model_diagnostic_independently_verified'
            and audit['request_noise_pairs'] == audit['singleton_fidelity_records'] == 600
            and audit['fidelity_sha256'] == raw_hash, 'Complete accepted fidelity source required')
    rows, groups, seen = [], defaultdict(list), set()
    with gzip.open(args.raw, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            require(row['comparison'] == 'batch_invariant_singleton_vs_native_singleton'
                    and row['replay_manifest_sha256'] == audit['manifest_sha256']
                    and row['adapter_execution_configuration'] == audit['adapter_execution_configuration'], 'Foreign fidelity comparison')
            require(row['request_id'] not in seen, 'Duplicate fidelity request')
            seen.add(row['request_id'])
            rows.append(row)
            groups[str(row['task'])].append(row)
    require(len(rows) == 600 and sum(not row['output']['exact'] for row in rows) == audit['singleton_fidelity_nonexact_records'],
            'Incomplete or contradictory fidelity cohort')
    result = dict(status='singleton_fidelity_aggregates_extracted_independent_recount_pending',
                  model_id=audit['model_id'], source=str(args.raw.resolve()), source_sha256=raw_hash,
                  diagnostic_audit=str(args.diagnostic_audit.resolve()), diagnostic_audit_sha256=audit_hash,
                  reporter_sha256=sha(Path(__file__)), comparison='batch_invariant_singleton_vs_native_singleton',
                  configuration=audit['adapter_execution_configuration'], pooled=aggregate(rows),
                  tasks=[dict(task=task, **aggregate(items)) for task, items in sorted(groups.items())],
                  scope='Descriptive difference between implementations at B1; not singleton-versus-batched stability',
                  units='Normalized returned model action domain only. Physical components have mixed units and are not numerically pooled here.',
                  limitations='Controlled-native configuration; released-native padding/atomic-MoE supplements are separate. No centered sample-variance or independent-start CI claim.',
                  full_campaign_complete=False)
    require(sha(args.raw) == raw_hash and sha(args.diagnostic_audit) == audit_hash, 'Inputs changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f"Summarized600 singleton-fidelity comparisons for {audit['model_id']}; independent recount pending", flush=True)


if __name__ == '__main__':
    main()

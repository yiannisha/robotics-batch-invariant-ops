"""Independent recount of fidelity aggregate arithmetic, not model reexecution."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def verify(values, row):
    tensors = [v['output']['tensors']['output.normalized'] for v in values]
    counts = [math.prod(t['shape']) for t in tensors]
    errors = sorted(t['rmse'] for t in tensors)
    check(errors and all(math.isfinite(e) and e >= 0 for e in errors), 'Invalid source error')
    n = len(errors)
    def percentile(p):
        position = (n - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return errors[low] + (errors[high] - errors[low]) * (position - low)
    mse = sum(count * t['rmse'] ** 2 for count, t in zip(counts, tensors)) / sum(counts)
    expected = dict(comparisons=n, normalized_nonexact=sum(t['reference_sha256'] != t['candidate_sha256'] for t in tensors),
                    returned_output_nonexact=sum(not v['output']['exact'] for v in values),
                    physical_output_nonexact=sum(not v['output']['tensors']['output.physical']['exact'] for v in values),
                    normalized_elements=sum(counts), pooled_normalized_mse=mse, pooled_normalized_rms=math.sqrt(mse),
                    median_normalized_comparison_rmse=percentile(.5), p95_normalized_comparison_rmse=percentile(.95),
                    maximum_normalized_absolute_error=max(t['max_absolute_error'] for t in tensors))
    for field, value in expected.items():
        check(math.isfinite(row[field]) and math.isclose(row[field], value, rel_tol=1e-12, abs_tol=1e-20),
              'Fidelity aggregate differs: ' + field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Preserve prior audit')
    report_hash = sha(args.report)
    report = json.loads(args.report.read_text())
    raw, audit_path = Path(report['source']), Path(report['diagnostic_audit'])
    check(sha(raw) == report['source_sha256'] and sha(audit_path) == report['diagnostic_audit_sha256'], 'Source changed')
    acceptance = json.loads(audit_path.read_text())
    check(acceptance['status'] == 'complete_four_model_diagnostic_independently_verified'
          and acceptance['fidelity_sha256'] == sha(raw) and acceptance['request_noise_pairs'] == 600,
          'Complete fidelity source acceptance required')
    groups, rows, identities = defaultdict(list), [], set()
    with gzip.open(raw, 'rt') as stream:
        for line in stream:
            value = json.loads(line)
            check(value['request_id'] not in identities, 'Duplicate fidelity request')
            identities.add(value['request_id'])
            check(value['comparison'] == 'batch_invariant_singleton_vs_native_singleton'
                  and value['replay_manifest_sha256'] == acceptance['manifest_sha256']
                  and value['adapter_execution_configuration'] == acceptance['adapter_execution_configuration'], 'Foreign fidelity source')
            rows.append(value)
            groups[str(value['task'])].append(value)
    check(len(rows) == acceptance['singleton_fidelity_records'] == 600, 'Incomplete fidelity cohort')
    check(report['configuration'] == acceptance['adapter_execution_configuration'] and report['model_id'] == acceptance['model_id'], 'Report configuration differs')
    reported = {r['task']: r for r in report['tasks']}
    check(reported.keys() == groups.keys() and len(reported) == len(report['tasks']), 'Missing/duplicate fidelity task')
    verify(rows, report['pooled'])
    for task, values in groups.items():
        verify(values, reported[task])
    check(sha(args.report) == report_hash and sha(raw) == report['source_sha256']
          and sha(audit_path) == report['diagnostic_audit_sha256'], 'Sources changed during recount')
    result = dict(status='singleton_fidelity_aggregate_arithmetic_independently_recounted',
                  model_id=report['model_id'], report_sha256=report_hash, raw_sha256=sha(raw),
                  diagnostic_audit_sha256=sha(audit_path), auditor_sha256=sha(Path(__file__)),
                  comparisons=600, tasks=len(groups),
                  scope='Recount of producer-measured normalized B1 implementation differences. Not independent tensor-error rederivation or batching stability.',
                  full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f'Independently recounted600 fidelity records / {len(groups)} tasks', flush=True)


if __name__ == '__main__':
    main()

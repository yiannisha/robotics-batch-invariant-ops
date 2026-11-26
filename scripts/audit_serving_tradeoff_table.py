"""Independently recount serving figure values from raw queue requests."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(path):
    table = json.loads(path.read_text())
    raw, proof_path = Path(table['raw']), Path(table['performance_audit'])
    require(sha(raw) == table['raw_sha256'] and sha(proof_path) == table['performance_audit_sha256'], 'Changed queue/acceptance bytes')
    proof = json.loads(proof_path.read_text())
    require(proof['status'] == 'complete_isolated_four_model_performance_independently_verified'
            and proof['model_id'] == table['model_id']
            and proof['source_evidence_sha256'][str(raw.resolve())] == sha(raw), 'Foreign queue acceptance')
    groups, request_ids = defaultdict(list), set()
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['configuration'], row['offered_load']
            identity = (*key, row['request_id'])
            require(identity not in request_ids, 'Duplicate raw queue request')
            request_ids.add(identity)
            groups[key].append(row)
    modes = {'native_dynamic', 'batch_invariant_dynamic', 'serial_singleton', 'fixed_size_padded'}
    loads = {'underload', 'moderate', 'near_saturation', 'saturation'}
    require(set(groups) == {(mode, load) for mode in modes for load in loads}, 'Incomplete queue factorial')
    seen = set()
    for row in table['rows']:
        key = row['configuration'], row['offered_load']
        require(key in groups and key not in seen, 'Foreign or duplicate plotted group')
        seen.add(key)
        values = groups[key]
        count = len(values)
        require(count >= 250 and row['requests'] == count, 'Wrong plotted request count')
        exact = sum(v['numerical_contract'] == 'exact' for v in values)
        require(all(v['numerical_contract'] in ('exact', 'violated') for v in values), 'Unverified source request')
        require(row['exact_reference_requests'] == exact and row['violated_reference_requests'] == count - exact
                and row['all_requests_exact'] == (exact == count), 'Wrong plotted exactness')
        reference = 'fixed_padded_same_position' if key[0] == 'fixed_size_padded' else 'mode_matched_singleton'
        require(row['reference_type'] == reference and {v['reference_type'] for v in values} == {reference}, 'Numerical reference conflated')
        window = max(v['completion_time'] for v in values) - min(v['arrival_time'] for v in values)
        latency = np.array([v['arrival_to_completion_seconds'] for v in values], dtype=np.float64)
        require(window > 0 and np.isfinite(latency).all() and (latency > 0).all(), 'Invalid queue latency')
        rates = {v['offered_rate_requests_per_second'] for v in values}
        require(len(rates) == 1, 'Ambiguous load rate')
        expected = dict(observation_window_seconds=window, throughput_requests_per_second=count / window,
                        median_arrival_to_completion_seconds=float(np.median(latency)),
                        p95_arrival_to_completion_seconds=float(np.quantile(latency, .95, method='linear')),
                        maximum_absolute_difference=max(v['numerical_max_abs_difference'] for v in values),
                        offered_rate_requests_per_second=next(iter(rates)))
        require(all(math.isfinite(v) and math.isclose(row[k], v, rel_tol=1e-12, abs_tol=1e-20)
                    for k, v in expected.items()), 'Wrong serving plot arithmetic')
        require(row['unique_replay_fixtures'] == len({v['source_replay_index'] for v in values}), 'Wrong fixture count')
    require(seen == set(groups), 'Missing plotted queue group')
    for artifact in table['artifacts']:
        require(sha(artifact['path']) == artifact['sha256'], 'Changed plot/CSV artifact')
    csv_paths = [Path(a['path']) for a in table['artifacts'] if Path(a['path']).suffix == '.csv']
    require(len(csv_paths) == 1, 'Missing plot CSV')
    with csv_paths[0].open(newline='') as stream:
        reader = csv.DictReader(stream)
        csv_rows = list(reader)
        fields = reader.fieldnames
    require(len(csv_rows) == len(table['rows']) and set(fields) == set(table['rows'][0]), 'CSV schema/count differs')
    for actual, expected in zip(csv_rows, table['rows']):
        for key, value in expected.items():
            if isinstance(value, (bool, str)):
                require(actual[key] == str(value), 'CSV categorical value differs')
            elif isinstance(value, int):
                require(int(actual[key]) == value, 'CSV integer differs')
            else:
                require(float(actual[key]) == value, 'CSV float differs')
    require(sha(raw) == table['raw_sha256'], 'Queue source changed during recount')
    return dict(status='serving_tradeoff_table_independently_recounted', table_sha256=sha(path),
                model_id=table['model_id'], groups_verified=len(seen), requests_verified=len(request_ids),
                auditor_sha256=sha(__file__), visual_inspection_complete=False,
                full_campaign_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.table)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

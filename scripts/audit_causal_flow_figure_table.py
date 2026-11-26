"""Independently recount plotted causal distributions from retained metrics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(table_path):
    table = json.loads(table_path.read_text())
    raw, proof = Path(table['raw']), Path(table['audit'])
    if sha(raw) != table['raw_sha256'] or sha(proof) != table['audit_sha256']:
        raise ValueError('Figure source hash differs')
    acceptance = json.loads(proof.read_text())
    if (acceptance['status'] != 'complete_six_mode_causal_measurement_contracts_verified'
            or acceptance['raw_sha256'] != table['raw_sha256'] or acceptance['model'] != table['model']):
        raise ValueError('Wrong causal acceptance')
    groups = defaultdict(list)
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            for label, error in row['flow_steps'].items():
                if re.fullmatch(r'flow\.step_\d+\.[A-Za-z_]+', label):
                    groups[str(row['task']), row['mode'], label].append((error['rmse'], error['exact']))
    observed = set()
    for row in table['rows']:
        key = row['task'], row['mode'], row['stage']
        if key in observed or key not in groups:
            raise ValueError('Duplicate or foreign plotted group')
        observed.add(key)
        values = groups[key]
        if row['comparisons'] != len(values) or row['exact_comparisons'] != sum(x[1] for x in values):
            raise ValueError('Wrong plotted counts')
        errors = np.array([x[0] for x in values], dtype=np.float64)
        expected = [np.median(errors), np.quantile(errors, .95, method='linear'), errors.max()]
        actual = [row['median_rmse'], row['p95_rmse'], row['max_rmse']]
        if not np.isfinite(errors).all() or not np.allclose(actual, expected, rtol=1e-12, atol=1e-20):
            raise ValueError('Wrong plotted distribution')
    if observed != set(groups):
        raise ValueError('Missing plotted group')
    for artifact in table['artifacts']:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Changed figure artifact')
    if sha(raw) != table['raw_sha256']:
        raise ValueError('Raw measurements changed during recount')
    return dict(status='causal_flow_figure_table_independently_recounted',
                table_sha256=sha(table_path), groups_verified=len(groups),
                auditor_sha256=sha(__file__), visual_inspection_complete=False,
                tensor_errors_independently_rederived=False, full_campaign_complete=False)


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

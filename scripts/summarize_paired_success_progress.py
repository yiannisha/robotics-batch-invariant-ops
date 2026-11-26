"""Task-specific paired statistics from independently audited complete blocks.

The 10,000-resample percentile bootstrap matches pi-zero analyze_simpler.py.
Repeats verify determinism and never enter the number of independent trials.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CONDITIONS=('native_singleton','native_dynamic','patched_singleton','patched_dynamic')
COMPARISONS=(('patched_singleton','native_singleton'),('patched_dynamic','native_dynamic'),
             ('native_dynamic','native_singleton'),('patched_dynamic','patched_singleton'))


def task_statistics(blocks,seed):
    if not blocks:raise ValueError('No complete paired starts')
    identities=[(row['task'],row['initialization']) for row in blocks]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate paired task/start')
    if any(type(row['outcomes'].get(name)) is not bool for row in blocks for name in CONDITIONS):raise ValueError('Missing/nonboolean outcome')
    values={name:np.asarray([row['outcomes'][name] for row in blocks],dtype=np.int64) for name in CONDITIONS}
    n=len(blocks);comparisons={}
    for candidate,reference in COMPARISONS:
        differences=values[candidate]-values[reference]
        rng=np.random.default_rng(seed);estimates=np.empty(10000)
        for index in range(10000):estimates[index]=differences[rng.integers(0,n,n)].mean()
        gains=int(np.sum(differences==1));losses=int(np.sum(differences==-1));discordant=gains+losses
        p_value=1.0 if discordant==0 else min(1.0,2*sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))/(2**discordant))
        comparisons[candidate+'_minus_'+reference]=dict(estimate_percentage_points=100*float(differences.mean()),
            paired_bootstrap_ci95_percentage_points=[100*float(v) for v in np.quantile(estimates,[0.025,0.975])],
            bootstrap_seed=seed,bootstrap_resamples=10000,candidate_only_successes=gains,reference_only_successes=losses,
            both_successes=int(np.sum((values[candidate]==1)&(values[reference]==1))),
            both_failures=int(np.sum((values[candidate]==0)&(values[reference]==0))),
            exact_two_sided_mcnemar_p=p_value)
    return dict(paired_starts=n,unit='paired_task_initialization',repeats_counted_as_trials=False,
        successes={name:int(value.sum()) for name,value in values.items()},
        success_rate={name:float(value.mean()) for name,value in values.items()},comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    audit_bytes=args.audit.read_bytes();audit=json.loads(audit_bytes)
    audit_hash=hashlib.sha256(audit_bytes).hexdigest()
    if audit.get('status')!='paired_rollouts_independently_audited_provisional_pending_numerical_QA' or not audit.get('all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks') or not audit.get('reconstructed_policy_calls',0):raise ValueError('Independently audited raw-input reconstruction required')
    blocks=audit['blocks'];tasks=sorted({row['task'] for row in blocks})
    if len(blocks)!=audit['completed_paired_starts']:raise ValueError('Audit block count differs')
    rows={str(task):task_statistics([row for row in blocks if row['task']==task],20250401+index) for index,task in enumerate(tasks)}
    audit_snapshot=args.output.parent/'audit_snapshots'/(audit_hash+'.json')
    audit_snapshot.parent.mkdir(parents=True,exist_ok=True)
    if audit_snapshot.exists() and audit_snapshot.read_bytes()!=audit_bytes:raise ValueError('Audit snapshot differs')
    if not audit_snapshot.exists():audit_snapshot.write_bytes(audit_bytes)
    report=dict(status='provisional_task_specific_paired_statistics',completed_paired_starts=len(blocks),
        prescribed_paired_starts=audit['prescribed_paired_starts'],by_task=rows,
        bootstrap_method='pi_zero_analyze_simpler_percentile_paired_episode_bootstrap',
        bootstrap_limitations='Empirical paired resampling cannot represent discordance types absent from the observed sample; small samples require caution.',
        numerical_qa_acceptance_pending=True,success_rate_campaign_complete=False,
        source_audit=str(args.audit.resolve()),source_audit_sha256=audit_hash,source_audit_snapshot=str(audit_snapshot.resolve()),
        statistics_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

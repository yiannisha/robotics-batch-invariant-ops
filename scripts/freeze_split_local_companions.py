"""Replace a corpus schedule with independently checked split-local companions.

Input bytes/noises/episode membership are preserved from the parent corpus
audit. This produces a new manifest identity; previous inference runs cannot
be resumed against it or treated as held-out evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.run_model_replay_invariance import split_local_companion_rows


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,required=True)
    args=p.parse_args()
    if args.output_directory.exists():raise FileExistsError(args.output_directory)
    m=json.loads(args.manifest.read_text());parent=Path(m['corpus_audit_path'])
    require(sha(parent)==m['corpus_audit_sha256'],'Parent corpus audit changed')
    require(json.loads(parent.read_text())['status']=='complete_corpus_independently_verified','Parent corpus unverified')
    rows=m['observations'];require(len(rows)==1000,'Incomplete corpus')
    by_index={r['replay_index']:r for r in rows}
    require(set(by_index)==set(range(1000)),'Invalid replay indices')
    membership={};counts=Counter();distinct=Counter()
    parent_input_fingerprint=hashlib.sha256(json.dumps([{k:v for k,v in row.items() if k!='companion_replay_indices'} for row in rows],sort_keys=True).encode()).hexdigest()
    for row in rows:
        key=(row['task'],row['episode_id']);split=row['split']
        require(split in ('diagnostic','heldout'),'Unknown split')
        require(membership.setdefault(key,split)==split,'Episode split leakage')
        counts[split]+=1
        row['companion_replay_indices']=[r['replay_index'] for r in split_local_companion_rows(rows,row,row['replay_index'],31)]
    # Check properties independently of the selector implementation.
    for row in rows:
        indexes=row['companion_replay_indices'];require(len(indexes)==31 and len(set(indexes))==31,'Duplicate/missing companion')
        companion_episodes=set()
        for i in indexes:
            other=by_index[i]
            require(other['split']==row['split'],'Cross-split companion leak')
            key=(other['task'],other['episode_id'])
            require(key!=(row['task'],row['episode_id']),'Target-episode companion leak')
            companion_episodes.add(key)
        expected=min(31,len({key for key,split in membership.items() if split==row['split']})-1)
        require(len(companion_episodes)==expected,'Did not maximize distinct companion episodes')
        distinct[(row['split'],len(companion_episodes))]+=1
    require(dict(counts)=={'diagnostic':200,'heldout':800},'Split counts differ')
    fingerprint=hashlib.sha256(json.dumps([{k:v for k,v in row.items() if k!='companion_replay_indices'} for row in rows],sort_keys=True).encode()).hexdigest()
    require(fingerprint==parent_input_fingerprint,'Input/noise/episode identity changed')
    args.output_directory.mkdir(parents=True)
    report=dict(status='complete_corpus_independently_verified',scope='split_local_schedule_revision_over_verified_raw_corpus',
                parent_manifest_path=str(args.manifest.resolve()),parent_manifest_sha256=sha(args.manifest),
                parent_corpus_audit_path=str(parent.resolve()),parent_corpus_audit_sha256=sha(parent),
                input_noise_episode_fingerprint=fingerprint,all_input_noise_episode_fields_unchanged=True,
                all_companions_same_split=True,all_companions_outside_target_episode=True,
                companion_distinct_episode_coverage=[dict(split=k[0],distinct_episodes=k[1],observations=v) for k,v in sorted(distinct.items())],
                interpretation='Diagnostic B32 uses19 distinct non-target episodes and31 distinct observations. Heldout B32 uses31 distinct non-target episodes.',
                paired_success_rate_campaign_complete=False,auditor_sha256=sha(Path(__file__)))
    audit=args.output_directory/'corpus_audit.json';audit.write_text(json.dumps(report,indent=2)+'\n')
    m['corpus_audit_path']=str(audit.resolve());m['corpus_audit_sha256']=sha(audit)
    m['companion_schedule']=dict(version='split_local_unrelated_observations_v1',max_companions=31,
        description='Same split, no target episode; maximize distinct episodes, then use distinct other-episode observations.')
    (args.output_directory/'replay_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

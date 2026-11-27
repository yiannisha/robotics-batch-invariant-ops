"""Verify the completed partial derived audit without issuing a freshness seal."""
import pathlib,json,hashlib,time,datetime,shutil
r=pathlib.Path('/scratch/hadjiyiannis_3_gmail_com/lingbot-recovery-20261126');repo=pathlib.Path('/home/hadjiyiannis_3_gmail_com/batch-invariance-research/robotics-batch-invariant-ops');d=r/'replay/native_source_reselected_smoke001'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
while True:
 try:
  f=pathlib.Path('/proc/3472626/stat').read_text().rsplit(') ',1)[1].split()
  if f[19]!='51775050' or f[0]=='Z':break
 except FileNotFoundError:break
 time.sleep(5)
p=d/'independent_audit001.json';a=json.loads(p.read_text());launch=json.loads((r/'logs/native_source_reselected_smoke001_audit.launch.json').read_text())
assert a['status']=='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt' and a['completed_native_episodes']==64
assert a['raw_frames_verified']==20371 and a['model_calls_rebuilt']==436 and a['selected_inputs_rebuilt']==320
for key in ['corrected_precision_records_verified','official_cpu_transforms_only_no_neural_inference','derived_heldout_native_and_diagnostic_exclusion_verified','source_episodes_and_diagnostic_inputs_preserved']:assert a[key] is True
assert a['all_prescribed_native_episodes_rebuilt'] is False and a['final_historical_and_paired_freshness_pending'] is True
assert a['auditor_sha256']==sha(repo/'scripts/audit_lingbot_campaign_corpus_v3.py')==launch['auditor_sha256']
snapshot=pathlib.Path(a['manifest_snapshot']);assert sha(snapshot)==a['manifest_snapshot_sha256']==launch['manifest_sha256'];assert sha(d/'collection_manifest.json')==sha(snapshot)
m=json.loads(snapshot.read_text());derived=m['derived_selection'];source_audit=pathlib.Path(derived['source_audit_path']);source_snapshot=pathlib.Path(derived['source_manifest_snapshot'])
assert sha(source_audit)==a['original_source_audit_sha256']==derived['source_audit_sha256']=='b815818d9cfc86f18983b061d9b2ca96029589eec5d0136af72c533cb0ceb4e9'
assert sha(source_snapshot)==derived['source_manifest_sha256'];original=json.loads(source_snapshot.read_text());assert m['episodes']==original['episodes']
assert [x for x in m['observations'] if x['split']=='diagnostic']==[x for x in original['observations'] if x['split']=='diagnostic']
changed=sum(x!=y for x,y in zip(m['observations'],original['observations']));assert changed==7
assert sha(d/'candidate_lock.json')==a['candidate_lock_sha256']
for e in m['episodes']:assert sha(d/e['source_episode_file'])==e['sha256']
for file,expected in a['raw_files_sha256'].items():
 path=pathlib.Path(file);assert path.resolve().is_relative_to(d.resolve()) and sha(path)==expected
for file,expected in json.loads((d/'candidate_lock.json').read_text())['source_sha256'].items():assert sha(file)==expected and sha(d/'source_snapshots'/expected)==expected
keys=('images','img_masks','lang_tokens','lang_masks','state','image_grid_thw')
def ident(s):return hashlib.sha256(json.dumps({key:s[key] for key in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
import gzip
exposed=set()
for e in m['episodes']:
 with gzip.open(d/e['source_episode_file'],'rt') as stream:episode=json.load(stream)
 exposed.update(ident(call['target_input_signatures']) for call in episode['calls'])
exposed.update(ident(row['input_signatures']) for row in m['observations'] if row['split']=='diagnostic')
held=[ident(row['input_signatures']) for row in m['observations'] if row['split']=='heldout'];assert len(held)==len(set(held))==256 and not set(held)&exposed
record=dict(status='partial_64_episode_derived_selection_independently_rebuilt_and_raw_hashes_reverified',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),episodes=64,selected_inputs=320,heldout_inputs=256,changed_heldout_rows=7,source_episodes_and_diagnostic_inputs_preserved=True,native_and_diagnostic_collision_count=0,raw_file_count=len(a['raw_files_sha256']),all_audited_raw_hashes_reverified=True,audit_sha256=sha(p),manifest_snapshot_sha256=sha(snapshot),neural_inference_executed=False,full200_numerical_input_qualification_complete=False,final_historical_and_paired_freshness_pending=True,primary_paired_stability_measured=False,acceptance_script_sha256=sha(__file__))
ev=repo/'results/four_model/current_evidence';out=ev/'lingbot_recovery_20261127_reselected_smoke001_acceptance.json';audit_copy=ev/'lingbot_recovery_20261127_reselected_smoke001_independent_audit.json';assert not out.exists() and not audit_copy.exists();shutil.copyfile(p,audit_copy);out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)

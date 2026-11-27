"""Accept only the completed 64-episode audit, with every retained raw hash rechecked."""
import pathlib,json,hashlib,time,datetime,shutil
r=pathlib.Path('/scratch/hadjiyiannis_3_gmail_com/lingbot-recovery-20261126');repo=pathlib.Path('/home/hadjiyiannis_3_gmail_com/batch-invariance-research/robotics-batch-invariant-ops');c=r/'replay/native_source_v3_001'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def live():
 try:
  f=pathlib.Path('/proc/2391480/stat').read_text().rsplit(') ',1)[1].split();return f[19]=='51505342' and f[0]!='Z'
 except FileNotFoundError:return False
print('Waiting for original audit handle 2391480/51505342',flush=True)
while live():time.sleep(5)
p=c/'progress_audit007.json';a=json.loads(p.read_text());launch=json.loads((r/'logs/native_corpus_progress_audit007.launch.json').read_text());snapshot=pathlib.Path(a['manifest_snapshot']);m=json.loads(snapshot.read_text())
assert a['status']=='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt' and a['completed_native_episodes']==64 and len(m['episodes'])==64
assert a['corrected_precision_records_verified'] is True and a['official_cpu_transforms_only_no_neural_inference'] is True
assert a['all_prescribed_native_episodes_rebuilt'] is False and a['numerical_and_heldout_acceptance_complete'] is False
assert sha(snapshot)==a['manifest_snapshot_sha256']==launch['manifest_sha256']
assert a['auditor_sha256']==sha(repo/'scripts/audit_lingbot_campaign_corpus_v2.py')=='3cd1ef1c0762259d3ab0000557b098c88bee99025b07adfc5e966c4748207b9d'
assert sha(c/'candidate_lock.json')==a['candidate_lock_sha256']=='667d4a2eb5f670a71feec123d7dc46a203eeba38d3632d0bce810cd0a6fcc276'
current=json.loads((c/'collection_manifest.json').read_text());assert current['episodes'][:64]==m['episodes'];assert current['observations'][:len(m['observations'])]==m['observations']
prior=json.loads((c/'progress_audit006.json').read_text());old=json.loads(pathlib.Path(prior['manifest_snapshot']).read_text());assert m['episodes'][:44]==old['episodes']
for e in m['episodes']:assert sha(c/e['source_episode_file'])==e['sha256']
for file,expected in a['raw_files_sha256'].items():
 path=pathlib.Path(file);assert path.resolve().is_relative_to(c.resolve()) and sha(path)==expected
for file,expected in json.loads((c/'candidate_lock.json').read_text())['source_sha256'].items():assert sha(file)==expected and sha(c/'source_snapshots'/expected)==expected
ev=repo/'results/four_model/current_evidence';dest=ev/'lingbot_recovery_20261127_native_corpus_progress_audit007.json';receipt=ev/'lingbot_recovery_20261127_native_corpus_progress_acceptance006.json';assert not dest.exists() and not receipt.exists()
shutil.copyfile(p,dest)
record=dict(status='partial_native_corpus_audit_bound_raw_hashes_reverified',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_native_episodes=64,raw_frames_verified=a['raw_frames_verified'],model_calls_rebuilt=a['model_calls_rebuilt'],selected_inputs_rebuilt=a['selected_inputs_rebuilt'],audit_sha256=sha(p),manifest_snapshot_sha256=sha(snapshot),retained_raw_file_count=len(a['raw_files_sha256']),all_raw_file_hashes_reverified=True,append_only_current_manifest_prefix_verified=True,prior_44_episode_prefix_preserved=True,current_producer_source_pins_reverified=True,complete_200_episode_corpus_accepted=False,numerical_and_heldout_acceptance_complete=False,paired_stability_measured=False,acceptance_script_sha256=sha(__file__))
receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)

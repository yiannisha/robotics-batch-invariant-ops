import pathlib,json,hashlib,subprocess,os,time,datetime
r=pathlib.Path('/scratch/hadjiyiannis_3_gmail_com/lingbot-recovery-20261126');repo=pathlib.Path('/home/hadjiyiannis_3_gmail_com/batch-invariance-research/robotics-batch-invariant-ops');c=r/'replay/native_source_v3_001';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def handle(pid):
 try:return pathlib.Path(f'/proc/{pid}/stat').read_text().split()
 except FileNotFoundError:return None
s=handle(3676003);assert s and s[2]!='Z';audit_ticks=s[21]
print('Waiting for audit',3676003,audit_ticks,flush=True)
while True:
 s=handle(3676003)
 if not s or s[21]!=audit_ticks or s[2]=='Z':break
 time.sleep(5)
audit=c/'progress_audit004.json';a=json.loads(audit.read_text());pre=json.loads((r/'logs/native_source_v3_resume003.preflight.json').read_text());q=pathlib.Path(pre['qualification_snapshot'])
assert a['status']=='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt' and a['completed_native_episodes']==16 and a['corrected_precision_records_verified']
assert sha(c/'collection_manifest.json')==pre['manifest_sha256']==a['manifest_snapshot_sha256'];assert sha(a['manifest_snapshot'])==pre['manifest_sha256'];assert sha(q)==pre['qualification_snapshot_sha256'];assert sha(repo/'scripts/audit_lingbot_campaign_corpus_v2.py')==a['auditor_sha256'];assert not handle(2882319)
m=json.loads((c/'collection_manifest.json').read_text());lock=json.loads((c/'candidate_lock.json').read_text());assert sha(c/'candidate_lock.json')==m['candidate_lock_sha256']==a['candidate_lock_sha256']
assert {str((c/e['source_episode_file']).parent) for e in m['episodes']}=={str(p) for p in (c/'episodes').glob('*/*') if p.is_dir()}
for e in m['episodes']:assert sha(c/e['source_episode_file'])==e['sha256']
for p,h in lock['source_sha256'].items():assert sha(p)==h and sha(c/'source_snapshots'/h)==h
for p,h in a['raw_files_sha256'].items():assert sha(p)==h
for e in json.loads(q.read_text())['qualified_tasks'].values():
 d=pathlib.Path(e['directory']);assert sha(d/'independent_audit.json')==e['independent_audit_sha256'];assert sha(d/'replays.jsonl')==e['raw_replays_sha256']
 for p,h in json.loads((d/'runtime.json').read_text())['source_sha256'].items():assert sha(p)==h
receipt=r/'logs/native_source_v3_resume003.launch.json';assert not receipt.exists();old=json.loads((r/'logs/native_source_v3_resume002.launch.json').read_text());cmd=old['command'];cmd[cmd.index('--qualification-union')+1]=str(q)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-17594d78-b021-bba8-5fcd-7e09d2bdf8f9',OMP_NUM_THREADS='8',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_CACHE_HOME=str(r/'cache'),HF_HOME=str(r/'cache/huggingface'))
with open(r/'logs/native_source_v3_resume003.log','xb') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=handle(p.pid);assert s
receipt.write_text(json.dumps(dict(status='explicit_resume_after_terminal_collector_and_audited_sealed_episodes',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=p.pid,start_ticks=s[21],command=cmd,prior_collector_handle_terminal=True,prior_completed_episodes=16,bound_progress_audit_sha256=sha(audit),qualification_snapshot_sha256=sha(q),qualified_tasks=pre['qualified_tasks'],source_pins_reverified=True,no_unfinished_episode_directories=True,full_campaign_complete=False),indent=2)+'\n');print('LAUNCHED',p.pid,s[21],flush=True)

"""Append only after the original collector is terminal; audit its sealed snapshot concurrently."""
import pathlib,json,hashlib,subprocess,os,time,datetime,sys
r=pathlib.Path('/scratch/hadjiyiannis_3_gmail_com/lingbot-recovery-20261126');repo=pathlib.Path('/home/hadjiyiannis_3_gmail_com/batch-invariance-research/robotics-batch-invariant-ops');c=r/'replay/native_source_v3_001'
sys.path.insert(0,str(repo))
from scripts.lingbot_rigid_qualification import require_rigid_replay_proof
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def live(pid,ticks):
 try:
  f=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  return f[19]==str(ticks) and f[0]!='Z'
 except FileNotFoundError:return False
def launch(cmd,name,env):
 with (r/'logs'/(name+'.log')).open('xb') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 f=pathlib.Path(f'/proc/{p.pid}/stat').read_text().rsplit(') ',1)[1].split()
 return p.pid,f[19]
q=r/'logs/native_source_v3_resume006_qualification.json'
print('Waiting for collector 2391549/51505442',flush=True)
while live(2391549,51505442):time.sleep(5)
assert not q.exists();latest=max((r/'qualification').glob('rigid_qualified_union[0-9][0-9][0-9].json'));q.write_bytes(latest.read_bytes());qsha=sha(q)
assert not (r/'logs/native_source_v3_resume006.launch.json').exists()
m=json.loads((c/'collection_manifest.json').read_text());lock=json.loads((c/'candidate_lock.json').read_text())
assert len(m['episodes'])==92,'Prior segment did not complete all 92 expected sealed episodes'
assert sha(c/'candidate_lock.json')==m['candidate_lock_sha256']=='667d4a2eb5f670a71feec123d7dc46a203eeba38d3632d0bce810cd0a6fcc276'
assert {str((c/e['source_episode_file']).parent) for e in m['episodes']}=={str(p) for p in (c/'episodes').glob('*/*') if p.is_dir()},'Unfinished episode folders'
for e in m['episodes']:assert sha(c/e['source_episode_file'])==e['sha256']
for p,h in lock['source_sha256'].items():assert sha(p)==h and sha(c/'source_snapshots'/h)==h
assert sha(q)==qsha
qualification=json.loads(q.read_text());assert qualification['physics_capture_contract']=='rigid_components_v2_entities_v4' and qualification['prescribed_tasks']==lock['prescribed_tasks']
for entry in json.loads(q.read_text())['qualified_tasks'].values():
 d=pathlib.Path(entry['directory']);assert sha(d/'independent_audit.json')==entry['independent_audit_sha256'];proof=json.loads((d/'independent_audit.json').read_text());require_rigid_replay_proof(proof);assert sha(d/'replays.jsonl')==proof['raw_replays_sha256']
 for p,h in json.loads((d/'runtime.json').read_text())['source_sha256'].items():assert sha(p)==h
prior=json.loads((c/'progress_audit007.json').read_text());assert prior['completed_native_episodes']==64
assert sha(c/'progress_audit007.json')==json.loads((repo/'results/four_model/current_evidence/lingbot_recovery_20261127_native_corpus_progress_acceptance006.json').read_text())['audit_sha256']
for p,h in prior['raw_files_sha256'].items():assert sha(p)==h
snap=c/'corpus_audit_snapshots'/ (sha(c/'collection_manifest.json')+'.json');snap.parent.mkdir(exist_ok=True);snap.write_bytes((c/'collection_manifest.json').read_bytes())
pre=dict(status='sealed_segment_verified_resume_pending_independent_audit',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_completed_episodes=92,manifest_sha256=sha(snap),manifest_snapshot=str(snap),qualification_snapshot=str(q),qualification_snapshot_sha256=qsha,qualified_tasks=list(json.loads(q.read_text())['qualified_tasks']),prior_64_audited_raw_hashes_reverified=True,source_pins_reverified=True,no_unfinished_episode_directories=True)
(r/'logs/native_source_v3_resume006.preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-17594d78-b021-bba8-5fcd-7e09d2bdf8f9',OMP_NUM_THREADS='8',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_CACHE_HOME=str(r/'cache'),HF_HOME=str(r/'cache/huggingface'))
audit=json.loads((r/'logs/native_corpus_progress_audit007.launch.json').read_text())['command'];audit[audit.index('--output')+1]=str(c/'progress_audit008.json')
assert not pathlib.Path(audit[audit.index('--output')+1]).exists()
initial=snap.stat().st_mtime_ns
apid,aticks=launch(audit,'native_corpus_progress_audit008',env)
# Changed snapshot mtime acknowledges this auditor has read the sealed manifest.
for _ in range(120):
 if snap.stat().st_mtime_ns!=initial:break
 assert live(apid,aticks),'Auditor stopped before snapshot acknowledgment'
 time.sleep(1)
else:raise RuntimeError('Auditor snapshot acknowledgment timeout; no collector launched')
assert sha(snap)==pre['manifest_sha256']
(r/'logs/native_corpus_progress_audit008.launch.json').write_text(json.dumps(dict(pid=apid,start_ticks=aticks,command=audit,completed_episodes=92,manifest_sha256=sha(snap),snapshot_acknowledged_before_resume=True),indent=2)+'\n')
cmd=json.loads((r/'logs/native_source_v3_resume005.launch.json').read_text())['command'];cmd[cmd.index('--qualification-union')+1]=str(q)
pid,ticks=launch(cmd,'native_source_v3_resume006',env)
(r/'logs/native_source_v3_resume006.launch.json').write_text(json.dumps(dict(status='explicit_append_only_resume_while_sealed_snapshot_audit_runs',pid=pid,start_ticks=ticks,command=cmd,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_completed_episodes=92,prior_manifest_sha256=sha(snap),qualification_snapshot_sha256=qsha,qualified_tasks=pre['qualified_tasks'],audit_pid=apid,audit_start_ticks=aticks,sealed_episode_and_source_hashes_reverified=True,complete_corpus_audit_required_before_numerical_acceptance=True,full_model_campaign_complete=False),indent=2)+'\n')
print('Launched collector',pid,ticks,'audit',apid,aticks,flush=True)

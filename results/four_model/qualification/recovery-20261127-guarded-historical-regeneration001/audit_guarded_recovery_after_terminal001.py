"""Audit only the original guarded-recovery process after it terminates."""
import pathlib,json,time,subprocess
r=pathlib.Path('/scratch/hadjiyiannis_3_gmail_com/lingbot-recovery-20261126');repo=pathlib.Path('/home/hadjiyiannis_3_gmail_com/batch-invariance-research/robotics-batch-invariant-ops')
while True:
 try:
  f=pathlib.Path('/proc/350608/stat').read_text().rsplit(') ',1)[1].split()
  if f[19]!='52041438' or f[0]=='Z':break
 except FileNotFoundError:break
 time.sleep(5)
d=r/'history_recovery/native_guarded_tf32_on001';assert (d/'candidate_lock.json').exists(),'No recovery forward protocol sealed; inspect retained startup failure'
cmd=['/tmp/batch-invariance-archive-checks/bin/python','-m','scripts.audit_lingbot_guarded_recovery','--directory',str(d),'--output',str(d/'independent_guard_audit001.json')]
with (r/'logs/guarded_historical_regen001_audit.log').open('xb') as log:subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Independent guard audit complete; recovered raw bytes only, no primary inference/stability qualification',flush=True)

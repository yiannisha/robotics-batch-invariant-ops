"""Run corrected-adapter component cases with the versioned qualifier.

Run the complete actual-source linear sweep in isolated CUDA processes.

Each arrangement requires 100 measured seeds. Retry only explicit empty-CUPTI
capture failures in a new process, retaining every failed log. Other failures
stop the sweep. This runner does not certify completed measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','logs'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();config_sha=sha(a.config);config=json.loads(a.config.read_text())
    qualifier=Path(__file__).with_name('qualify_lingbot_exact_linear_ops_v2.py').resolve();source_sha=sha(qualifier)
    a.logs.mkdir(parents=True,exist_ok=False);(a.logs/(source_sha+'.py')).write_bytes(qualifier.read_bytes())
    (a.logs/'runner.py').write_bytes(Path(__file__).read_bytes())
    measured={}
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            r=json.loads(line)
            if r['configuration_sha256']!=config_sha or r['source_sha256'].get(str(qualifier))!=source_sha:
                raise ValueError('Existing raw output uses another source/configuration')
            seeds=measured.setdefault(r['arrangement'],set())
            if r['seed'] in seeds:raise ValueError('Duplicate existing seed')
            seeds.add(r['seed'])
    configurations=sorted(config['configurations'],key=lambda c:(0 if c['batch_axis']==1 else 1 if c['groups_per_request']==6 else 2,c['id']))
    keys=[f'{c["id"]}|B{b}|{position}' for c in configurations for b in (1,2,4,8,16,31,32,33,64)
        for position in (('first',) if b==1 else ('first','last'))]
    status=dict(configuration_sha256=config_sha,qualifier_source_sha256=source_sha,runner_source_sha256=sha(Path(__file__)),
        prescribed_arrangements=len(keys),already_complete=[key for key in keys if measured.get(key)==set(range(100))],
        attempts=[],full_operator_qualification_complete=False,shared_gpu_timing_not_qualified=True)
    for i,key in enumerate(keys):
        if measured.get(key)==set(range(100)):continue
        for attempt in range(1,4):
            if sha(a.config)!=config_sha or sha(qualifier)!=source_sha:raise ValueError('Source/configuration changed during sweep')
            command=[sys.executable,'-m','scripts.qualify_lingbot_exact_linear_ops','--config',str(a.config.resolve()),'--output',str(a.output.resolve()),'--resume','--arrangement-keys',key]
            log=a.logs/f'arrangement-{i:03d}-attempt-{attempt}.log'
            with log.open('w') as stream:
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=qualifier.parents[1])
                status['active']=dict(key=key,pid=proc.pid,attempt=attempt,log=str(log.resolve()))
                atomic(a.logs/'status.json',status);code=proc.wait()
            status['attempts'].append(dict(key=key,attempt=attempt,exit_code=code,log=str(log.resolve()),log_sha256=sha(log)))
            status.pop('active',None);atomic(a.logs/'status.json',status)
            print(f'{key}: attempt={attempt}, exit={code}',flush=True)
            if code==0:break
            if 'CUPTI captured no CUDA kernels after three attempts' not in log.read_text(errors='replace') or attempt==3:
                raise SystemExit(code)
    status['all_requested_processes_exited_successfully']=True;atomic(a.logs/'status.json',status)


if __name__=='__main__':main()

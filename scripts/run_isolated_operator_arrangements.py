"""Run requested 100-seed operator arrangements in separate CUDA processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','logs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arrangements',nargs='+',required=True);a=p.parse_args()
    qualifier=Path(__file__).with_name('qualify_real_shape_matrix_ops.py').resolve()
    config_sha=sha(a.config);source_sha=sha(qualifier);a.logs.mkdir(parents=True,exist_ok=False)
    (a.logs/(source_sha+'.py')).write_bytes(qualifier.read_bytes())
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            record=json.loads(line)
            if record['configuration_sha256']!=config_sha or record['qualifier_source_sha256']!=source_sha:
                raise ValueError('Existing raw measurements use another config or qualifier source')
    status=dict(config=str(a.config.resolve()),configuration_sha256=config_sha,qualifier_source_sha256=source_sha,
        runner_source_sha256=sha(Path(__file__)),shared_gpu_timing_not_qualified=True,arrangements=[])
    for i,key in enumerate(a.arrangements):
        if sha(a.config)!=config_sha or sha(qualifier)!=source_sha:raise ValueError('Qualification source changed between processes')
        command=[sys.executable,str(qualifier),'--config',str(a.config.resolve()),'--output',str(a.output.resolve()),
            '--resume','--arrangement-keys',key]
        log=a.logs/f'arrangement-{i:02d}.log'
        with log.open('w') as stream:code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
        status['arrangements'].append(dict(key=key,exit_code=code,log=str(log.resolve()),log_sha256=sha(log),command=command))
        (a.logs/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(f'{key}: exit={code}',flush=True)
        if code:raise SystemExit(code)
    status['all_requested_processes_exited_successfully']=True
    (a.logs/'status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()

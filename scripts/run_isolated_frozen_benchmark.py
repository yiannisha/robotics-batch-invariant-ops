"""Run a frozen benchmark with observed compute/graphics process isolation.

All observations and raw child logs are retained. Foreign GPU activity rejects
the entire run; only this wrapper's benchmark process group is terminated.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.gpu_isolation import observe,foreign_processes

SCRIPTS={'policy':'benchmark_model_policy','serving':'benchmark_model_serving','queue':'benchmark_model_request_queue'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def argument(argv,key):
    for i,value in enumerate(argv):
        if value==key and i+1<len(argv):return argv[i+1]
        if value.startswith(key+'='):return value.split('=',1)[1]
    raise ValueError('Required benchmark argument missing: '+key)


def validate_freeze(path,argv):
    freeze=json.loads(path.read_text());bound=freeze.get('content_addressed_freeze',{})
    if not bound or not freeze.get('prepared_input_disjointness_verified') or not freeze.get('invariant_diagnostic_exact'):raise ValueError('Complete four-model numerical freeze required')
    manifest=Path(argument(argv,'--replay-manifest')).resolve()
    if str(manifest) not in bound or sha(manifest)!=bound[str(manifest)]['sha256']:raise ValueError('Benchmark replay inputs not bound by freeze')
    runtime_files=[Path(p) for p in bound if p.endswith('invariance.jsonl.gz.runtime.json')]
    if len(runtime_files)!=1:raise ValueError('Frozen diagnostic runtime is ambiguous')
    runtime=json.loads(runtime_files[0].read_text())
    if argument(argv,'--adapter')!=runtime['adapter'] or runtime['execution_configuration']!=freeze['adapter_execution_configuration']:raise ValueError('Benchmark adapter differs from frozen numerical path')
    for filename,expected in bound.items():
        file=Path(filename)
        if not file.is_file() or file.stat().st_size!=expected['bytes'] or sha(file)!=expected['sha256']:raise ValueError('Frozen input changed: '+filename)
    return runtime


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def validate_calibration(timings,proof_path,freeze_sha,gpu_uuid):
    proof=json.loads(proof_path.read_text())
    if proof.get('accepted') is not True or proof.get('status')!='completed' or proof.get('benchmark')!='serving':
        raise ValueError('Queue requires accepted isolated serving calibration')
    if proof.get('freeze_sha256')!=freeze_sha or proof.get('gpu_uuid')!=gpu_uuid:
        raise ValueError('Serving calibration freeze/GPU differs')
    if proof.get('benchmark_output')!=str(timings.resolve()) or proof.get('benchmark_output_sha256')!=sha(timings):
        raise ValueError('Serving calibration bytes differ from isolated measurement')
    return {str(p.resolve()):sha(p) for p in (timings,proof_path)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',choices=SCRIPTS,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--observer-output',type=Path,required=True);p.add_argument('--child-log',type=Path,required=True)
    p.add_argument('--serving-isolation-proof',type=Path)
    a,argv=p.parse_known_args()
    if argv and argv[0]=='--':argv=argv[1:]
    if a.observer_output.exists() or a.child_log.exists():raise FileExistsError('Retain prior observer/log files; select new outputs')
    if '--resume' in argv or Path(argument(argv,'--output')).exists():raise ValueError('Isolated qualification requires a fresh benchmark output')
    if a.benchmark in ('policy','serving') and '--full-campaign' not in argv:raise ValueError('Full5-session/20-warmup/200-repetition protocol required')
    a.observer_output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='checking_gpu',accepted=False,gpu_uuid=a.gpu_uuid,freeze=str(a.freeze.resolve()),freeze_sha256=sha(a.freeze),
        benchmark=a.benchmark,poll_interval_seconds=1.0,isolation_scope='Compute/graphics process inventory sampled before, during and after execution; not an exclusive hardware reservation',observations=[],
        wrapper_source_sha256=sha(Path(__file__)),isolation_source_sha256=sha(Path(__file__).with_name('gpu_isolation.py')))
    initial=observe(a.gpu_uuid);report['observations'].append(initial)
    if foreign_processes(initial):
        report['status']='initial_gpu_busy';atomic(a.observer_output,report);raise SystemExit(2)
    runtime=validate_freeze(a.freeze,argv)
    if initial['gpu_name']!=runtime['gpu']:raise ValueError('Timing GPU hardware differs from frozen diagnostic hardware family')
    calibration={}
    if a.benchmark=='queue':
        if a.serving_isolation_proof is None:raise ValueError('Queue requires --serving-isolation-proof')
        calibration=validate_calibration(Path(argument(argv,'--serving-timings')),a.serving_isolation_proof,report['freeze_sha256'],a.gpu_uuid)
    sources=[Path(__file__),Path(__file__).with_name('gpu_isolation.py'),
        Path(__file__).with_name(SCRIPTS[a.benchmark]+'.py'),Path(__file__).parent/'model_invariance/harness.py']
    source_hashes={str(p.resolve()):sha(p) for p in sources}
    archive=a.observer_output.parent/(a.observer_output.stem+'.sources');archive.mkdir(exist_ok=False)
    for source in sources:(archive/source_hashes[str(source.resolve())]).write_bytes(source.read_bytes())
    report.update(benchmark_output=str(Path(argument(argv,'--output')).resolve()),benchmark_source_sha256=source_hashes,
        calibration_input_sha256=calibration,source_archive=str(archive.resolve()))
    command=[sys.executable,'-m','scripts.'+SCRIPTS[a.benchmark],*argv]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid);a.child_log.parent.mkdir(parents=True,exist_ok=True)
    report.update(status='running',command=command,started_at_utc=datetime.now(timezone.utc).isoformat())
    with a.child_log.open('x') as log:
        child=subprocess.Popen(command,env=environment,cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report['benchmark_pid']=child.pid;atomic(a.observer_output,report)
        try:
            while True:
                state=observe(a.gpu_uuid);report['observations'].append(state)
                if foreign_processes(state,allowed_pids=[child.pid],allowed_process_group=child.pid):
                    report['status']='foreign_gpu_activity';break
                code=child.poll()
                if code is not None:
                    report.update(status='completed' if code==0 else 'benchmark_failed',child_exit_code=code);break
                atomic(a.observer_output,report);time.sleep(1)
        except BaseException as error:
            report.update(status='observer_failed',error=repr(error));raise
        finally:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
            report['child_exit_code']=child.returncode
            if report['status']=='completed':
                try:
                    validate_freeze(a.freeze,argv)
                    if sha(a.freeze)!=report['freeze_sha256']:raise ValueError('Numerical freeze changed')
                    for filename,expected in {**source_hashes,**calibration}.items():
                        if sha(Path(filename))!=expected:raise ValueError('Benchmark source/calibration changed: '+filename)
                    report['benchmark_output_sha256']=sha(Path(report['benchmark_output']))
                    report['accepted']=True
                except Exception as error:report.update(status='frozen_inputs_changed',error=repr(error))
            report['finished_at_utc']=datetime.now(timezone.utc).isoformat();report['child_log_sha256']=sha(a.child_log)
            atomic(a.observer_output,report)
    print(report['status']);raise SystemExit(0 if report['accepted'] else 2)


if __name__=='__main__':main()

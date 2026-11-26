"""Observe compute AND graphics GPU processes for isolated timing qualification.

Uses the installed nvidia-smi XML interface; no package install, device setting,
process termination, clock adjustment, or model inference occurs here.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def parse_state(payload,expected_uuid):
    root=ET.fromstring(payload);devices=root.findall('gpu')
    if len(devices)!=1 or devices[0].findtext('uuid')!=expected_uuid:raise ValueError('GPU UUID observation is ambiguous or mismatched')
    gpu=devices[0];processes=gpu.find('processes')
    if processes is None or (processes.text or '').strip():raise ValueError('GPU process visibility unavailable')
    rows=[]
    for process in processes.findall('process_info'):
        pid=process.findtext('pid');kind=process.findtext('type')
        if not pid or not pid.isdigit() or not kind:raise ValueError('Unidentified GPU process')
        rows.append(dict(pid=int(pid),type=kind,used_memory=process.findtext('used_memory')))
    return dict(gpu_uuid=expected_uuid,gpu_name=gpu.findtext('product_name'),driver_version=root.findtext('driver_version'),
        memory_used=gpu.findtext('fb_memory_usage/used'),gpu_utilization=gpu.findtext('utilization/gpu_util'),temperature=gpu.findtext('temperature/gpu_temp'),
        sm_clock=gpu.findtext('clocks/sm_clock'),processes=rows,observed_at_utc=datetime.now(timezone.utc).isoformat())


def observe(gpu_uuid):
    if not gpu_uuid.startswith('GPU-'):raise ValueError('Select an explicit physical GPU UUID')
    result=subprocess.run(['nvidia-smi','-q','-x','-i',gpu_uuid],check=True,capture_output=True,text=True,timeout=15)
    return parse_state(result.stdout,gpu_uuid)


def foreign_processes(state,allowed_pids=(),allowed_process_group=None,group_resolver=os.getpgid):
    allowed=set(allowed_pids);foreign=[]
    for process in state['processes']:
        if process['pid'] in allowed:continue
        if allowed_process_group is not None:
            try:
                if group_resolver(process['pid'])==allowed_process_group:continue
            except (ProcessLookupError,PermissionError):pass
        foreign.append(process)
    return foreign


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    state=observe(a.gpu_uuid);foreign=foreign_processes(state)
    report=dict(status='idle_gpu_observed' if not foreign else 'busy_gpu_not_qualified_for_isolated_timing',observation=state,foreign_processes=foreign,
        observation_only=True,continuous_isolation_not_claimed=True,probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status']);raise SystemExit(0 if not foreign else 2)


if __name__=='__main__':main()

"""Extend the pinned pi-zero worker with observable-state evidence only.

The original snapshot/reset/step implementations are loaded unchanged. The
new protocol adds raw state retention and stricter canonical comparisons.
This module also supplies a small cross-Python client without policy imports.
"""
import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import os
from pathlib import Path
import pickle
import random
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qualify_simpler_fixed_action_replay import digest


def encode(value):
    if isinstance(value,(np.ndarray,np.generic)):return np.array(value,copy=True)
    if isinstance(value,float) and not np.isfinite(value):return np.asarray(value)
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    if isinstance(value,Path):return str(value)
    if dataclasses.is_dataclass(value):return {f.name:encode(getattr(value,f.name)) for f in dataclasses.fields(value)}
    cls=type(value)
    if cls.__module__.startswith('mani_skill2_real2sim.agents.configs.'):
        source=Path(sys.modules[cls.__module__].__file__).resolve()
        return dict(type=cls.__module__+'.'+cls.__name__,source=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fields=encode(vars(value)))
    if cls.__module__=='mani_skill2_real2sim.agents.robots.googlerobot' and cls.__name__ in ('GoogleRobotStaticBase','GoogleRobot'):
        return dict(type=cls.__module__+'.'+cls.__name__,robot_name=value.robot.get_name(),
                    state=encode(value.get_state()),controller_reference='same_frame.controller')
    if cls.__module__.startswith('sapien'):
        if cls.__name__=='Pose':return dict(type='sapien.Pose',position=np.array(value.p,copy=True),quaternion=np.array(value.q,copy=True))
        if hasattr(value,'get_name') and hasattr(value,'get_pose'):
            result=dict(type=cls.__module__+'.'+cls.__name__,name=value.get_name(),pose=encode(value.get_pose()))
            for name in ('get_velocity','get_angular_velocity','get_qpos','get_qvel','get_qacc','get_qf'):
                if hasattr(value,name):result[name]=encode(getattr(value,name)())
            if hasattr(value,'get_active_joints'):
                result['drives']=[dict(name=j.get_name(),position=encode(j.get_drive_target()),velocity=encode(j.get_drive_velocity_target())) for j in value.get_active_joints()]
            return result
    raise TypeError(f'Uncovered observable state: {cls.__module__}.{cls.__name__}')


def task_state(base):
    sources={};fields=set()
    for cls in type(base).__mro__:
        if not cls.__module__.startswith('mani_skill2_real2sim.envs.custom_scenes'):continue
        module=sys.modules[cls.__module__];path=Path(module.__file__).resolve();payload=path.read_bytes()
        sources[str(path)]=hashlib.sha256(payload).hexdigest()
        for node in ast.walk(ast.parse(payload.decode())):
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self' and isinstance(node.ctx,(ast.Store,ast.Del)):fields.add(node.attr)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('setattr','delattr') and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='self':
                if len(node.args)<2 or not isinstance(node.args[1],ast.Constant) or not isinstance(node.args[1].value,str):raise ValueError('Dynamic task mutation requires coverage')
                fields.add(node.args[1].value)
    result={};errors={}
    for name in sorted(fields):
        if hasattr(base,name):
            try:result[name]=dict(present=True,value=encode(getattr(base,name)))
            except TypeError as error:errors[name]=str(error)
        else:result[name]=dict(present=False)
    if errors:raise TypeError(f'Uncovered task fields: {errors}')
    return dict(source_sha256=sources,fields=result)


def controllers(controller):
    fields={};opaque={}
    for name,value in vars(controller).items():
        if name=='controllers':fields[name]={k:controllers(v) for k,v in value.items()};continue
        try:fields[name]=encode(value)
        except TypeError:opaque[name]=type(value).__module__+'.'+type(value).__name__
    return dict(type=type(controller).__module__+'.'+type(controller).__name__,fields=fields,
                opaque_handle_fields=opaque,public_state=encode(controller.get_state()))


def observable_state(environment,reference,observation=None):
    base=environment.unwrapped
    wrapper=environment;elapsed=[]
    while wrapper is not base:
        if '_elapsed_steps' in vars(wrapper):elapsed.append(vars(wrapper)['_elapsed_steps'])
        wrapper=wrapper.env
    raw=dict(observation=encode(reference._observe(environment) if observation is None else observation),
        scene_pack=encode(base._scene.pack()),physics=encode(base.get_state()),agent=encode(base.agent.get_state()),
        controller=controllers(base.agent.controller),task=task_state(base),elapsed_steps=base.elapsed_steps,wrapper_elapsed_steps=elapsed,
        rng=dict(main=encode(base._main_rng.get_state()),episode=encode(base._episode_rng.get_state()),
                 numpy=encode(np.random.get_state()),python=encode(random.getstate()),torch_used_in_simulator=False),
        instruction=environment.get_wrapper_attr('get_language_instruction')())
    return dict(raw=raw,hashes={key+'_sha256':digest(value) for key,value in raw.items()})


def serve():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-worker',type=Path,required=True);p.add_argument('--socket',type=Path,required=True);p.add_argument('--environment',required=True)
    p.add_argument('--failure-directory',type=Path,required=True)
    p.add_argument('--process-seed',type=int,default=0)
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('pinned_pizero_simpler_worker',args.reference_worker)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    original=reference._handle;canonical=None
    def handle(env,request):
        nonlocal canonical
        command=request['command']
        if command=='observable_state':return observable_state(env,reference),False
        if command=='step_with_state':
            result,finished=original(env,dict(request,command='step'))
            result['observable']=observable_state(env,reference,result['observation']);return result,finished
        result,finished=original(env,request)
        if command=='capture_snapshot':
            canonical=observable_state(env,reference,reference._snapshot['observation'])
            result['observable']=canonical
        elif command=='restore_snapshot':
            # The original controller/qacc setters perturb packed articulation
            # fields even when its public-state checks pass. Apply the common
            # scene pack last, then reobserve and compare every expanded field.
            env.unwrapped._scene.unpack(copy.deepcopy(reference._snapshot['scene_pack']))
            result['observation']=reference._observe(env)
            current=observable_state(env,reference,result['observation'])
            if canonical is None or current['hashes']!=canonical['hashes']:
                args.failure_directory.mkdir(parents=True,exist_ok=True)
                payload=args.failure_directory/'canonical-restore-mismatch.pkl'
                payload.write_bytes(pickle.dumps(dict(canonical=canonical,current=current),protocol=pickle.HIGHEST_PROTOCOL))
                changed=[key for key in current['hashes'] if canonical is None or current['hashes'][key]!=canonical['hashes'].get(key)]
                raise RuntimeError(f'Expanded canonical snapshot differs: {changed}; raw evidence: {payload}')
            result.update(observable=current,expanded_state_exact=True)
        elif command=='reset':canonical=None
        return result,finished
    reference._handle=handle
    random.seed(args.process_seed);np.random.seed(args.process_seed)
    sys.argv=[str(args.reference_worker),'--socket',str(args.socket),'--environment',args.environment]
    reference.main()


class RemoteStateEnv:
    def __init__(self,environment,python,simpler_root,reference_worker,vulkan_icd,log_path,process_seed=0):
        self.temporary=tempfile.TemporaryDirectory(prefix='cogact-state-')
        root=Path(self.temporary.name);runtime=root/'runtime';runtime.mkdir(mode=0o700);self.socket_path=root/'worker.sock'
        env=os.environ.copy();paths=[str(simpler_root),str(Path(simpler_root)/'ManiSkill2_real2sim'),str(Path(__file__).resolve().parents[1])]
        if env.get('PYTHONPATH'):paths.append(env['PYTHONPATH'])
        env.update(PYTHONPATH=os.pathsep.join(paths),VK_ICD_FILENAMES=str(vulkan_icd),XDG_RUNTIME_DIR=str(runtime),DISPLAY='',PYTHONHASHSEED='0')
        self.log=Path(log_path).open('a');self.connection=None
        self.process=subprocess.Popen([str(python),str(Path(__file__).resolve()),'--reference-worker',str(reference_worker),
            '--socket',str(self.socket_path),'--environment',environment,'--failure-directory',str(Path(log_path).with_suffix('.failures')),
            '--process-seed',str(process_seed)],env=env,stdout=self.log,stderr=subprocess.STDOUT)
        self.connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);deadline=time.monotonic()+120
        try:
            while True:
                if self.process.poll() is not None:raise RuntimeError(f'State worker exited {self.process.returncode}; inspect {log_path}')
                try:self.connection.connect(str(self.socket_path));break
                except (FileNotFoundError,ConnectionRefusedError):
                    if time.monotonic()>=deadline:raise TimeoutError('State worker startup timed out')
                    time.sleep(0.05)
        except BaseException:
            self.connection.close();self.connection=None
            if self.process.poll() is None:self.process.terminate()
            self.process.wait();self.log.close();self.temporary.cleanup();raise
    @staticmethod
    def receive(connection,size):
        chunks=[]
        while size:
            chunk=connection.recv(size)
            if not chunk:raise EOFError('SIMPLER worker disconnected')
            chunks.append(chunk);size-=len(chunk)
        return b''.join(chunks)
    def request(self,value):
        payload=pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL)
        self.connection.sendall(struct.pack('!Q',len(payload))+payload)
        size=struct.unpack('!Q',self.receive(self.connection,8))[0];response=pickle.loads(self.receive(self.connection,size))
        if not response['ok']:raise RuntimeError(response['error']+'\n'+response['traceback'])
        return response['result']
    def close(self):
        if self.connection is None:return
        try:self.request(dict(command='close'))
        finally:self.connection.close();self.connection=None;self.process.wait(timeout=30);self.log.close();self.temporary.cleanup()


if __name__=='__main__':serve()

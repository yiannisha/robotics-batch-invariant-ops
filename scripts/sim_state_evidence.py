"""Observable MuJoCo/controller/RNG evidence for canonical fresh-scene resets."""
import random

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def numeric(value):
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,np.generic):return np.asarray(value).copy()
    if isinstance(value,(float,int,bool)):return np.asarray(value)
    if value is None or isinstance(value,str):return value
    if isinstance(value,(list,tuple)):return [numeric(v) for v in value]
    if isinstance(value,dict) and all(isinstance(k,(str,int)) for k in value):return {str(k):numeric(v) for k,v in value.items()}
    raise TypeError(type(value).__name__)


def controller_snapshot(env):
    result={}
    for index,robot in enumerate(env._env.robots):
        controller=robot.controller;fields={};excluded=[]
        for name,value in vars(controller).items():
            try:fields[name]=numeric(value)
            except TypeError:excluded.append(name)
        result[str(index)]={'class':type(controller).__module__+'.'+type(controller).__name__,
                            'numeric_fields':fields,'opaque_handle_fields':sorted(excluded)}
    return result


def rng_snapshot(env):
    result={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().numpy(),
            'cuda_initialized':torch.cuda.is_initialized()}
    if torch.cuda.is_initialized():result['torch_cuda']=torch.cuda.get_rng_state().cpu().numpy()
    queue=[('wrapper',env)];seen=set();local={}
    while queue:
        prefix,obj=queue.pop()
        if id(obj) in seen:continue
        seen.add(id(obj))
        for name,value in vars(obj).items():
            if isinstance(value,np.random.Generator):local[prefix+'.'+name]=value.bit_generator.state
            elif isinstance(value,np.random.RandomState):local[prefix+'.'+name]=value.get_state()
            elif name in ('env','_env') and hasattr(value,'__dict__'):queue.append((prefix+'.'+name,value))
    result['environment_generators']=local
    return result


def observation_evidence(env,obs,*,include_rng=False):
    physical=np.asarray(env._env.sim.get_state().flatten()).copy()
    if not np.isfinite(physical).all():raise RuntimeError('Nonfinite simulator state')
    controller=controller_snapshot(env)
    controls=np.asarray(env._env.sim.data.ctrl).copy()
    record={'observation_sha256':digest(obs),'sim_state_sha256':digest(physical),'sim_state':physical.tolist(),
            'controller_state_sha256':digest(controller),'actuator_control_sha256':digest(controls),
            'eef':np.asarray(obs['robot_state']['eef']['pos']).tolist(),
            'gripper':np.asarray(obs['robot_state']['gripper']['qpos']).tolist()}
    snapshot={'physical':physical,'controller':controller,'actuator_controls':controls}
    if include_rng:
        snapshot['rng']=rng_snapshot(env);record['rng_state_sha256']=digest(snapshot['rng'])
        record['instruction']=env.task_description
    return record,snapshot

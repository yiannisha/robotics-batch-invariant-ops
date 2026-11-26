"""Observable ManiSkill task/controller/physics/RNG evidence for cold resets."""
import ast
import dataclasses
import hashlib
import inspect
from pathlib import Path
import random

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def data(value):
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().copy()
    if isinstance(value,(np.ndarray,np.generic)):return np.array(value,copy=True)
    if isinstance(value,dict):return {str(k):data(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [data(v) for v in value]
    if dataclasses.is_dataclass(value):return data(dataclasses.asdict(value))
    raise TypeError(type(value).__module__+'.'+type(value).__name__)


def task_value(value):
    try:return data(value)
    except TypeError:pass
    cls=type(value)
    if cls.__module__=='mani_skill.utils.structs.pose' and cls.__name__=='Pose':
        return {'type':'mani_skill.Pose','raw_pose':data(value.raw_pose)}
    if cls.__module__=='mani_skill.utils.structs.actor' and cls.__name__=='Actor':
        return {'type':'mani_skill.Actor','name':value.name,'state':data(value.get_state())}
    if cls.__module__=='mani_skill.utils.scene_builder.table.scene_builder' and cls.__name__=='TableSceneBuilder':
        return {'type':'TableSceneBuilder','environment_reference':'captured_separately',
                'fields':{k:task_value(v) for k,v in vars(value).items() if k!='env'}}
    if isinstance(value,(list,tuple)):return [task_value(v) for v in value]
    if isinstance(value,dict):return {str(k):task_value(v) for k,v in value.items()}
    raise TypeError(f'Uncovered task state: {cls.__module__}.{cls.__name__}')


def task_snapshot(env):
    path=Path(inspect.getsourcefile(type(env))).resolve();source=path.read_bytes()
    tree=ast.parse(source.decode());names=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self' and isinstance(node.ctx,(ast.Store,ast.Del)):
            names.add(node.attr)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('setattr','delattr') and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='self':
            if len(node.args)<2 or not isinstance(node.args[1],ast.Constant) or not isinstance(node.args[1].value,str):raise ValueError('Dynamic task attribute mutation is uncovered')
            names.add(node.args[1].value)
    return {'source':str(path),'source_sha256':hashlib.sha256(source).hexdigest(),
            'fields':{name:{'present':True,'value':task_value(getattr(env,name))} if hasattr(env,name) else {'present':False} for name in sorted(names)}}


def controller_snapshot(controller):
    fields={};opaque={}
    for name,value in vars(controller).items():
        if name=='controllers':
            fields[name]={k:controller_snapshot(v) for k,v in value.items()};continue
        try:fields[name]=data(value)
        except TypeError:opaque[name]=type(value).__module__+'.'+type(value).__name__
    return {'class':type(controller).__module__+'.'+type(controller).__name__,
            'numeric_fields':fields,'opaque_engine_handle_fields':opaque,'public_state':data(controller.get_state())}


def rng_snapshot(env):
    result={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().numpy(),
            'cuda_initialized':torch.cuda.is_initialized(),'environment':{}}
    if torch.cuda.is_initialized():result['torch_cuda']=torch.cuda.get_rng_state().cpu().numpy()
    objects=[('wrapper',env)];seen=set()
    while objects:
        label,obj=objects.pop()
        if id(obj) in seen:continue
        seen.add(id(obj))
        for name,value in vars(obj).items():
            key=label+'.'+name
            if type(value).__name__=='BatchedRNG' and type(value).__module__.startswith('mani_skill.'):
                result['environment'][key]=[rng.get_state() for rng in value.rngs]
            elif isinstance(value,np.random.RandomState):result['environment'][key]=value.get_state()
            elif isinstance(value,np.random.Generator):result['environment'][key]=value.bit_generator.state
            elif isinstance(value,torch.Generator):result['environment'][key]=value.get_state().cpu().numpy()
            elif name in ('env','_env') and hasattr(value,'__dict__'):objects.append((key,value))
    return result


def capture(env,observation,rgb,*,include_rng=True):
    root=env.unwrapped
    raw={'observation':data(observation),'policy_rgb':data(rgb),'physics':data(root.get_state_dict()),
         'controller':controller_snapshot(root.agent.controller),'task':task_snapshot(root),
         'drives':{'position':data(root.agent.robot.drive_targets),'velocity':data(root.agent.robot.drive_velocities)},
         'elapsed_steps':data(root.elapsed_steps)}
    if include_rng:raw['rng']=rng_snapshot(env)
    return {key+'_sha256':digest(value) for key,value in raw.items()},raw

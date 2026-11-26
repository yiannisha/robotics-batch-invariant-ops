"""Capture source-defined RoboTwin task fields; reject unsupported state.

Physics is captured separately. Actor references here retain stable names and
task metadata rather than process addresses. This is evidence for comparing
cold reconstruction, not an engine snapshot restoration API.
"""
import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch


def field_inventory(source):
    tree = ast.parse(source)
    assigned, referenced = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
            referenced.add(node.attr)
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('setattr', 'delattr'):
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'self':
                if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
                    raise ValueError('Dynamic task attribute mutation requires explicit coverage')
                assigned.add(node.args[1].value)
        if isinstance(node, ast.Attribute) and node.attr == '__dict__' and isinstance(node.value, ast.Name) and node.value.id == 'self':
            raise ValueError('Task __dict__ access requires explicit coverage')
    return {'assigned': sorted(assigned), 'referenced': sorted(referenced)}


def encode(value, path='task', active=None):
    if active is None:
        active = set()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f'Object array at {path}')
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    identity = id(value)
    if identity in active:
        raise ValueError(f'Cyclic state at {path}')
    active.add(identity)
    try:
        if isinstance(value, dict):
            if any(not isinstance(k, (str, int, bool)) for k in value):
                raise TypeError(f'Unsupported dictionary keys at {path}')
            return {k: encode(v, f'{path}.{k}', active) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {'sequence_type': type(value).__name__, 'values': [encode(v, f'{path}[{i}]', active) for i, v in enumerate(value)]}
        cls = type(value)
        if cls.__module__ == 'envs.utils.action' and cls.__name__ == 'ArmTag':
            return {'type': 'ArmTag', 'arm': value.arm}
        if cls.__module__ == 'envs.utils.action' and cls.__name__ == 'Action':
            fields = dict(vars(value))
            # These optional values live on the released class when unused by
            # a particular command. Include them explicitly as well as every
            # instance field, including planner kwargs.
            for name in ('arm_tag', 'action', 'target_pose', 'target_gripper_pos', 'args'):
                if name not in fields:
                    fields[name] = getattr(value, name)
            source_path = Path(inspect.getsourcefile(cls)).resolve()
            return {'type': 'robotwin.Action', 'source': str(source_path),
                    'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    'fields': {name: encode(item, f'{path}.{name}', active) for name, item in fields.items()}}
        if cls.__module__ == 'envs.utils.actor_utils' and cls.__name__ in ('Actor', 'ArticulationActor'):
            # Actor may acquire additional Python metadata after construction.
            # Capture every field except the engine handle, represented by name.
            return {'type': cls.__name__, 'name': value.get_name(),
                    'metadata': {k: encode(v, f'{path}.{k}', active) for k, v in vars(value).items() if k != 'actor'}}
        if cls.__module__.startswith('sapien') and cls.__name__ == 'Entity':
            scene=value.get_scene()
            if scene is None:raise TypeError(f'Detached entity at {path}')
            indexes=[i for i,item in enumerate(scene.get_entities()) if item is value]
            if len(indexes)!=1:raise ValueError(f'Entity scene reference is ambiguous at {path}')
            components=[]
            for component in value.get_components():
                kind=type(component).__name__
                if kind=='PhysxRigidDynamicComponent':
                    methods=('get_pose','get_linear_velocity','get_angular_velocity','get_mass','get_inertia',
                        'get_cmass_local_pose','get_linear_damping','get_angular_damping','get_disable_gravity',
                        'get_max_contact_impulse','get_max_depenetration_velocity','get_kinematic',
                        'get_locked_motion_axes','get_sleep_threshold','get_solver_position_iterations','get_solver_velocity_iterations')
                    body={name:encode(getattr(component,name)(),f'{path}.{kind}.{name}',active) for name in methods}
                    body['get_kinematic_target']=(encode(component.get_kinematic_target(),path+'.kinematic_target',active) if component.get_kinematic() else {'applicable':False})
                    shapes=[]
                    for shape in component.get_collision_shapes():
                        if type(shape).__name__!='PhysxCollisionShapeSphere':
                            raise TypeError(f'Uncovered entity collision shape at {path}: {type(shape).__name__}')
                        shape_methods=('get_radius','get_collision_groups','get_contact_offset','get_density',
                            'get_local_pose','get_min_patch_radius','get_patch_radius','get_rest_offset')
                        shape_state={name:encode(getattr(shape,name)(),f'{path}.sphere.{name}',active) for name in shape_methods}
                        material=shape.get_physical_material()
                        shape_state['material']={name:getattr(material,name)() for name in
                            ('get_static_friction','get_dynamic_friction','get_restitution')}
                        shapes.append(shape_state)
                    body['collision_shapes']=shapes;components.append({'type':kind,'state':body})
                elif kind=='RenderBodyComponent':
                    components.append({'type':kind,'pose':encode(component.get_pose(),path+'.render.pose',active)})
                else:raise TypeError(f'Uncovered entity component at {path}: {kind}')
            return {'type':'sapien.Entity','scene_entity_index':indexes[0],'name':value.get_name(),
                    'pose':encode(value.get_pose(),path+'.pose',active),'components':components}
        if cls.__module__.startswith('sapien') and cls.__name__ == 'Pose':
            return {'type': 'sapien.Pose', 'position': np.array(value.p, copy=True), 'quaternion': np.array(value.q, copy=True)}
        if cls.__module__.startswith('sapien') and cls.__name__ == 'PhysxArticulationLinkComponent':
            methods = ('get_name', 'get_index', 'get_pose', 'get_linear_velocity', 'get_angular_velocity',
                       'get_mass', 'get_inertia', 'get_cmass_local_pose', 'get_linear_damping',
                       'get_angular_damping', 'get_disable_gravity', 'get_max_contact_impulse',
                       'get_max_depenetration_velocity')
            return {'type': 'sapien.PhysxArticulationLinkComponent',
                    'state': {name: encode(getattr(value, name)(), f'{path}.{name}', active) for name in methods}}
        raise TypeError(f'Uncovered task state at {path}: {cls.__module__}.{cls.__name__}')
    finally:
        active.remove(identity)


def capture_task_state(env):
    source_path = Path(inspect.getsourcefile(type(env))).resolve()
    source = source_path.read_bytes()
    inventory = field_inventory(source.decode())
    fields = {}
    for name in inventory['assigned']:
        if not hasattr(env,name):
            fields[name]={'present':False};continue
        value=getattr(env,name)
        if inspect.ismethod(value):
            if value.__self__ is not env:raise TypeError('Task callback is bound to a different owner')
            function=value.__func__;path=Path(inspect.getsourcefile(function)).resolve()
            value=dict(type='bound_task_method',owner_reference='current_task_instance',
                owner_class=type(env).__module__+'.'+type(env).__qualname__,
                function_module=function.__module__,function_qualname=function.__qualname__,
                function_source=str(path),function_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        else:value=encode(value,'task.'+name)
        fields[name]={'present':True,'value':value}
    # Referenced inherited methods and engine objects remain visible in the
    # inventory. Their observable physics/controller state is covered by the
    # separate worker captures; this does not claim complete engine coverage.
    return {'schema': 'source_defined_task_fields_v1', 'source': str(source_path),
            'source_sha256': hashlib.sha256(source).hexdigest(), 'inventory': inventory, 'fields': fields}

"""Version 5: distinguish rigid Entity references and render-only targets.

Capture source-defined RoboTwin task fields; reject unsupported state.

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


def encode(value, path='task', active=None, owner=None):
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
        if inspect.ismethod(value):
            if owner is None or value.__self__ is not owner:
                raise TypeError(f'Foreign or unbound task callback at {path}')
            function = value.__func__
            if function.__closure__:
                raise TypeError(f'Closure-backed task callback at {path}')
            function_path = Path(inspect.getsourcefile(function)).resolve()
            owner_path = Path(inspect.getsourcefile(type(owner))).resolve()
            return {'type': 'task_bound_method',
                    'owner_class': type(owner).__module__ + '.' + type(owner).__qualname__,
                    'owner_source': str(owner_path),
                    'owner_source_sha256': hashlib.sha256(owner_path.read_bytes()).hexdigest(),
                    'function_qualname': function.__qualname__,
                    'function_module': function.__module__,
                    'function_source': str(function_path),
                    'function_source_sha256': hashlib.sha256(function_path.read_bytes()).hexdigest(),
                    'defaults': encode(function.__defaults__, path + '.defaults', active, owner),
                    'kwdefaults': encode(function.__kwdefaults__, path + '.kwdefaults', active, owner)}
        if isinstance(value, dict):
            if any(not isinstance(k, (str, int, bool)) for k in value):
                raise TypeError(f'Unsupported dictionary keys at {path}')
            return {k: encode(v, f'{path}.{k}', active, owner) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {'sequence_type': type(value).__name__, 'values': [encode(v, f'{path}[{i}]', active, owner) for i, v in enumerate(value)]}
        cls = type(value)
        if cls.__module__.startswith('sapien') and cls.__name__ == 'Entity':
            if owner is None:
                raise TypeError(f'Unscoped Entity reference at {path}')
            actors = owner.scene.get_all_actors()
            indices = [i for i, actor in enumerate(actors) if actor is value or actor == value]
            if not indices:
                components = value.get_components()
                if len(components) != 1 or type(components[0]).__name__ != 'RenderBodyComponent' or not type(components[0]).__module__.startswith('sapien.'):
                    raise TypeError(f'Entity reference is not a unique scene actor or covered visual at {path}')
                entities = owner.scene.get_entities()
                visual_indices = [i for i, entity in enumerate(entities) if entity is value or entity == value]
                if len(visual_indices) != 1:
                    raise TypeError(f'Visual Entity is not a unique scene entity at {path}')
                pose = value.get_pose()
                return {'type': 'sapien.VisualEntity', 'scene_entity_index': visual_indices[0],
                        'name': value.get_name(), 'position': np.array(pose.p, copy=True),
                        'quaternion': np.array(pose.q, copy=True),
                        'component_types': [type(components[0]).__module__ + '.RenderBodyComponent']}
            if len(indices) != 1:
                raise TypeError(f'Entity reference is not a unique scene actor at {path}')
            pose = value.get_pose()
            return {'type': 'sapien.Entity', 'actor_index': indices[0], 'name': value.get_name(),
                    'position': np.array(pose.p, copy=True), 'quaternion': np.array(pose.q, copy=True)}
        if cls.__module__ == 'envs.utils.action' and cls.__name__ == 'Action':
            if cls.__bases__ != (object,):
                raise TypeError(f'Unsupported Action inheritance at {path}')
            source = Path(inspect.getsourcefile(cls)).resolve()
            defaults = {k: encode(v, f'{path}.class.{k}', active, owner)
                        for k, v in vars(cls).items()
                        if not k.startswith('__') and not callable(v)}
            return {'type': 'robotwin.Action', 'source': str(source),
                    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'instance_fields': encode(vars(value), path + '.instance', active, owner),
                    'class_defaults': defaults}
        if cls.__module__ == 'envs.utils.action' and cls.__name__ == 'ArmTag':
            return {'type': 'ArmTag', 'arm': value.arm}
        if cls.__module__ == 'envs.utils.actor_utils' and cls.__name__ in ('Actor', 'ArticulationActor'):
            # Actor may acquire additional Python metadata after construction.
            # Capture every field except the engine handle, represented by name.
            return {'type': cls.__name__, 'name': value.get_name(),
                    'metadata': {k: encode(v, f'{path}.{k}', active, owner) for k, v in vars(value).items() if k != 'actor'}}
        if cls.__module__.startswith('sapien') and cls.__name__ == 'Pose':
            return {'type': 'sapien.Pose', 'position': np.array(value.p, copy=True), 'quaternion': np.array(value.q, copy=True)}
        if cls.__module__.startswith('sapien') and cls.__name__ == 'PhysxArticulationLinkComponent':
            methods = ('get_name', 'get_index', 'get_pose', 'get_linear_velocity', 'get_angular_velocity',
                       'get_mass', 'get_inertia', 'get_cmass_local_pose', 'get_linear_damping',
                       'get_angular_damping', 'get_disable_gravity', 'get_max_contact_impulse',
                       'get_max_depenetration_velocity')
            return {'type': 'sapien.PhysxArticulationLinkComponent',
                    'state': {name: encode(getattr(value, name)(), f'{path}.{name}', active, owner) for name in methods}}
        raise TypeError(f'Uncovered task state at {path}: {cls.__module__}.{cls.__name__}')
    finally:
        active.remove(identity)


def capture_task_state(env):
    source_path = Path(inspect.getsourcefile(type(env))).resolve()
    source = source_path.read_bytes()
    inventory = field_inventory(source.decode())
    fields = {}
    for name in inventory['assigned']:
        fields[name] = ({'present': True, 'value': encode(getattr(env, name), 'task.' + name, owner=env)}
                        if hasattr(env, name) else {'present': False})
    # Referenced inherited methods and engine objects remain visible in the
    # inventory. Their observable physics/controller state is covered by the
    # separate worker captures; this does not claim complete engine coverage.
    return {'schema': 'source_defined_task_fields_v1', 'source': str(source_path),
            'source_sha256': hashlib.sha256(source).hexdigest(), 'inventory': inventory, 'fields': fields}

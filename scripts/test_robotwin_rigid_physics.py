"""Entity velocities and stable references must survive independent coverage checks."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from scripts.probe_robotwin_exact_initialization_v3 import capture_state
from scripts.robotwin_task_state_v4 import encode
from scripts.audit_robotwin_benchmark_replay_v4 import verify_rigid_physics
from scripts.qualify_simpler_fixed_action_replay import digest


class PhysxRigidDynamicComponent:
    def __init__(self): self.velocity = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    def __getattr__(self, name):
        if name == 'get_linear_velocity':return lambda:self.velocity.copy()
        if name == 'get_angular_velocity':return lambda:np.array([0, 0, 0.4], dtype=np.float32)
        if name == 'get_cmass_local_pose':return lambda:SimpleNamespace(p=np.zeros(3), q=np.array([1, 0, 0, 0]))
        if name in ('get_kinematic', 'is_sleeping', 'get_disable_gravity', 'get_auto_compute_mass'):return lambda:False
        if name.startswith('get_'):return lambda:0.5
        raise AttributeError(name)


class Entity:
    def __init__(self):self.body = PhysxRigidDynamicComponent()
    def get_pose(self):return SimpleNamespace(p=np.array([1., 2., 3.]), q=np.array([1., 0., 0., 0.]))
    def get_name(self):return 'garbage'
    def get_components(self):return [self.body]

Entity.__module__ = 'sapien.pysapien'


class RigidPhysics(unittest.TestCase):
    def setUp(self):
        self.first, self.second = Entity(), Entity()
        self.scene = SimpleNamespace(get_all_actors=lambda:[self.first, self.second], get_all_articulations=lambda:[])
        self.env = SimpleNamespace(scene=self.scene)

    def raw(self):
        return dict(physics=capture_state(self.env), task_state={'fields': {'sphere': encode(self.second, owner=self.env)}})

    def test_velocity_changes_are_visible_without_pose_changes(self):
        first = capture_state(self.env)
        self.first.body.velocity[0] += 0.1
        second = capture_state(self.env)
        self.assertEqual(digest(first['actors'][0]['position']), digest(second['actors'][0]['position']))
        self.assertNotEqual(digest(first), digest(second))

    def test_duplicate_names_have_distinct_reference_indices(self):
        self.assertEqual(encode(self.first, owner=self.env)['actor_index'], 0)
        self.assertEqual(encode(self.second, owner=self.env)['actor_index'], 1)
        self.assertEqual(verify_rigid_physics(self.raw()), 1)

    def test_sleeping_property_is_captured_without_calling_bool(self):
        self.first.body.is_sleeping = True
        self.assertIs(capture_state(self.env)['actors'][0]['body_state']['is_sleeping'], True)

    def test_missing_velocity_or_legacy_schema_is_rejected(self):
        raw = self.raw()
        del raw['physics']['actors'][0]['body_state']['get_linear_velocity']
        with self.assertRaisesRegex(ValueError, 'coverage'):verify_rigid_physics(raw)
        raw = self.raw()
        del raw['physics']['schema']
        with self.assertRaisesRegex(ValueError, 'legacy'):verify_rigid_physics(raw)

    def test_reference_pose_or_index_tampering_is_rejected(self):
        for field, value in [('actor_index', 2), ('position', np.zeros(3))]:
            raw = self.raw()
            raw['task_state']['fields']['sphere'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):verify_rigid_physics(raw)

    def test_foreign_or_ambiguous_entities_are_rejected(self):
        with self.assertRaisesRegex(TypeError, 'unique'):encode(Entity(), owner=self.env)
        self.scene.get_all_actors = lambda:[self.first, self.first]
        with self.assertRaisesRegex(TypeError, 'unique'):encode(self.first, owner=self.env)
        with self.assertRaisesRegex(TypeError, 'Unscoped'):encode(self.first)


if __name__ == '__main__':unittest.main()

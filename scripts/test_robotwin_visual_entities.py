"""Render-only task targets are explicit state, never mislabeled as rigid actors."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from scripts.robotwin_task_state_v5 import encode
from scripts.audit_robotwin_benchmark_replay_v5 import verify_rigid_physics, verify_visual_entities
from scripts.test_robotwin_rigid_physics import Entity

RenderBodyComponent = type('RenderBodyComponent', (), {'__module__': 'sapien.pysapien.render'})


class VisualEntities(unittest.TestCase):
    def setUp(self):
        self.first, self.second = Entity(), Entity()
        self.first.body = RenderBodyComponent()
        self.second.body = RenderBodyComponent()
        self.env = SimpleNamespace(scene=SimpleNamespace(get_all_actors=lambda:[],
            get_entities=lambda:[self.first,self.second]))

    def raw(self):
        return dict(physics=dict(schema='robotwin_physics_rigid_components_v2',actors=[]),
                    task_state={'fields': {'target': encode(self.first, owner=self.env)}})

    def test_render_only_targets_have_distinct_scene_indices(self):
        first, second = encode(self.first, owner=self.env), encode(self.second, owner=self.env)
        self.assertEqual(first['type'], 'sapien.VisualEntity')
        self.assertEqual((first['scene_entity_index'],second['scene_entity_index']), (0,1))
        self.assertNotIn('actor_index', first)
        self.assertEqual(verify_visual_entities(self.raw()), 1)
        self.assertEqual(verify_rigid_physics(self.raw()), 0)

    def test_missing_pose_or_uncovered_component_is_rejected(self):
        for field, value in [('position', np.zeros(4)),
                             ('component_types', ['sapien.pysapien.physx.PhysxRigidDynamicComponent']),
                             ('scene_entity_index', -1)]:
            raw = self.raw()
            raw['task_state']['fields']['target'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):verify_visual_entities(raw)

    def test_aliases_cannot_disagree(self):
        raw = self.raw()
        alias = copy.deepcopy(raw['task_state']['fields']['target'])
        alias['position'][0] += 1
        raw['task_state']['fields']['alias'] = alias
        with self.assertRaisesRegex(ValueError, 'aliases'):verify_visual_entities(raw)

    def test_entity_must_be_unique_in_scene_and_have_only_render_component(self):
        self.env.scene.get_entities = lambda:[self.first,self.first]
        with self.assertRaisesRegex(TypeError, 'unique'):encode(self.first, owner=self.env)
        self.first.body = object()
        with self.assertRaisesRegex(TypeError, 'covered visual'):encode(self.first, owner=self.env)


if __name__ == '__main__':unittest.main()

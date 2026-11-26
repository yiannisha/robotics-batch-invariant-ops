import unittest
import numpy as np
from scripts.robotwin_task_state import encode, field_inventory


class TaskStateTests(unittest.TestCase):
    def test_mutable_subscript_and_late_fields(self):
        fields = field_inventory('''class Task:
 def load(self):
  self.targets = []
 def step(self):
  self.targets[0] = 3
  self.last_actor = self.bottle
''')
        self.assertEqual(fields['assigned'], ['last_actor', 'targets'])
        self.assertIn('bottle', fields['referenced'])

    def test_dynamic_mutation_rejected(self):
        with self.assertRaises(ValueError):
            field_inventory('setattr(self, key, 2)')
        self.assertEqual(field_inventory('setattr(self, "goal", 2)')['assigned'], ['goal'])

    def test_unknown_handles_rejected(self):
        with self.assertRaises(TypeError):
            encode({'actor': object()})

    def test_cycles_rejected(self):
        value = []; value.append(value)
        with self.assertRaises(ValueError):
            encode(value)

    def test_array_snapshot_does_not_alias(self):
        value = np.array([1., 2.], dtype=np.float32)
        saved = encode({'goal': value})
        value[0] = 9
        self.assertEqual(saved['goal'][0], 1.)
        self.assertEqual(saved['goal'].dtype, np.float32)

    def test_articulation_links_retain_numeric_state(self):
        methods = ('get_name', 'get_index', 'get_pose', 'get_linear_velocity', 'get_angular_velocity',
                   'get_mass', 'get_inertia', 'get_cmass_local_pose', 'get_linear_damping',
                   'get_angular_damping', 'get_disable_gravity', 'get_max_contact_impulse',
                   'get_max_depenetration_velocity')
        values = {name: i for i, name in enumerate(methods)}
        values['get_name'] = 'base'
        cls = type('PhysxArticulationLinkComponent', (), {
            '__module__': 'sapien.pysapien.physx',
            **{name: (lambda self, key=name: values[key]) for name in methods}})
        saved = encode(cls())
        self.assertEqual(saved['state'], values)


if __name__ == '__main__':
    unittest.main()

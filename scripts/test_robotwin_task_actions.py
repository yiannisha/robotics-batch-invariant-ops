"""Source-bound Action state must cover targets, arguments and class defaults."""
import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.robotwin_task_state_v3 import encode
from scripts.audit_robotwin_benchmark_replay_v3 import verify_task_actions
from scripts.qualify_simpler_fixed_action_replay import digest

SOURCE = '''class ArmTag:
 def __init__(self, arm): self.arm = arm
class Action:
 target_pose: list = None
 target_gripper_pos: float = None
 def __init__(self, arm, kind, target, **args):
  self.arm_tag = ArmTag(arm)
  self.action = kind
  if kind == 'move': self.target_pose = target
  else: self.target_gripper_pos = target
  self.args = args
'''


class ActionEvidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        utils = root / 'utils'
        utils.mkdir()
        self.source = utils / 'action.py'
        self.source.write_text(SOURCE)
        self.module = types.ModuleType('envs.utils.action')
        self.module.__file__ = str(self.source)
        self.patcher = patch.dict(sys.modules, {'envs.utils.action': self.module})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        exec(compile(SOURCE, str(self.source), 'exec'), self.module.__dict__)
        self.output = root / 'audit'
        self.output.mkdir()
        self.task_path = root / 'dump_bin_bigbin.py'
        self.task_path.write_text('class dump_bin_bigbin: pass\n')

    def task(self, value):
        return dict(source=str(self.task_path), fields={'pour_actions': {'value': value}})

    def test_distinct_instances_stable_and_arguments_and_targets_are_retained(self):
        args = dict(speed=np.array([0.1, -0.0], dtype=np.float32))
        first = encode(self.module.Action('left', 'move', [1, 2, 3], **args))
        second = encode(self.module.Action('left', 'move', [1, 2, 3], **args))
        self.assertEqual(digest(first), digest(second))
        self.assertEqual(first['class_defaults'], {'target_pose': None, 'target_gripper_pos': None})
        self.assertEqual(verify_task_actions(self.task(first), self.output, {}), 1)
        second['instance_fields']['args']['speed'][1] = 0.0
        self.assertNotEqual(digest(first), digest(second))

    def test_move_and_gripper_records_are_covered(self):
        values = encode([self.module.Action('left', 'move', [1, 2, 3]),
                         self.module.Action('right', 'gripper', 0.7)])
        self.assertEqual(verify_task_actions(self.task(values), self.output, {}), 2)

    def test_missing_target_argument_or_default_is_rejected(self):
        value = encode(self.module.Action('right', 'gripper', 0.7))
        for section, field in [('instance_fields', 'target_gripper_pos'),
                               ('instance_fields', 'args'), ('class_defaults', 'target_pose')]:
            changed = copy.deepcopy(value)
            del changed[section][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_task_actions(self.task(changed), self.output, {})

    def test_runtime_class_default_mutation_is_rejected(self):
        self.module.Action.target_pose = [1, 2, 3]
        value = encode(self.module.Action('right', 'gripper', 0.7))
        with self.assertRaisesRegex(ValueError, 'defaults'):
            verify_task_actions(self.task(value), self.output, {})

    def test_source_change_is_rejected(self):
        value = encode(self.module.Action('right', 'gripper', 0.7))
        self.source.write_text(SOURCE + '# changed\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            verify_task_actions(self.task(value), self.output, {})

    def test_cyclic_arguments_are_rejected(self):
        value = self.module.Action('left', 'move', [1, 2, 3])
        value.args['cycle'] = value
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            encode(value)


if __name__ == '__main__':
    unittest.main()

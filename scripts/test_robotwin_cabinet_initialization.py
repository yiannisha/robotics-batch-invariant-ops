"""Cold initialization must match upstream cabinet pre-motion assignments."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.robotwin_exact_worker_v2 import restore_task_initialization


class ArmTag:
    def __init__(self, arm):
        self.arm = arm


class CabinetInitialization(unittest.TestCase):
    def test_arm_selection_and_origin_height_match_upstream(self):
        package = ModuleType('envs')
        package.__path__ = []
        utils = ModuleType('envs.utils')
        utils.ArmTag = ArmTag
        package.utils = utils
        with patch.dict(sys.modules, {'envs': package, 'envs.utils': utils}):
            for x, arm in [(-0.1, 'left'), (0.1, 'right'), (0.0, 'left')]:
                with self.subTest(x=x):
                    position = np.array([x, -0.2, 0.73], dtype=np.float32)
                    actor = Mock()
                    actor.get_pose.return_value = SimpleNamespace(p=position)
                    env = SimpleNamespace(object=actor, origin_z=-99)
                    restore_task_initialization('put_object_cabinet', env)
                    self.assertEqual(env.arm_tag.arm, arm)
                    self.assertEqual(env.origin_z.tobytes(), position[2].tobytes())
                    self.assertIsInstance(env.origin_z, np.float32)

    def test_other_tasks_are_untouched(self):
        actor = Mock()
        env = SimpleNamespace(object=actor, arm_tag='original', origin_z=-99)
        restore_task_initialization('click_bell', env)
        actor.get_pose.assert_not_called()
        self.assertEqual((env.arm_tag, env.origin_z), ('original', -99))


if __name__ == '__main__':
    unittest.main()

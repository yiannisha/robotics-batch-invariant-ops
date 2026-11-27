"""Scale reconstruction needs the released pre-motion arm selection."""
import sys
import unittest
from types import ModuleType,SimpleNamespace
from unittest.mock import Mock,patch
from scripts.robotwin_exact_worker_v4 import restore_task_initialization


class ScaleInitialization(unittest.TestCase):
    def test_released_object_x_selects_arm_without_cabinet_fields(self):
        package=ModuleType('envs');package.__path__=[]
        utils=ModuleType('envs.utils');utils.ArmTag=lambda arm:SimpleNamespace(arm=arm)
        with patch.dict(sys.modules,{'envs':package,'envs.utils':utils}):
            for x,expected in [(-.1,'left'),(0.,'left'),(.1,'right')]:
                with self.subTest(x=x):
                    actor=Mock();actor.get_pose.return_value=SimpleNamespace(p=[x,0.,.7])
                    env=SimpleNamespace(object=actor)
                    restore_task_initialization('place_object_scale',env)
                    self.assertEqual(env.arm_tag.arm,expected)
                    self.assertEqual(set(vars(env)),{'object','arm_tag'})
                    actor.get_pose.assert_called_once_with()


if __name__=='__main__':unittest.main()

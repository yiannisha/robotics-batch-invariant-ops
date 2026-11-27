"""Laptop cold reconstruction must set the released pre-motion arm choice."""
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from scripts.robotwin_exact_worker_v3 import restore_task_initialization


class LaptopInitialization(unittest.TestCase):
    def test_released_face_product_selects_arm_before_capture(self):
        package=ModuleType('envs');package.__path__=[]
        utils=ModuleType('envs.utils');utils.ArmTag=lambda arm:SimpleNamespace(arm=arm)
        quaternion=[0.7,0,0,0.7]
        laptop=Mock();laptop.get_pose.return_value=SimpleNamespace(q=quaternion)
        with patch.dict(sys.modules,{'envs':package,'envs.utils':utils}):
            for value,expected in [(0.9,'left'),(0.,'right'),(-0.9,'right')]:
                with self.subTest(face_product=value):
                    utils.get_face_prod=Mock(return_value=value)
                    env=SimpleNamespace(laptop=laptop)
                    restore_task_initialization('open_laptop',env)
                    self.assertEqual(env.arm_tag.arm,expected)
                    utils.get_face_prod.assert_called_once_with(quaternion,[1,0,0],[1,0,0])
                    self.assertEqual(set(vars(env)),{'laptop','arm_tag'})


if __name__=='__main__':unittest.main()

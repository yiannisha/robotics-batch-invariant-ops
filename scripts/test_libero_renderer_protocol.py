from types import ModuleType,SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from scripts.libero_deterministic_renderer import install


class RendererProtocolTests(unittest.TestCase):
    def fixture(self,ignore=False):
        class Context:
            def __init__(self):
                self.model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))
            def _set_mujoco_context_and_buffers(self):
                self.observed_before_allocation=self.model.vis.quality.offsamples
                self.con=SimpleNamespace(offSamples=4 if ignore else self.model.vis.quality.offsamples)
        module=ModuleType('robosuite.utils.binding_utils');module.MjRenderContext=Context
        return Context,module

    def test_pins_samples_before_context_allocation(self):
        cls,module=self.fixture()
        with patch.dict(sys.modules,{'robosuite.utils.binding_utils':module}):install()
        context=cls();context._set_mujoco_context_and_buffers()
        self.assertEqual(context.observed_before_allocation,0);self.assertEqual(context.con.offSamples,0)

    def test_driver_ignoring_configuration_is_rejected(self):
        cls,module=self.fixture(ignore=True)
        with patch.dict(sys.modules,{'robosuite.utils.binding_utils':module}):install()
        with self.assertRaisesRegex(RuntimeError,'ignored'):cls()._set_mujoco_context_and_buffers()

    def test_installation_is_idempotent(self):
        cls,module=self.fixture()
        with patch.dict(sys.modules,{'robosuite.utils.binding_utils':module}):
            install();first=cls._set_mujoco_context_and_buffers;install()
            self.assertIs(first,cls._set_mujoco_context_and_buffers)


if __name__=='__main__':unittest.main()

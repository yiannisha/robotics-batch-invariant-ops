import dataclasses
import unittest
import numpy as np

from scripts.cogact_simpler_state_worker import encode,controllers


class StateEvidenceTests(unittest.TestCase):
    def test_arrays_keep_dtype_and_do_not_alias(self):
        original=np.array([1.,-0.],dtype=np.float32);saved=encode(original)
        original[0]=9
        self.assertEqual(saved.dtype,np.float32);self.assertEqual(saved[0],1.)
        self.assertTrue(np.signbit(saved[1]))

    def test_nonfinite_configuration_bounds_keep_bytes(self):
        value=encode(float('inf'))
        self.assertIsInstance(value,np.ndarray);self.assertTrue(np.isposinf(value))

    def test_unknown_task_object_is_rejected(self):
        with self.assertRaises(TypeError):encode(object())

    def test_controller_numeric_state_and_handle_inventory(self):
        class Controller:
            def __init__(self):self._step=7;self._target=np.array([0.1]);self.solver=object()
            def get_state(self):return {'target':self._target}
        record=controllers(Controller())
        self.assertEqual(record['fields']['_step'],7)
        self.assertIn('_target',record['fields']);self.assertEqual(record['opaque_handle_fields'],{'solver':'builtins.object'})

    def test_dataclass_configuration_recurses(self):
        @dataclasses.dataclass
        class Config:
            limit:float
            names:list
        self.assertEqual(encode(Config(3.,['a','b'])),{'limit':3.,'names':['a','b']})


if __name__=='__main__':unittest.main()

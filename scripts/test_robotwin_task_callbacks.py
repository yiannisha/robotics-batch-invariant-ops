"""Callback capture must distinguish selection without recording process addresses."""
import copy
import hashlib
from pathlib import Path
import tempfile
import types
import unittest

from scripts.robotwin_task_state_v2 import encode
from scripts.audit_robotwin_benchmark_replay_v2 import verify_task_callbacks


class Owner:
    def left(self):
        return True

    def right(self):
        return False


class CallbackCapture(unittest.TestCase):
    def test_identity_is_stable_across_reconstruction_but_distinguishes_methods(self):
        first, second = Owner(), Owner()
        self.assertEqual(encode(first.left, owner=first), encode(second.left, owner=second))
        self.assertNotEqual(encode(first.left, owner=first), encode(first.right, owner=first))

    def test_foreign_and_unscoped_callbacks_are_rejected(self):
        first, second = Owner(), Owner()
        with self.assertRaisesRegex(TypeError, 'Foreign'):
            encode(first.left, owner=second)
        with self.assertRaisesRegex(TypeError, 'unbound'):
            encode(first.left)

    def test_nested_callback_is_captured_and_mutable_closure_is_rejected(self):
        owner = Owner()
        self.assertEqual(encode({'callback': owner.left}, owner=owner)['callback']['type'],
                         'task_bound_method')
        state = []

        def callback(self):
            return state

        with self.assertRaisesRegex(TypeError, 'Closure'):
            encode(types.MethodType(callback, owner), owner=owner)


class CallbackAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.task_path = root / 'click_bell.py'
        self.task_path.write_text('class click_bell:\n def load(self):\n  self.check = self.left\n')
        self.method_path = root / '_base_task.py'
        self.method_path.write_text('class Base_Task:\n def left(self):\n  return True\n')
        self.output = root / 'audit'
        self.output.mkdir()
        self.value = dict(type='task_bound_method', owner_class='envs.click_bell.click_bell',
                          owner_source=str(self.task_path), owner_source_sha256=self.digest(self.task_path),
                          function_qualname='Base_Task.left', function_module='envs._base_task',
                          function_source=str(self.method_path),
                          function_source_sha256=self.digest(self.method_path),
                          defaults=None, kwdefaults=None)
        self.task = dict(source=str(self.task_path), source_sha256=self.digest(self.task_path),
                         inventory=dict(referenced=['left', 'check']),
                         fields=dict(check=dict(present=True, value=self.value)))

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_source_bound_callback_passes(self):
        cache = {}
        self.assertEqual(verify_task_callbacks(self.task, self.output, cache), 1)
        self.assertIn(str(self.method_path), cache)

    def test_wrong_owner_or_selection_is_rejected(self):
        for key, changed in [('owner_class', 'envs.other.other'),
                             ('function_qualname', 'Base_Task.right'),
                             ('function_module', 'different')]:
            with self.subTest(key=key):
                task = copy.deepcopy(self.task)
                task['fields']['check']['value'][key] = changed
                with self.assertRaises(ValueError):
                    verify_task_callbacks(task, self.output, {})

    def test_changed_function_source_is_rejected(self):
        self.method_path.write_text(self.method_path.read_text() + '# changed\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            verify_task_callbacks(self.task, self.output, {})

    def test_uncovered_defaults_are_rejected(self):
        self.value['defaults'] = {'sequence_type': 'tuple', 'values': [0.5]}
        with self.assertRaisesRegex(ValueError, 'defaults'):
            verify_task_callbacks(self.task, self.output, {})


if __name__ == '__main__':
    unittest.main()

import unittest
import torch

from scripts.captured_leaf_operator import capture_operator, replay_operator, recorded_runtime


class CapturedLeaf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def capture(self, module, operand, autocast=False):
        module.eval()
        with torch.inference_mode(), torch.autocast('cpu', enabled=autocast, dtype=torch.bfloat16):
            output = module(operand)
            return capture_operator(module, operand, output, provenance={'scope': 'synthetic_cpu_control'})

    def test_strided_linear_replayed_twice(self):
        operand = torch.arange(24, dtype=torch.float32).reshape(4, 6)[1:4, 1:5]
        record = self.capture(torch.nn.Linear(4, 3), operand)
        self.assertTrue(replay_operator(record, 'cpu', require_pointer_alignment=False)['repeats_exact'])

    def test_broadcast_linear(self):
        operand = torch.arange(4, dtype=torch.float32).expand(3, 4)
        record = self.capture(torch.nn.Linear(4, 2), operand)
        self.assertTrue(replay_operator(record, 'cpu', require_pointer_alignment=False)['repeats_exact'])

    def test_autocast_linear(self):
        record = self.capture(torch.nn.Linear(4, 3), torch.arange(8, dtype=torch.float32).reshape(2, 4), autocast=True)
        self.assertEqual(record['output']['dtype'], 'torch.bfloat16')
        self.assertTrue(replay_operator(record, 'cpu', require_pointer_alignment=False)['repeats_exact'])

    def test_channels_last_convolution(self):
        operand = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8).to(memory_format=torch.channels_last)
        record = self.capture(torch.nn.Conv2d(3, 4, 3, padding=1), operand)
        self.assertTrue(replay_operator(record, 'cpu', require_pointer_alignment=False)['repeats_exact'])

    def test_corrupt_weight_rejected(self):
        record = self.capture(torch.nn.Linear(4, 3), torch.ones(2, 4))
        record['weight']['payload'] = record['weight']['payload'].clone()
        record['weight']['payload'][0] += 1
        with self.assertRaisesRegex(ValueError, 'physical bytes changed'):
            replay_operator(record, 'cpu', require_pointer_alignment=False)

    def test_wrong_runtime_rejected(self):
        record = self.capture(torch.nn.Linear(4, 3), torch.ones(2, 4))
        record['runtime']['torch_version'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            replay_operator(record, 'cpu', require_pointer_alignment=False)

    def test_unsupported_module_rejected(self):
        module = torch.nn.ReLU().eval()
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, 'Unsupported exact leaf'):
            capture_operator(module, torch.ones(2), torch.ones(2), provenance={'scope': 'synthetic'})

    def test_runtime_restored_after_failure(self):
        record = self.capture(torch.nn.Linear(4, 3), torch.ones(2, 4))
        before = torch.are_deterministic_algorithms_enabled()
        record['runtime']['deterministic_algorithms'] = not before
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with recorded_runtime(record['runtime']):
                self.assertEqual(torch.are_deterministic_algorithms_enabled(), not before)
                raise RuntimeError('intentional replay failure')
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), before)


if __name__ == '__main__':
    unittest.main()

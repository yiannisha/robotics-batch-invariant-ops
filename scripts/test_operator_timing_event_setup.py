"""Mock-only control: CUDA events must initialize before measured calls."""
import unittest
from unittest.mock import patch

import torch

from scripts.benchmark_qualified_operator_cases_v2 import measure


class EventSetupTest(unittest.TestCase):
    def test_all_events_primed_before_timed_operator_calls(self):
        calls = [0]
        initializations = []
        events = []

        class Event:
            def __init__(self, **kwargs):
                self.initialized = False
                events.append(self)

            def record(self):
                if not self.initialized:
                    initializations.append(calls[0])
                    self.initialized = True

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 1.

        def operation():
            calls[0] += 1
            return torch.tensor([2.])

        with patch('torch.cuda.Event', Event), patch('torch.cuda.synchronize'), \
                patch('torch.cuda.memory_allocated', return_value=0), \
                patch('torch.cuda.reset_peak_memory_stats'), \
                patch('torch.cuda.max_memory_allocated', return_value=0):
            result = measure(operation)
        self.assertEqual(calls[0], 221)  # 20 warm-ups + 1 repeat reference + 200 samples
        self.assertEqual(len(events), 400)
        self.assertEqual(initializations, [21] * 400)
        self.assertEqual(len(result['cuda_seconds']), 200)
        self.assertTrue(result['same_case_output_repeat_exact'])


if __name__ == '__main__':
    unittest.main()

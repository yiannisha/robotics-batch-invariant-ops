import unittest
import torch
from scripts.trace_model_operator_shapes import describe


class MaskTraceTests(unittest.TestCase):
    def test_large_broadcast_mask_reconstructs_exactly(self):
        source = torch.arange(30)[None, None, None, :] < torch.arange(8)[:, None, None, None] + 5
        expanded = source.expand(8, 1, 67, 30)
        record = describe(expanded)
        self.assertEqual(record['shape'], [8, 1, 67, 30])
        self.assertEqual(record['stride'], list(expanded.stride()))
        self.assertEqual(record['values_shape'], [8, 1, 1, 30])
        rebuilt = torch.tensor(record['values']).expand(record['shape'])
        self.assertTrue(torch.equal(rebuilt, expanded))

    def test_large_nonbroadcast_mask_is_not_truncated(self):
        record = describe(torch.ones(5000, dtype=torch.bool))
        self.assertNotIn('values', record)

    def test_small_mask_preserves_existing_schema(self):
        mask = torch.tensor([[True, False]])
        record = describe(mask)
        self.assertEqual(record['values'], [[True, False]])
        self.assertNotIn('values_shape', record)


if __name__ == '__main__':
    unittest.main()

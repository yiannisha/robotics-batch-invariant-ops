"""CPU controls for layouts/indexing; not GPU timing or kernel qualification."""
import base64
import hashlib
import unittest
import zlib

import torch
import torch.nn.functional as F

from scripts.operator_performance_operands import (
    copy_values, matrixwise, operands, per_matrix_attention, strided_tensor,
    tensor_signatures,
)


class OperatorPerformanceOperandTests(unittest.TestCase):
    def test_physical_strided_generation(self):
        desc = dict(shape=[3, 4, 5], stride=[0, 10, 2], dtype='torch.float32')
        a = strided_tensor(desc, torch.Generator().manual_seed(7), 'cpu')
        b = strided_tensor(desc, torch.Generator().manual_seed(7), 'cpu')
        self.assertEqual(a.stride(), (0, 10, 2))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[0], a[2]))
        self.assertEqual(tensor_signatures(a), tensor_signatures(b))

    def test_broadcast_mask_copy(self):
        target = torch.empty(6, dtype=torch.bool).as_strided((2, 4, 3), (3, 0, 1))
        source = torch.tensor([[[True, False, True]], [[False, True, False]]])
        copy_values(target, source)
        self.assertEqual(target.stride(), (3, 0, 1))
        self.assertTrue(torch.equal(target, source.expand(2, 4, 3)))
        bad = source.expand(2, 4, 3).clone()
        bad[0, 2, 0] = False
        with self.assertRaises(ValueError):
            copy_values(target, bad)

    def test_matrixwise_strides_broadcast_and_rank2(self):
        left = torch.arange(2 * 3 * 8, dtype=torch.float64).reshape(2, 3, 8)[..., ::2]
        right = torch.arange(5 * 4, dtype=torch.float64).reshape(5, 4).T.unsqueeze(0)
        self.assertTrue(torch.equal(matrixwise(left, right), torch.matmul(left, right)))
        self.assertTrue(torch.equal(matrixwise(left[0], right[0]), left[0] @ right[0]))

    def test_attention_masks_and_gqa(self):
        torch.manual_seed(7)
        q = torch.randn(2, 4, 3, 5, dtype=torch.float64)
        k = torch.randn(2, 2, 6, 5, dtype=torch.float64)
        v = torch.randn(2, 2, 6, 7, dtype=torch.float64)
        mask = torch.ones(2, 1, 3, 6, dtype=torch.bool)
        mask[..., -2:] = False
        for value in [mask, torch.where(mask, 0., float('-inf')).double()]:
            expected = F.scaled_dot_product_attention(q, k, v, value, enable_gqa=True, scale=.3)
            actual = per_matrix_attention(q, k, v, value, enable_gqa=True, scale=.3)
            self.assertTrue(torch.allclose(actual, expected, rtol=1e-12, atol=1e-12))

    def test_causal_and_invalid_attention(self):
        q = torch.randn(1, 2, 3, 4, dtype=torch.float64)
        expected = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        self.assertTrue(torch.allclose(per_matrix_attention(q, q, q, is_causal=True), expected))
        with self.assertRaises(ValueError):
            per_matrix_attention(q, q, q, dropout_p=.1)
        with self.assertRaises(ValueError):
            per_matrix_attention(q, q, q, torch.ones(3, 3, dtype=torch.bool), is_causal=True)

    def test_exact_bfloat16_capture_and_selector(self):
        values = torch.tensor([[[[0., float('-inf'), -0.]]], [[[1., 2., 3.]]]], dtype=torch.bfloat16)
        payload = values.reshape(-1).view(torch.uint8).numpy().tobytes()
        cap = dict(encoding='zlib_base64_original_dtype_bytes_v1',
                   data=base64.b64encode(zlib.compress(payload)).decode(), dtype='torch.bfloat16',
                   original_dtype_bytes_count=len(payload), original_dtype_bytes_sha256=hashlib.sha256(payload).hexdigest(),
                   values_shape=[2, 1, 1, 3], shape=[2, 1, 4, 3], stride=[3, 3, 0, 1])
        case = dict(entry=dict(arguments=[None, None, None,
                                         dict(dtype='torch.bfloat16', shape=[1, 1, 4, 3], stride=[3, 3, 0, 1])],
                               keyword_arguments={}), mask_binding=dict(capture=cap, selector=0))
        args, _ = operands(case, 1, 'cpu')
        self.assertEqual(args[3].stride(), (3, 3, 0, 1))
        self.assertTrue(torch.equal(args[3].reshape(-1).view(torch.int16),
                                    values[0:1].expand(1, 1, 4, 3).reshape(-1).view(torch.int16)))
        cap['original_dtype_bytes_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            operands(case, 1, 'cpu')


if __name__ == '__main__':
    unittest.main()

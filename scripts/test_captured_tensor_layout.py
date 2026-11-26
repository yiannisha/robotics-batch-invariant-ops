import copy
import unittest
import torch

from scripts.captured_tensor_layout import byte_hash, capture, restore


class CapturedLayout(unittest.TestCase):
    def roundtrip(self, value):
        snapshot = capture(value)
        replay = restore(snapshot, 'cpu', require_pointer_alignment=False)
        self.assertEqual(byte_hash(value), byte_hash(replay))
        self.assertEqual(value.stride(), replay.stride())
        self.assertEqual(value.storage_offset(), replay.storage_offset())
        return snapshot

    def test_transpose_with_nonzero_offset(self):
        self.roundtrip(torch.arange(40, dtype=torch.float32).reshape(5, 8)[1:4, 1:7].t())

    def test_zero_stride_broadcast(self):
        self.roundtrip(torch.arange(4, dtype=torch.bfloat16).reshape(1, 4).expand(7, 4))

    def test_gapped_storage(self):
        self.roundtrip(torch.arange(60, dtype=torch.float64)[3:55:4])

    def test_signed_zero_and_integer_bits(self):
        self.roundtrip(torch.tensor([0., -0.], dtype=torch.float32))
        self.roundtrip(torch.tensor([2 ** 62, 2 ** 62 + 1], dtype=torch.int64))

    def test_scalar_and_empty(self):
        self.roundtrip(torch.tensor(3., dtype=torch.bfloat16))
        self.roundtrip(torch.empty(2, 0, 3))

    def test_corrupt_payload_rejected(self):
        snapshot = capture(torch.arange(5, dtype=torch.float32))
        snapshot['payload'][0] += 1
        with self.assertRaisesRegex(ValueError, 'physical bytes changed'):
            restore(snapshot, 'cpu', require_pointer_alignment=False)

    def test_corrupt_stride_rejected(self):
        snapshot = capture(torch.arange(5, dtype=torch.float32))
        snapshot['strides'] = [2]
        with self.assertRaisesRegex(ValueError, 'span differs'):
            restore(snapshot, 'cpu', require_pointer_alignment=False)

    def test_capacity_limit_rejected(self):
        snapshot = capture(torch.arange(5, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, 'declared capacity'):
            restore(snapshot, 'cpu', require_pointer_alignment=False, maximum_storage_bytes=1)

    def test_alignment_mismatch_rejected(self):
        snapshot = capture(torch.arange(5, dtype=torch.float32))
        snapshot['data_pointer_mod256'] = 1
        with self.assertRaisesRegex(ValueError, 'alignment differs'):
            restore(snapshot, 'cpu')


if __name__ == '__main__':
    unittest.main()

import base64
import hashlib
import unittest
import zlib

import torch

from scripts.trace_model_operator_shapes_with_masks import mask_record


class AttentionMaskCaptureTests(unittest.TestCase):
    def test_additive_bfloat16_mask_round_trips_original_bits(self):
        mask=torch.tensor([[0.,-0.,torch.finfo(torch.bfloat16).min]],dtype=torch.bfloat16)
        saved=mask_record(mask);data=zlib.decompress(base64.b64decode(saved['data']))
        self.assertEqual(hashlib.sha256(data).hexdigest(),saved['original_dtype_bytes_sha256'])
        self.assertEqual(data,mask.view(torch.uint8).numpy().tobytes())
        reconstructed=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(saved['values_shape'])
        self.assertEqual(reconstructed.view(torch.uint8).tolist(),mask.view(torch.uint8).tolist())

    def test_expanded_boolean_mask_retains_compact_values_and_source_stride(self):
        compact=torch.tensor([[[[True,False,True]]],[[[False,True,True]]]])
        expanded=compact.expand(2,1,67,3);saved=mask_record(expanded)
        self.assertEqual(saved['shape'],[2,1,67,3]);self.assertEqual(saved['values_shape'],[2,1,1,3])
        self.assertEqual(saved['stride'],list(expanded.stride()))
        data=zlib.decompress(base64.b64decode(saved['data']))
        reconstructed=torch.frombuffer(bytearray(data),dtype=torch.bool).reshape(saved['values_shape']).expand(saved['shape'])
        self.assertTrue(torch.equal(reconstructed,expanded))
        self.assertEqual(saved['original_dtype_bytes_count'],6)


if __name__=='__main__':unittest.main()

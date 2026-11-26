"""Source softmax output precision must survive operator qualification."""
import unittest
from unittest.mock import patch

import torch

from scripts.qualify_real_shape_matrix_ops import call, explicit_per_matrix_reference


class OperatorOutputDtypeTests(unittest.TestCase):
    def test_native_and_explicit_reference_preserve_requested_float32_output(self):
        x=torch.tensor([[1.,2.,3.],[3.,2.,1.]],dtype=torch.bfloat16)
        config=dict(dim=-1,m_per_request=1,output_dtype='torch.float32')
        expected=torch.softmax(x,dim=-1,dtype=torch.float32)
        result=call('softmax',x,None,None,invariant=False,config=config)
        reference=explicit_per_matrix_reference('softmax',x,None,None,config)
        self.assertEqual(result.dtype,torch.float32)
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
        torch.testing.assert_close(reference,expected,rtol=0,atol=0)
        self.assertFalse(torch.equal(result,torch.softmax(x,dim=-1).float()))

    def test_invariant_receives_source_output_dtype(self):
        x=torch.ones(1,3,dtype=torch.bfloat16)
        with patch('scripts.qualify_real_shape_matrix_ops.softmax_batch_invariant') as function:
            call('softmax',x,None,None,invariant=True,config=dict(dim=-1,output_dtype='float32'))
            function.assert_called_once_with(x,dim=-1,dtype=torch.float32)

    def test_float64_reference_and_legacy_unspecified_dtype(self):
        x=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
        result=call('softmax',x,None,None,invariant=False,config=dict(dim=-1,output_dtype='float64'))
        self.assertEqual(result.dtype,torch.float64)
        x=x.bfloat16()
        self.assertEqual(call('softmax',x,None,None,invariant=False,config=dict(dim=-1)).dtype,torch.bfloat16)


if __name__=='__main__':unittest.main()

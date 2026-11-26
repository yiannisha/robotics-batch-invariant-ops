"""Guard logical request coordinates in CogACT's split CFG branch layout."""
import unittest
import torch

from scripts.qualify_cogact_source_linear_ops import select_request,explicit_reference


class Coordinates(unittest.TestCase):
    def config(self):
        x=dict(shape=[2,3],stride=[3,1])
        return dict(batch_axis=0,groups_per_request=2,coordinate_layout='cfg_split_branches',
                    source=dict(b1=dict(arguments=[x]),b8=dict(arguments=[dict(shape=[16,3],stride=[3,1])])))

    def test_target_pairs_conditional_and_unconditional(self):
        value=torch.arange(24).reshape(8,3)
        self.assertTrue(torch.equal(select_request(self.config(),value,3),value[[3,7]]))
        self.assertTrue(torch.equal(select_request(self.config(),value,0),value[[0,4]]))

    def test_reference_restores_branch_major_output(self):
        value=torch.arange(24,dtype=torch.float32).reshape(8,3)
        weight=torch.tensor([[1.,2.,3.],[-1.,0.,1.]])
        actual=explicit_reference(self.config(),value,weight,None,4)
        self.assertTrue(torch.equal(actual,torch.nn.functional.linear(value,weight)))

    def test_contiguous_requests_remain_contiguous(self):
        c=self.config();c['coordinate_layout']='contiguous_requests'
        value=torch.arange(24).reshape(8,3)
        self.assertTrue(torch.equal(select_request(c,value,3),value[6:8]))


if __name__=='__main__':unittest.main()

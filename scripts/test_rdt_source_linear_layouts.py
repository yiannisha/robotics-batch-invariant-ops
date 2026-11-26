import unittest

import torch

from scripts.build_rdt_exact_linear_config import build
from scripts.qualify_rdt_source_linear_ops import allocate,explicit_reference,input_layout,operands,singleton


def tensor(shape,stride):return dict(shape=shape,stride=stride,dtype='torch.float64',device='cuda')


def source(shape,stride):
    return dict(aten_operation='aten::linear',arguments=[tensor(shape,stride),tensor([5,4],[4,1]),tensor([5],[1])],count=1)


def config(one,eight,axis,groups):return dict(source=dict(b1=one,b8=eight),batch_axis=axis,groups_per_request=groups)


class SourceLinearLayoutTests(unittest.TestCase):
    def test_sequence_major_grouped_targets_keep_b1_stride_and_coordinates(self):
        c=config(source([3,2,4],[4,12,1]),source([3,16,4],[4,12,1]),1,2)
        x,w,b=operands(c,2,7,device='cpu')
        self.assertEqual(list(x.stride()),[4,12,1])
        target=singleton(c,x,1)
        self.assertEqual(list(target.shape),[3,2,4]);self.assertEqual(list(target.stride()),[4,12,1])
        torch.testing.assert_close(target,x[:,2:4,:],rtol=0,atol=0)
        target[0,0,0]=999
        self.assertNotEqual(x[0,2,0],999)
        expected=torch.nn.functional.linear(x,w,b)
        actual=explicit_reference(c,x,w,b,2)
        torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)

    def test_batch_linear_stride_and_unknown_stride_rejection(self):
        c=config(source([3,2,4],[8,4,1]),source([3,16,4],[64,4,1]),1,2)
        self.assertEqual(input_layout(c,31),([3,62,4],[248,4,1]))
        c['source']['b8']['arguments'][0]['stride'][0]=17
        with self.assertRaisesRegex(ValueError,'Uncovered source stride'):input_layout(c,2)

    def test_unbatched_global_projection_remains_unbatched(self):
        c=config(source([1,4],[4,1]),source([1,4],[4,1]),None,None)
        self.assertEqual(input_layout(c,64),([1,4],[4,1]))

    def test_overlap_is_not_accepted_as_an_actual_linear_layout(self):
        with self.assertRaisesRegex(ValueError,'Overlapping source'):
            allocate([2,2],[0,1],torch.float32,'cpu')

    def test_union_retains_global_and_batched_paths_from_one_b1_pattern(self):
        def trace(size,operations):return dict(adapter='scripts.model_invariance.adapters.rdt_maniskill_exact',
            batch_size=size,adapter_execution_configuration={'tf32':False},replay_manifest_sha256='same',operations=operations)
        a=trace(1,[source([1,4],[4,1])]);b=trace(8,[source([1,4],[4,1]),source([8,4],[4,1])])
        result=build(a,b)
        self.assertEqual(len(result['configurations']),2)
        self.assertEqual({c['batch_axis'] for c in result['configurations']},{None,0})
        b['operations'].append(source([7,4],[4,1]))
        with self.assertRaisesRegex(ValueError,'Unmatched B8'):build(a,b)


if __name__=='__main__':unittest.main()

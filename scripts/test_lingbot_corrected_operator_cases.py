"""Corrected trace identity/precision must survive component case generation."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.build_lingbot_corrected_operator_cases import ADAPTER,validate_traces,main


def tensor(shape):
    stride=[];size=1
    for value in reversed(shape):stride.insert(0,size);size*=value
    return dict(shape=shape,stride=stride,dtype='torch.float32',device='cuda')


def trace(batch):
    def op(name,args):return dict(aten_operation='aten::'+name,arguments=args,keyword_arguments={},count=1)
    operations=[op('linear',[tensor([batch,2,4]),tensor([5,4]),tensor([5])]),
                op('mean',[tensor([batch,10,4]),[-1],False]),op('softmax',[tensor([batch,4]),-1]),
                op('mean',[tensor([8,32,2560]),[1]])]
    for width in [10,16]:operations.append(op('matmul',[tensor([3,batch,width,width]),tensor([3,batch,width,1])]))
    return dict(adapter=ADAPTER,batch_size=batch,implementation='native',tf32_matmul=False,tf32_cudnn=False,
        adapter_execution_configuration=dict(tf32=False,language_tokens=72,checkpoint_revision='0451855729ec904f970600e0aec8b84661423afe'),
        replay_manifest_sha256='manifest',gpu='same-gpu',torch='same-runtime',operations=operations)


class CorrectedOperatorCases(unittest.TestCase):
    def test_corrected_native_trace_pair_is_valid(self):validate_traces(trace(1),trace(8))

    def test_legacy_labels_or_enabled_actual_precision_are_rejected(self):
        for key,value in [('adapter',ADAPTER.removesuffix('_v2')),('tf32_matmul',True),('tf32_cudnn',True),('implementation','batch_invariant')]:
            one,eight=trace(1),trace(8);eight[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_traces(one,eight)

    def test_different_source_runtime_is_rejected(self):
        for key in ['gpu','torch','replay_manifest_sha256']:
            one,eight=trace(1),trace(8);eight[key]='different'
            with self.subTest(key=key),self.assertRaises(ValueError):validate_traces(one,eight)

    def test_builder_preserves_adapter_and_does_not_claim_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);one=d/'b1.json';eight=d/'b8.json';one.write_text(json.dumps(trace(1)));eight.write_text(json.dumps(trace(8)))
            with patch.object(sys,'argv',['builder','--b1',str(one),'--b8',str(eight),'--output',str(d/'cases')]),contextlib.redirect_stdout(io.StringIO()):main()
            for family,count in [('linear',1),('reduction',2),('remaining',3)]:
                output=json.loads((d/'cases'/(family+'_cases.json')).read_text())
                self.assertEqual(output['adapter'],ADAPTER);self.assertEqual(len(output['configurations']),count)
                self.assertFalse(output['measurements_complete']);self.assertTrue(output['corrected_actual_trace_precision_verified'])
                self.assertEqual(len(output['source' if family=='reduction' else 'traces']),2)

    def test_new_qualifiers_reject_legacy_config_before_measurement(self):
        from scripts.qualify_lingbot_exact_linear_ops_v2 import main as linear
        from scripts.qualify_lingbot_exact_remaining_ops_v2 import main as remaining
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);config=d/'config.json';config.write_text(json.dumps(dict(adapter=ADAPTER.removesuffix('_v2'))))
            for entry in [linear,remaining]:
                with self.subTest(entry=entry.__module__),patch.object(sys,'argv',['qualifier','--config',str(config),'--output',str(d/'records.jsonl')]),self.assertRaises(ValueError):entry()
                self.assertFalse((d/'records.jsonl').exists())


if __name__=='__main__':unittest.main()

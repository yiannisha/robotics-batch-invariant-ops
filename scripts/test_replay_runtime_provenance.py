"""Check that resumed research records cannot silently change their runtime."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_model_replay_invariance as replay


class RuntimeProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.args=argparse.Namespace(adapter='test.adapter',output=Path(self.temp.name)/'raw.jsonl',
            split='diagnostic',batch_sizes=[1,2,4],modes=['native','batch_invariant'],freeze=None)
        self.configuration={'steps':5}
        self.adapter=SimpleNamespace(execution_configuration=lambda:dict(self.configuration))
        self.properties=SimpleNamespace(name='test GPU',uuid='GPU-test')
        self.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                      cudnn=SimpleNamespace(allow_tf32=False))
        self.stack=[patch.object(replay.torch.cuda,'current_device',return_value=0),
                    patch.object(replay.torch.cuda,'get_device_properties',return_value=self.properties),
                    patch.object(replay.torch,'backends',self.backends)]
        for item in self.stack:item.start();self.addCleanup(item.stop)

    def write(self):replay.record_runtime_provenance(self.args,self.adapter,'a'*64)

    def test_matching_resume_preserves_recorded_identity(self):
        self.write();path=Path(str(self.args.output)+'.runtime.json');original=path.read_bytes()
        self.args.output.write_text('{}\n');self.write()
        self.assertEqual(original,path.read_bytes())
        self.assertEqual(json.loads(original)['gpu_uuid'],'GPU-test')

    def test_refuses_changed_device(self):
        self.write();self.properties.uuid='GPU-other'
        with self.assertRaisesRegex(ValueError,'identity differs'):self.write()

    def test_refuses_changed_model_configuration(self):
        self.write();self.configuration['steps']=10
        with self.assertRaisesRegex(ValueError,'identity differs'):self.write()

    def test_refuses_changed_source_hash(self):
        self.write()
        with patch.object(replay,'digest',return_value='f'*64):
            with self.assertRaisesRegex(ValueError,'identity differs'):self.write()

    def test_refuses_changed_precision_flag(self):
        self.write();self.backends.cudnn.allow_tf32=True
        with self.assertRaisesRegex(ValueError,'identity differs'):self.write()

    def test_refuses_legacy_records_without_sidecar(self):
        self.args.output.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'lack runtime provenance'):self.write()


if __name__=='__main__':unittest.main()

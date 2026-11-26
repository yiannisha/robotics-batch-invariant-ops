"""Ensure timing cannot silently use a different numerical path or corpus."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_isolated_frozen_benchmark import sha,validate_freeze,validate_calibration


class FrozenBenchmark(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.manifest=root/'manifest.json';self.manifest.write_text('{}')
        self.runtime=root/'invariance.jsonl.gz.runtime.json'
        self.runtime.write_text(json.dumps(dict(adapter='test.adapter',execution_configuration={'padding':64})))
        self.freeze=root/'freeze.json'
        self.freeze.write_text(json.dumps(dict(prepared_input_disjointness_verified=True,invariant_diagnostic_exact=True,
            adapter_execution_configuration={'padding':64},content_addressed_freeze={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (self.manifest,self.runtime)})))
        self.argv=['--adapter','test.adapter','--replay-manifest',str(self.manifest)]

    def test_matching_frozen_inputs_accepted(self):
        self.assertEqual(validate_freeze(self.freeze,self.argv)['adapter'],'test.adapter')

    def test_changed_manifest_rejected(self):
        self.manifest.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'replay inputs'):validate_freeze(self.freeze,self.argv)

    def test_changed_numerical_configuration_rejected(self):
        self.runtime.write_text(json.dumps(dict(adapter='test.adapter',execution_configuration={'padding':32})))
        with self.assertRaisesRegex(ValueError,'numerical path'):validate_freeze(self.freeze,self.argv)

    def test_other_adapter_rejected(self):
        self.argv[1]='other.adapter'
        with self.assertRaisesRegex(ValueError,'numerical path'):validate_freeze(self.freeze,self.argv)

    def test_progress_only_freshness_rejected(self):
        data=json.loads(self.freeze.read_text());data['prepared_input_disjointness_verified']=False
        self.freeze.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'Complete four-model'):validate_freeze(self.freeze,self.argv)

    def calibration(self):
        timings=Path(self.temp.name)/'serving.jsonl';timings.write_text('measured timing bytes\n')
        proof=Path(self.temp.name)/'serving-proof.json'
        data=dict(accepted=True,status='completed',benchmark='serving',freeze_sha256='freeze',gpu_uuid='GPU-test',
            benchmark_output=str(timings.resolve()),benchmark_output_sha256=sha(timings))
        proof.write_text(json.dumps(data));return timings,proof,data

    def test_queue_calibration_requires_exact_accepted_serving_bytes(self):
        timings,proof,_=self.calibration()
        self.assertEqual(len(validate_calibration(timings,proof,'freeze','GPU-test')),2)
        timings.write_text('other measurements\n')
        with self.assertRaisesRegex(ValueError,'bytes differ'):validate_calibration(timings,proof,'freeze','GPU-test')

    def test_queue_calibration_rejects_contaminated_serving_run(self):
        timings,proof,data=self.calibration();data['accepted']=False;proof.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'accepted isolated'):validate_calibration(timings,proof,'freeze','GPU-test')

    def test_queue_calibration_rejects_other_gpu_or_freeze(self):
        timings,proof,_=self.calibration()
        for freeze,gpu in [('other','GPU-test'),('freeze','GPU-other')]:
            with self.assertRaisesRegex(ValueError,'freeze/GPU differs'):validate_calibration(timings,proof,freeze,gpu)


if __name__=='__main__':unittest.main()

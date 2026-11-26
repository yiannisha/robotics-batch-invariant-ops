"""Regression gates for exposed/partial corpora and incomplete operator sweeps."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.freeze_four_model_numerics import disjointness_gate,operator_gate,diagnostic_audit_gate,sha256


class FreezeGates(unittest.TestCase):
    def test_diagnostic_audit_must_bind_unchanged_raw_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=root/'manifest.json';m.write_text('{}');a=root/'audit.json'
            a.write_text(json.dumps(dict(status='complete_four_model_diagnostic_independently_verified',model_id='pi05_libero',adapter_execution_configuration={},invariant_output_and_all_recorded_stages_exact=True,manifest_sha256=sha256(m),auditor_sha256=sha256(Path(__file__).with_name('audit_four_model_diagnostic.py')))))
            diagnostic_audit_gate(a,'pi05_libero',{},dict(manifest=m))
            m.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Audited diagnostic input changed'):diagnostic_audit_gate(a,'pi05_libero',{},dict(manifest=m))

    def test_progress_only_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=root/'manifest.json';m.write_text('{}');a=root/'audit.json'
            a.write_text(json.dumps(dict(status='prepared_input_exposure_progress_only',candidate_manifest_sha256=sha256(m),collisions=[])))
            with self.assertRaisesRegex(ValueError,'Final manifest-bound'):disjointness_gate(m,a)

    def test_raw_only_or_noise_inclusive_identity_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=root/'manifest.json';m.write_text('{}');a=root/'audit.json'
            value=dict(status='complete_observation_disjoint_heldout_independently_verified',candidate_manifest_sha256=sha256(m),identity_kind='raw_observation',prepared_input_disjointness_verified=True,noise_and_request_ids_excluded=True)
            a.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'prepared neural'):disjointness_gate(m,a)
            value.update(identity_kind='prepared_neural_inputs_before_first_model_operation',noise_and_request_ids_excluded=False);a.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'prepared neural'):disjointness_gate(m,a)
            value['noise_and_request_ids_excluded']=True;a.write_text(json.dumps(value));disjointness_gate(m,a)

    def test_inventory_mapping_is_not_measurement_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'audit.json';a.write_text(json.dumps(dict(status='source_operator_inventory_reconciled',all_source_patterns_mapped=True,full_traced_operator_qualification_complete=False)))
            with self.assertRaisesRegex(ValueError,'Complete source inventory'):operator_gate(a,'rdt_maniskill_exact',{})

    def test_rdt_requires_all_task_masks_even_if_shapes_covered(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'audit.json';a.write_text(json.dumps(dict(status='source_operator_inventory_reconciled',full_traced_operator_qualification_complete=True,all_five_task_and_mixed_mask_measurements_complete=False)))
            with self.assertRaisesRegex(ValueError,'task-mask'):operator_gate(a,'rdt_maniskill_exact',{})


if __name__=='__main__':unittest.main()

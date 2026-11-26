"""Synthetic protocol tests; no campaign seal or model acceptance is produced."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.validate_lingbot_exposure_freeze_schema import main
from scripts.freeze_four_model_numerics import disjointness_gate


class ExposureSchema(unittest.TestCase):
    def run_case(self,change=None,change_evidence=False):
        with tempfile.TemporaryDirectory(prefix='synthetic-lingbot-exposure-') as directory:
            root=Path(directory);source=Path(__file__).with_name('audit_lingbot_heldout_observation_disjointness.py')
            manifest=root/'synthetic-manifest.json';manifest.write_text('{}')
            evidence=root/'synthetic-evidence';evidence.write_text('original')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            record=dict(status='complete_observation_disjoint_heldout_independently_verified',model_id='lingbot_robotwin_exact',
                complete_paired_starts=2500,prepared_input_disjointness_verified=True,collision_count=0,
                heldout_observations=800,heldout_unique_prepared_inputs=800,noise_excluded_from_input_identity=True,
                identity_kind='prepared_neural_inputs_before_first_model_operation',candidate_manifest_sha256=sha(manifest),
                auditor_sha256=sha(source),source_evidence_sha256={str(evidence):sha(evidence)},synthetic_fixture=True)
            if change:record.update(change)
            if change_evidence:evidence.write_text('changed')
            input_file=root/'synthetic-input.json';input_file.write_text(json.dumps(record));output=root/'synthetic-output.json'
            with patch('sys.argv',['validator','--input',str(input_file),'--output',str(output)]):
                if change or change_evidence:
                    with self.assertRaises(ValueError):main()
                    self.assertFalse(output.exists())
                else:
                    main();converted=json.loads(output.read_text())
                    self.assertTrue(converted['noise_and_request_ids_excluded'])
                    self.assertEqual(converted['collisions'],[])
                    disjointness_gate(manifest,output)

    def test_valid_synthetic_proof_matches_common_gate(self):self.run_case()
    def test_incomplete_cohort_rejected(self):self.run_case({'complete_paired_starts':1})
    def test_changed_auditor_rejected(self):self.run_case({'auditor_sha256':'0'*64})
    def test_changed_evidence_rejected(self):self.run_case(change_evidence=True)


if __name__=='__main__':unittest.main()

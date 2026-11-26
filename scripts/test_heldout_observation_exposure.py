import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_rdt_heldout_observation_disjointness import observation_identity
from scripts.rdt_prepared_observation_identity import prepared_state
from scripts.run_exposure_qualified_heldout import validate


class HeldoutExposureTests(unittest.TestCase):
    def test_new_noise_or_request_identifier_does_not_make_observation_unseen(self):
        request=dict(task_name='PickCube-v1',history_valid=[False,True],image_history=np.zeros((2,2,2,3),dtype=np.uint8),
            proprio=np.zeros(8,dtype=np.float32),noise=np.zeros((64,128),dtype=np.float32),paired_request_id='old')
        other={**request,'noise':np.ones((64,128),dtype=np.float32),'paired_request_id':'new'}
        self.assertEqual(observation_identity(request),observation_identity(other))
        other={**request,'proprio':request['proprio'].copy()};other['proprio'][0]=np.nextafter(np.float32(0),np.float32(1))
        self.assertNotEqual(observation_identity(request),observation_identity(other))

    def test_exposure_denial_and_disjointness_audit_must_be_bound_by_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            identity=hashlib.sha256(manifest.read_bytes()).hexdigest();audit=root/'audit.json'
            audit.write_text(json.dumps(dict(status='complete_observation_disjoint_heldout_independently_verified',candidate_manifest_sha256=identity,identity_kind='prepared_neural_inputs_before_first_model_operation',prepared_input_disjointness_verified=True)))
            freeze=root/'freeze.json';freeze.write_text(json.dumps({'content_addressed_freeze':{}}))
            with self.assertRaisesRegex(ValueError,'freeze must bind'):validate(manifest,freeze,audit,[])
            bound={str(p.resolve()):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (manifest,audit)}
            freeze.write_text(json.dumps({'content_addressed_freeze':bound}));validate(manifest,freeze,audit,[])
            evidence=root/'exposure.json';evidence.write_text(json.dumps(dict(status='old_numerical_heldout_observation_partition_exposed_before_freeze',candidate_manifest_sha256=identity)))
            with self.assertRaisesRegex(ValueError,'exposed before freeze'):validate(manifest,freeze,audit,[evidence])
            audit.write_text(json.dumps(dict(status='provisional_snapshot',candidate_manifest_sha256=identity)))
            with self.assertRaisesRegex(ValueError,'completed observation-disjointness'):validate(manifest,freeze,audit,[])

    def test_raw_joint_change_can_disappear_before_neural_inference(self):
        import torch
        a=np.zeros(8,dtype=np.float32);b=a.copy();b[0]=np.nextafter(np.float32(0),np.float32(1))
        self.assertNotEqual(a.tobytes(),b.tobytes())
        self.assertTrue(torch.equal(prepared_state(a),prepared_state(b)))

    def test_raw_only_disjointness_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            audit=root/'audit.json';audit.write_text(json.dumps(dict(status='complete_observation_disjoint_heldout_independently_verified',candidate_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
            with self.assertRaisesRegex(ValueError,'disjoint prepared neural inputs'):validate(manifest,root/'unused-freeze.json',audit,[])


if __name__=='__main__':unittest.main()

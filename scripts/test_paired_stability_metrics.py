"""Check numerical and geometric interpretation of paired stability results."""
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from scripts.summarize_paired_stability import compare, numerical_pair, rotation_distance


def call(values, input_id='same', batch=1):
    value=np.asarray(values,dtype=np.float32)
    return dict(normalized_actions=value.tolist(),normalized_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                target_input_sha256=input_id,noise_id=7,batch_size=batch,position=0,call=0)


class StabilityMetricsTests(unittest.TestCase):
    def test_quaternion_identity_antipodes_and_known_small_angle(self):
        q=np.array([[0.,0.,0.,1.]])
        self.assertEqual(rotation_distance(q,q)[0],0.)
        self.assertEqual(rotation_distance(q,-q)[0],0.)
        angle=0.001
        r=np.array([[0.,0.,np.sin(angle/2),np.cos(angle/2)]])
        self.assertAlmostEqual(rotation_distance(q,r)[0],angle,places=14)

    def test_single_ulp_and_signed_zero_keep_bitwise_definition(self):
        x=np.float32(1);y=np.nextafter(x,np.float32(2))
        result=numerical_pair(call([x]),call([y]))
        self.assertFalse(result['exact'])
        self.assertEqual(result['max_ulp'],1)
        self.assertEqual(result['max_absolute_error'],float(y-x))
        result=numerical_pair(call([0.]),call([-0.]))
        self.assertFalse(result['exact'])
        self.assertEqual(result['max_absolute_error'],0.)

    def test_feedback_differences_excluded_from_fixed_input_statistics(self):
        frame=dict(physics_sha256='state',observation_sha256='observation')
        def episode(c):
            return dict(steps=[dict(action=[0.]*7)],trajectory=[frame,frame],calls=[c],success=False)
        pose=np.array([[0.,0.,0.,0.,0.,0.,1.]]*2)
        with patch('scripts.summarize_paired_stability.pose_cache',return_value=pose):
            result=compare('rdt',None,episode(call([1.])),episode(call([2.],input_id='feedback',batch=8)),None)
            self.assertEqual(result['later_calls_with_different_inputs'],1)
            self.assertEqual(result['fixed_identical_input_nonsingleton_pairs'],[])
            result=compare('rdt',None,episode(call([1.])),episode(call([2.],batch=8)),None)
            self.assertEqual(len(result['fixed_identical_input_nonsingleton_pairs']),1)
            self.assertEqual(result['fixed_identical_input_nonsingleton_pairs'][0]['max_absolute_error'],1.)
            self.assertEqual(result['eef_rotation_max_rad'],0.)

    def test_bfloat16_storage_is_measured_at_original_precision(self):
        def stored(bits):
            payload=np.asarray(bits,dtype=np.uint16).tobytes()
            return dict(normalized_bytes=payload.hex(),normalized=dict(dtype='torch.bfloat16',
                shape=[len(bits)],sha256=hashlib.sha256(payload).hexdigest()))
        result=numerical_pair(stored([0x3f80]),stored([0x3f81]))
        self.assertEqual(result['source_float_bits'],16)
        self.assertEqual(result['max_ulp'],1)
        self.assertEqual(result['max_absolute_error'],1/128)
        with self.assertRaisesRegex(ValueError,'Normalized bytes changed'):
            invalid=stored([0x3f80]);invalid['normalized_bytes']='0000'
            numerical_pair(invalid,stored([0x3f80]))


if __name__=='__main__':unittest.main()

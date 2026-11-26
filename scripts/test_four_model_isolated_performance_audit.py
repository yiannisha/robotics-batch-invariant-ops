import copy
import unittest
from scripts.audit_four_model_isolated_performance import isolation,finite


class IsolationEvidence(unittest.TestCase):
    def evidence(self):
        initial=dict(gpu_uuid='GPU-test',gpu_name='test GPU',driver_version='test',processes=[],observed_at_utc='2026-01-01T00:00:00+00:00')
        running=dict(initial,processes=[{'pid':123,'type':'C+G'}],observed_at_utc='2026-01-01T00:00:01+00:00')
        return dict(accepted=True,status='completed',child_exit_code=0,benchmark='policy',freeze_sha256='freeze',poll_interval_seconds=1.0,
                    benchmark_pid=123,gpu_uuid='GPU-test',observations=[initial,running],started_at_utc='2026-01-01T00:00:00+00:00',finished_at_utc='2026-01-01T00:00:02+00:00')

    def test_valid_samples(self):self.assertEqual(isolation(self.evidence(),'policy','freeze','test GPU')['samples'],2)
    def test_foreign_graphics_rejected(self):
        e=self.evidence();e['observations'][1]['processes'].append({'pid':456,'type':'G'})
        with self.assertRaisesRegex(ValueError,'Foreign'):isolation(e,'policy','freeze','test GPU')
    def test_wrong_freeze_rejected(self):
        with self.assertRaises(ValueError):isolation(self.evidence(),'policy','other','test GPU')
    def test_wrong_hardware_rejected(self):
        with self.assertRaises(ValueError):isolation(self.evidence(),'policy','freeze','other GPU')
    def test_missing_idle_observation_rejected(self):
        e=self.evidence();e['observations'][0]['processes']=[{'pid':123}]
        with self.assertRaises(ValueError):isolation(e,'policy','freeze','test GPU')
    def test_nonfinite_measurements_rejected(self):
        with self.assertRaises(ValueError):finite({'latency':[float('nan')]})
        with self.assertRaises(ValueError):finite({'latency':float('inf')})


if __name__=='__main__':unittest.main()

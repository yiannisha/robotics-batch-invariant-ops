"""Adversarial metadata and independent projection tests; no GPU proof."""
import copy
import unittest

import torch

from scripts.audit_captured_leaf_replay import select_target, validate_records


class CapturedLeafAudit(unittest.TestCase):
    def fixture(self):
        manifest = dict(observations=[dict(split='diagnostic', task='a', episode_id=0,
                                          request_ids=['request0', 'request1', 'request2'],
                                          noise=[{'sha256': str(i) * 64} for i in range(3)],
                                          companion_replay_indices=[1]),
                                      dict(split='diagnostic', task='b', episode_id=1)])
        trace = dict(adapter='scripts.model_invariance.adapters.cogact_simpler_exact',
                     modes={'native': {}, 'batch_invariant': {}}, seed=0, target_replay_index=0, noise_id=0,
                     replay_manifest_sha256='a' * 64, companion_replay_indices=[1], batch_size=2,
                     host_leaf_scan={'first_divergent_leaf': dict(name='model.action_model.net.x_embedder',
                                                                occurrence=0, module_type='Linear')},
                     request_projection_contract={'scope': 'synthetic_control'})
        records = {}
        for mode in ('native', 'batch_invariant'):
            for batch in ('singleton', 'batched'):
                p = dict(adapter=trace['adapter'], mode=mode, seed=0,
                         logical_batch_size=1 if batch == 'singleton' else 2,
                         replay_manifest_sha256=trace['replay_manifest_sha256'], target_replay_index=0,
                         noise_id=0, request_noise_sha256='0' * 64, target_request_id='request0',
                         companion_replay_indices=[1], module_name='model.action_model.net.x_embedder', module_occurrence=0,
                         projection_contract=trace['request_projection_contract'],
                         execution_configuration={'checkpoint_revision': 'synthetic_fixture'})
                records[mode + '_' + batch] = dict(operator='linear', provenance=p,
                                                   runtime={'invariant_mode': mode == 'batch_invariant'},
                                                   weight={'dtype': 'torch.float32', 'shape': [2, 2], 'logical_sha256': 'b' * 64},
                                                   bias=None)
        return records, trace, manifest

    def test_complete_synthetic_metadata(self):
        validate_records(*self.fixture())

    def test_missing_capture_rejected(self):
        records, trace, manifest = self.fixture()
        records.pop('native_batched')
        with self.assertRaisesRegex(ValueError, 'four-artifact'):
            validate_records(records, trace, manifest)

    def test_wrong_invariant_flag_rejected(self):
        records, trace, manifest = self.fixture()
        records['batch_invariant_batched']['runtime']['invariant_mode'] = False
        with self.assertRaisesRegex(ValueError, 'implementation flag'):
            validate_records(records, trace, manifest)

    def test_changed_weights_rejected(self):
        records, trace, manifest = self.fixture()
        records['batch_invariant_batched']['weight']['logical_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'weights differ'):
            validate_records(records, trace, manifest)

    def test_changed_noise_rejected(self):
        records, trace, manifest = self.fixture()
        records['native_batched']['provenance']['request_noise_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'request/noise'):
            validate_records(records, trace, manifest)

    def test_independent_cfg_projection(self):
        reference = torch.tensor([[1.], [10.]])
        candidate = torch.tensor([[1.], [2.], [10.], [20.]])
        self.assertTrue(torch.equal(select_target(reference, candidate, 2, 2), reference))

    def test_unresolved_projection(self):
        self.assertIsNone(select_target(torch.ones(2, 1), torch.ones(2, 1), 2, 2))


if __name__ == '__main__':
    unittest.main()

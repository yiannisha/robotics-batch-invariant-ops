"""Adversarial synthetic fixtures; these do not constitute model evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('causal_audit', Path(__file__).with_name('audit_exact_model_causal_ablation.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metric(exact, outputs=False):
    tensor = dict(shape=[1], dtype='torch.float32', exact=exact,
                  reference_sha256='a' * 64, candidate_sha256=('a' if exact else 'b') * 64,
                  unequal_element_count=0 if exact else 1, max_absolute_error=0. if exact else .25,
                  rmse=0. if exact else .25, nonfinite_count=0)
    tensors = ({'output.normalized': tensor, 'output.physical': copy.deepcopy(tensor)}
               if outputs else {'output': tensor})
    return dict(tensors=tensors, exact=exact, unequal_element_count=sum(t['unequal_element_count'] for t in tensors.values()),
                nonfinite_count=0, rmse=tensor['rmse'], max_absolute_error=tensor['max_absolute_error'])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        observations = [dict(split='diagnostic', task=str(i % 2), episode_id=i, observation_index=0,
                             request_ids=[f'{i}/{j}' for j in range(3)],
                             noise=[dict(sha256=str(j) * 64) for j in range(3)],
                             companion_replay_indices=[(i + 1) % 200]) for i in range(200)]
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(dict(observations=observations)))
        manifest_sha = module.sha(manifest)
        plan = self.root / 'plan.json'
        repo = Path(__file__).resolve().parents[1]
        pins = {str(repo / p): module.sha(repo / p) for p in (
            'scripts/run_exact_model_causal_ablation_v2.py', 'scripts/run_model_causal_ablation.py',
            'batch_invariant_ops/batch_invariant_ops.py', 'scripts/model_invariance/adapters/pi05_libero_exact.py')}
        adapter = 'scripts.model_invariance.adapters.pi05_libero_exact'
        plan.write_text(json.dumps(dict(model='pi05', manifest=str(manifest),
                                        manifest_sha256=manifest_sha, source_sha256=pins, adapter=adapter)))
        self.sidecar = dict(status='measurements_completed_independent_audit_pending',
                            plan=str(plan), plan_sha256=module.sha(plan), source_sha256=pins,
                            manifest_sha256=manifest_sha, inapplicable_operations={})
        self.acceptance = self.root / 'acceptance.json'
        self.acceptance.write_text(json.dumps(dict(status='complete_four_model_diagnostic_independently_verified',
                                                  request_noise_pairs=600, manifest_sha256=manifest_sha,
                                                  required_denoising_labels=['flow.step_00.sample'],
                                                  adapter_execution_configuration={'fixture': True})))
        self.rows = []
        for mode in sorted(module.MODES):
            for source in observations:
                for noise, request in enumerate(source['request_ids']):
                    exact = mode == 'full_invariant'
                    flow = metric(exact)
                    output = metric(exact, outputs=True)
                    companion = observations[source['companion_replay_indices'][0]]
                    self.rows.append(dict(mode=mode, request_id=request, task=source['task'],
                                          episode_id=source['episode_id'], observation_index=0, noise_id=noise,
                                          request_noise_sha256=source['noise'][noise]['sha256'],
                                          replay_manifest_sha256=manifest_sha, adapter=adapter,
                                          adapter_execution_configuration={'fixture': True}, batch_size=2,
                                          deterministic_algorithms=mode == 'native_deterministic',
                                          companion_replay_index=source['companion_replay_indices'][0],
                                          companion_request_id=companion['request_ids'][noise],
                                          flow_steps={'flow.step_00.sample': flow}, output=output,
                                          first_divergence=None if exact else dict(observed_boundary='flow.step_00.sample', metrics=flow),
                                          override_invocations={'mm': 1, 'convolution': 1, 'explicit_sdpa': 1},
                                          inapplicable_operations={}))

    def execute(self):
        raw = self.root / 'measurements.jsonl'
        with raw.open('w') as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + '\n')
        Path(str(raw) + '.provenance.json').write_text(json.dumps(dict(self.sidecar, output_sha256=module.sha(raw))))
        return module.audit(raw, self.acceptance)

    def test_complete_synthetic_contract(self):
        self.assertEqual(self.execute()['measurement_records'], 3600)

    def test_rehashed_missing_request(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete six-mode'):
            self.execute()

    def test_rehashed_duplicate_request(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate mode/request'):
            self.execute()

    def test_missing_stage(self):
        self.rows[0]['flow_steps'] = {}
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            self.execute()

    def test_skipped_attention(self):
        row = next(r for r in self.rows if r['mode'] == 'explicit_per_matrix_reference')
        row['override_invocations']['explicit_sdpa'] = 0
        with self.assertRaisesRegex(ValueError, 'Attention intervention bypassed'):
            self.execute()

    def test_wrong_noise(self):
        self.rows[0]['request_noise_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'Noise fingerprint'):
            self.execute()

    def test_contradictory_exactness(self):
        row = next(r for r in self.rows if r['mode'] == 'full_invariant')
        row['output']['tensors']['output.normalized']['candidate_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Contradictory byte'):
            self.execute()

    def test_wrong_boundary(self):
        row = next(r for r in self.rows if r['first_divergence'] is not None)
        row['first_divergence']['observed_boundary'] = 'returned_output'
        with self.assertRaisesRegex(ValueError, 'Wrong first'):
            self.execute()

    def test_signed_zero_difference_is_not_exact(self):
        value = metric(False)
        value['rmse'] = value['max_absolute_error'] = 0.
        value['tensors']['output']['rmse'] = value['tensors']['output']['max_absolute_error'] = 0.
        module.validate_metrics(value)


if __name__ == '__main__':
    unittest.main()

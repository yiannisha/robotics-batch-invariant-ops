"""Protocol/measurement controls; no model or full-campaign evidence."""
import copy
import hashlib
import unittest

import numpy as np

from scripts.summarize_lingbot_full_paired_stability import compare_actions, require_complete_proof


def call(index, batch, value=0.):
    output = np.full((50, 55), value, dtype=np.float32)
    return dict(call=index, batch_size=batch, target_position=0, noise_id=f'noise-{index}',
                target_input_sha256=f'input-{index}', normalized_actions=output.tolist(),
                normalized_sha256=hashlib.sha256(output.tobytes()).hexdigest())


class FullStabilityTests(unittest.TestCase):
    def episodes(self):
        left = dict(steps=[dict(action=[0.] * 14) for _ in range(100)],
                    calls=[call(0, 1), call(1, 1)])
        right = copy.deepcopy(left)
        right['calls'][1] = call(1, 2, .125)
        return left, right

    def proof(self):
        tasks = [f'task-{i}' for i in range(50)]
        lock = dict(global_tasks=tasks, initialization_ids=list(range(50)))
        proof = dict(status='lingbot_paired_raw_inputs_actions_and_observable_trajectories_independently_verified',
                     cohort_rollouts_complete=True, completed_paired_starts=2500, episodes=20000,
                     all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks=True,
                     episode_files_bound_to_verified_bytes=True,
                     episode_file_sha256={str(i): 'a' * 64 for i in range(20000)},
                     blocks=[dict(task=t, initialization=i) for t in tasks for i in range(50)])
        return proof, lock

    def test_complete_roster(self):
        require_complete_proof(*self.proof())

    def test_partial_or_duplicate_cohort_rejected(self):
        proof, lock = self.proof()
        proof['blocks'][-1] = proof['blocks'][0]
        with self.assertRaisesRegex(ValueError, 'Missing, duplicate'):
            require_complete_proof(proof, lock)

    def test_missing_episode_binding_rejected(self):
        proof, lock = self.proof()
        proof['episode_file_sha256'].pop('0')
        with self.assertRaisesRegex(ValueError, 'episode-byte binding'):
            require_complete_proof(proof, lock)

    def test_bitwise_physical_divergence_and_component_units(self):
        left, right = self.episodes()
        right['steps'][50]['action'][3] = .25
        r = compare_actions(left, right)
        self.assertEqual(r['first_action_divergence_control_index'], 50)
        self.assertEqual(r['action_disagreement_controls'], 1)
        self.assertEqual(r['action_component_max_absolute_difference'][3], .25)
        self.assertAlmostEqual(r['fixed_identical_input_nonsingleton_pairs'][0]['rms_error'], .125)

    def test_post_feedback_inputs_excluded(self):
        left, right = self.episodes()
        right['calls'][1]['target_input_sha256'] = 'different-input'
        r = compare_actions(left, right)
        self.assertEqual(r['fixed_identical_input_nonsingleton_pairs'], [])
        self.assertFalse(r['fixed_input_batching_error_observed'])
        self.assertEqual(r['later_calls_with_different_inputs'], 1)

    def test_noise_mismatch_rejected(self):
        left, right = self.episodes()
        right['calls'][1]['noise_id'] = 'different-noise'
        with self.assertRaisesRegex(ValueError, 'different noise'):
            compare_actions(left, right)

    def test_b1_only_episode_is_unobserved_not_zero_error(self):
        left, right = self.episodes()
        left['calls'] = left['calls'][:1]
        right['calls'] = right['calls'][:1]
        r = compare_actions(left, right)
        self.assertEqual(r['batched_policy_calls'], 0)
        self.assertFalse(r['fixed_input_batching_error_observed'])

    def test_unshared_control_tail_excluded(self):
        left, right = self.episodes()
        right['steps'].append(dict(action=[100.] * 14))
        r = compare_actions(left, right)
        self.assertFalse(r['horizons_equal'])
        self.assertEqual(r['aligned_controls'], 100)
        self.assertEqual(r['action_disagreement_controls'], 0)


if __name__ == '__main__':
    unittest.main()

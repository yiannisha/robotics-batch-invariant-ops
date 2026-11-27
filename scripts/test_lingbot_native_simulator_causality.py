"""Simulator-only drift cannot become evidence of native batching error."""
import copy
import unittest
from scripts.run_lingbot_robotwin_paired_study_v2 import CONDITIONS,compare_episodes
from scripts.audit_lingbot_paired_study_bound_v2 import verify_native_causal_prefix


class NativeSimulatorCausality(unittest.TestCase):
    def episodes(self):
        base=dict(calls=[dict(target_input_sha256='input',normalized_sha256='normalized',physical_sha256='physical')],
                  steps=[dict(policy_call=0,action=[0.]*14)],trajectory=[{'state':0},{'state':1}],success=False)
        episodes={name:copy.deepcopy(base) for name in CONDITIONS}
        return self.repeat(episodes)

    def repeat(self,episodes):
        for name in CONDITIONS:episodes[name+'_repeat']=copy.deepcopy(episodes[name])
        return episodes

    def rejected_by_both(self,episodes):
        with self.assertRaises(ValueError):compare_episodes(self.repeat(episodes))
        with self.assertRaises(ValueError):verify_native_causal_prefix(episodes['native_singleton'],episodes['native_dynamic'])

    def test_identical_controls_and_states_are_valid(self):
        episodes=self.episodes()
        self.assertIsNone(compare_episodes(episodes)['native_first_action_divergence'])
        self.assertIsNone(verify_native_causal_prefix(episodes['native_singleton'],episodes['native_dynamic']))

    def test_drift_without_action_difference_is_rejected(self):
        episodes=self.episodes();episodes['native_dynamic']['trajectory'][1]['state']=2
        self.rejected_by_both(episodes)

    def test_different_horizon_with_identical_common_controls_is_rejected(self):
        episodes=self.episodes();right=episodes['native_dynamic']
        right['steps'].append(copy.deepcopy(right['steps'][0]));right['trajectory'].append({'state':2})
        self.rejected_by_both(episodes)

    def test_different_outcome_with_identical_controls_and_states_is_rejected(self):
        episodes=self.episodes();episodes['native_dynamic']['success']=True
        self.rejected_by_both(episodes)

    def test_action_divergence_from_identical_input_remains_valid(self):
        episodes=self.episodes();right=episodes['native_dynamic']
        right['steps'][0]['action'][0]=0.01;right['trajectory'][1]['state']=2;self.repeat(episodes)
        self.assertEqual(compare_episodes(episodes)['native_first_action_divergence'],0)
        self.assertEqual(verify_native_causal_prefix(episodes['native_singleton'],right),0)


if __name__=='__main__':unittest.main()

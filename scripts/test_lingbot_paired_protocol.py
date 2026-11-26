import copy
import unittest

from scripts.run_lingbot_robotwin_paired_study import CONDITIONS,call_plan,compare_episodes


class PairedProtocol(unittest.TestCase):
    def test_noise_is_request_indexed_and_companions_are_unrelated(self):
        rows=[dict(task=str(i%5),replay_index=i) for i in range(40)]
        one=call_plan('0',300000,2,False,rows);batch=call_plan('0',300000,2,True,rows)
        self.assertEqual(one['noise_seed'],batch['noise_seed']);self.assertEqual(batch['batch_size'],4)
        self.assertEqual(batch['target_position'],2)
        self.assertTrue(all(rows[i]['task']!='0' for i in batch['companion_replay_indices']))
        self.assertEqual(batch,call_plan('0',300000,2,True,rows))
        self.assertNotEqual(batch['noise_seed'],call_plan('0',300000,3,True,rows)['noise_seed'])

    def test_short_episode_row_counts_do_not_duplicate_companions(self):
        rows=[dict(task='other',replay_index=i) for i in range(102)]
        result=call_plan('target',300000,3,True,rows)
        self.assertEqual(len(set(result['companion_replay_indices'])),7)

    def episodes(self):
        base=dict(calls=[dict(target_input_sha256='input',normalized_sha256='normalized',physical_sha256='physical')],
            steps=[dict(policy_call=0,action=[0.0])],trajectory=[{'state':0},{'state':1}],success=False)
        result={name:copy.deepcopy(base) for name in CONDITIONS}
        result['native_dynamic']['steps'][0]['action']=[0.01]
        result['native_dynamic']['trajectory'][1]['state']=2
        for name in CONDITIONS:result[name+'_repeat']=copy.deepcopy(result[name])
        return result

    def test_repeat_and_invariant_controls_with_native_divergence(self):
        self.assertEqual(compare_episodes(self.episodes())['native_first_action_divergence'],0)

    def test_same_condition_repeat_failure_is_rejected(self):
        e=self.episodes();e['native_singleton_repeat']['trajectory'][1]['state']=99
        with self.assertRaisesRegex(ValueError,'repeat differs'):compare_episodes(e)

    def test_invariant_trajectory_failure_is_rejected(self):
        e=self.episodes()
        for name in ('patched_dynamic','patched_dynamic_repeat'):e[name]['trajectory'][1]['state']=99
        with self.assertRaisesRegex(ValueError,'Invariant trajectories'):compare_episodes(e)

    def test_native_divergence_requires_identical_input(self):
        e=self.episodes()
        # Change a later call's input so the common reset-input guard passes.
        for name in e:
            e[name]['calls'].append(copy.deepcopy(e[name]['calls'][0]));e[name]['steps'][0]['policy_call']=1
        for name in ('native_dynamic','native_dynamic_repeat'):e[name]['calls'][1]['target_input_sha256']='other'
        with self.assertRaisesRegex(ValueError,'equal-input attribution'):compare_episodes(e)


if __name__=='__main__':unittest.main()

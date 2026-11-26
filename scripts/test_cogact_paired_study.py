import copy
import unittest
from scripts.run_cogact_simpler_paired_study import PRIMARY,SUPPLEMENT,check_block


def fixture():
    rows={}
    for name in PRIMARY+SUPPLEMENT:
        row=dict(trajectory=[{'state':0},{'state':1},{'state':2}],success=False,
            steps=[{'action':[0.]*7},{'action':[0.1]*7}],
            calls=[{'target_input_sha256':'input0','normalized':{'sha256':'n0'}},
                   {'target_input_sha256':'input1','normalized':{'sha256':'n1'}}])
        rows[name]=row;rows[name+'_repeat']=copy.deepcopy(row)
    return rows


class CogACTPairedTests(unittest.TestCase):
    def test_complete_primary_and_supplemental_controls(self):
        report=check_block(fixture())
        self.assertIsNone(report['native']['first_action_divergence'])

    def test_released_repeat_is_required(self):
        rows=fixture();rows['released_native_dynamic_repeat']['success']=True
        with self.assertRaisesRegex(RuntimeError,'A/A'):check_block(rows)

    def test_initial_state_includes_supplemental_runs(self):
        rows=fixture()
        for suffix in ('','_repeat'):rows['released_native_dynamic'+suffix]['trajectory'][0]['state']=4
        with self.assertRaisesRegex(RuntimeError,'Canonical'):check_block(rows)

    def test_controlled_divergence_rejects_different_inputs(self):
        rows=fixture()
        for suffix in ('','_repeat'):
            row=rows['native_dynamic'+suffix];row['steps'][1]['action'][0]=0.2;row['calls'][1]['target_input_sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'different inputs'):check_block(rows)

    def test_released_extent_effect_is_explicit(self):
        rows=fixture()
        for suffix in ('','_repeat'):
            row=rows['released_native_dynamic'+suffix];row['steps'][1]['action'][0]=0.2;row['calls'][1]['target_input_sha256']='different_padding'
        report=check_block(rows)['released_native']
        self.assertEqual(report['first_action_divergence'],1)
        self.assertFalse(report['identical_preprocessed_inputs_at_divergence'])

    def test_float64_environment_action_difference_is_retained(self):
        rows=fixture()
        for suffix in ('','_repeat'):rows['native_dynamic'+suffix]['steps'][1]['action'][0]+=1e-12
        self.assertEqual(check_block(rows)['native']['first_action_divergence'],1)


if __name__=='__main__':unittest.main()

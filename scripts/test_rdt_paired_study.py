import copy
import unittest

from scripts.run_rdt_maniskill_paired_study import compare_block


def fixture():
    result={}
    for name in ('native_singleton','native_dynamic','patched_singleton','patched_dynamic'):
        row={'trajectory':[{'state':0},{'state':1},{'state':2}],
             'calls':[{'target_input_sha256':'same0','normalized':{'sha256':'n0'}},
                      {'target_input_sha256':'same1','normalized':{'sha256':'n1'}}],
             'steps':[{'action':[0.0]*8,'policy_call':0},{'action':[0.1]*8,'policy_call':1}],
             'success':False}
        result[name]=row;result[name+'_repeat']=copy.deepcopy(row)
    return result


class PairedBlockTests(unittest.TestCase):
    def test_complete_controls_pass(self):
        self.assertIsNone(compare_block(fixture()))

    def test_repeat_change_is_rejected(self):
        rows=fixture();rows['native_dynamic_repeat']['steps'][1]['action'][0]=0.2
        with self.assertRaisesRegex(RuntimeError,'A/A'):compare_block(rows)

    def test_reset_change_is_rejected_even_if_repeated(self):
        rows=fixture()
        for suffix in ('','_repeat'):rows['native_dynamic'+suffix]['trajectory'][0]['state']=4
        with self.assertRaisesRegex(RuntimeError,'canonical'):compare_block(rows)

    def test_normalized_chunk_change_is_rejected_even_if_trajectory_matches(self):
        rows=fixture()
        for suffix in ('','_repeat'):rows['patched_dynamic'+suffix]['calls'][1]['normalized']['sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'normalized'):compare_block(rows)

    def test_first_divergence_requires_identical_prior_inputs(self):
        rows=fixture()
        for suffix in ('','_repeat'):
            row=rows['native_dynamic'+suffix];row['steps'][1]['action'][0]=0.2
            row['calls'][1]['target_input_sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'attribution'):compare_block(rows)

    def test_attributed_native_divergence_passes(self):
        rows=fixture()
        for suffix in ('','_repeat'):rows['native_dynamic'+suffix]['steps'][1]['action'][0]=0.2
        self.assertEqual(compare_block(rows),1)


if __name__=='__main__':unittest.main()

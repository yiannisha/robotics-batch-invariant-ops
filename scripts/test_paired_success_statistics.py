import unittest
from scripts.summarize_paired_success_progress import CONDITIONS,task_statistics


class PairedStatisticsTests(unittest.TestCase):
    def test_discordance_and_trials(self):
        rows=[]
        for initial,(native,invariant) in enumerate([(True,True),(False,True),(True,False),(False,False)]):
            rows.append(dict(task=0,initialization=initial,outcomes=dict(native_singleton=native,native_dynamic=native,patched_singleton=invariant,patched_dynamic=invariant)))
        report=task_statistics(rows,20250401);pair=report['comparisons']['patched_singleton_minus_native_singleton']
        self.assertEqual(report['paired_starts'],4)
        self.assertEqual(pair['candidate_only_successes'],1);self.assertEqual(pair['reference_only_successes'],1)
        self.assertEqual(pair['both_successes'],1);self.assertEqual(pair['both_failures'],1)
        self.assertEqual(pair['estimate_percentage_points'],0);self.assertEqual(pair['exact_two_sided_mcnemar_p'],1)

    def test_duplicate_starts_are_rejected(self):
        row=dict(task=0,initialization=0,outcomes={name:True for name in CONDITIONS})
        with self.assertRaises(ValueError):task_statistics([row,row],1)

    def test_six_unopposed_discordances(self):
        rows=[dict(task=0,initialization=i,outcomes=dict(native_singleton=False,native_dynamic=False,patched_singleton=True,patched_dynamic=True)) for i in range(6)]
        pair=task_statistics(rows,1)['comparisons']['patched_singleton_minus_native_singleton']
        self.assertEqual(pair['exact_two_sided_mcnemar_p'],0.03125)
        self.assertEqual(pair['estimate_percentage_points'],100)


if __name__=='__main__':unittest.main()

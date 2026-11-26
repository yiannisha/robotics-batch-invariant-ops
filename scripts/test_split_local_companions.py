"""Regression checks for episode-level holdout isolation in batch companions."""
import unittest

from scripts.run_model_replay_invariance import split_local_companion_rows


class SplitLocalCompanionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[]
        for task in range(10):
            for episode in range(10):
                for frame in range(10):
                    cls.rows.append(dict(replay_index=len(cls.rows),task=task,episode_id=episode,
                        observation_index=frame,split='diagnostic' if episode<2 else 'heldout'))

    def test_diagnostic_B32_never_reads_heldout_or_target_episode(self):
        for target in (self.rows[i] for i in (0,9,10,110,510,910)):
            chosen=split_local_companion_rows(self.rows,target,target['replay_index'],31)
            self.assertEqual(len({r['replay_index'] for r in chosen}),31)
            self.assertEqual(len({(r['task'],r['episode_id']) for r in chosen}),19)
            self.assertTrue(all(r['split']=='diagnostic' for r in chosen))
            self.assertTrue(all((r['task'],r['episode_id'])!=(target['task'],target['episode_id']) for r in chosen))

    def test_heldout_B32_uses_31_distinct_other_heldout_episodes(self):
        target=self.rows[20]
        chosen=split_local_companion_rows(self.rows,target,20,31)
        self.assertEqual(len({(r['task'],r['episode_id']) for r in chosen}),31)
        self.assertTrue(all(r['split']=='heldout' for r in chosen))
        self.assertTrue(all((r['task'],r['episode_id'])!=(target['task'],target['episode_id']) for r in chosen))

    def test_sweep_sizes_share_one_frozen_schedule_prefix(self):
        for target in self.rows[::37]:
            largest=split_local_companion_rows(self.rows,target,target['replay_index'],31)
            for count in (1,3,7,15,31):
                selected=split_local_companion_rows(self.rows,target,target['replay_index'],count)
                self.assertEqual([r['replay_index'] for r in selected],[r['replay_index'] for r in largest[:count]])

    def test_does_not_fall_back_to_target_episode_or_other_split(self):
        rows=[dict(replay_index=i,task=0,episode_id=0,observation_index=i,split='diagnostic') for i in range(10)]
        rows.extend(dict(replay_index=i+10,task=1,episode_id=1,observation_index=i,split='heldout') for i in range(100))
        with self.assertRaises(ValueError):split_local_companion_rows(rows,rows[0],0,1)


if __name__=='__main__':unittest.main()

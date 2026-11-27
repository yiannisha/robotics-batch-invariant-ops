"""Source preparation must exclude prior model inputs without policy filtering."""
import copy
import unittest
import torch
from scripts.reselect_lingbot_source_holdout import select,identity,KEYS,signature
from scripts.audit_lingbot_campaign_corpus_v3 import verify_anchor_selection


class Reselection(unittest.TestCase):
    def prepared(self,index):return {k:torch.tensor([index],dtype=torch.float32) for k in KEYS}
    def identity_at(self,index):return identity({k:signature(v) for k,v in self.prepared(index).items()})

    def test_source_call_collision_uses_nearest_lower_frame(self):
        exposed={self.identity_at(100)};original=set(exposed)
        chosen,trace=select(141,exposed,self.prepared)
        self.assertIn(99,chosen);self.assertNotIn(100,chosen);self.assertEqual(len(chosen),8)
        verify_anchor_selection(141,chosen,trace,original,self.identity_at)

    def test_duplicate_prepared_inputs_are_excluded_across_episodes(self):
        exposed=set();first,_=select(30,exposed,self.prepared);second,_=select(30,exposed,self.prepared)
        self.assertFalse(set(first)&set(second))

    def test_auditor_rejects_skipping_an_unexposed_candidate(self):
        chosen,trace=select(30,set(),self.prepared);chosen[0]=2
        with self.assertRaises(ValueError):verify_anchor_selection(30,chosen,trace,set(),self.identity_at)

    def test_auditor_rejects_tampered_rejection_reason(self):
        original={self.identity_at(100)};chosen,trace=select(141,set(original),self.prepared)
        for entry in trace:
            for attempt in entry['attempts']:
                if attempt['already_exposed']:attempt['already_exposed']=False
        with self.assertRaises(ValueError):verify_anchor_selection(141,chosen,trace,set(original),self.identity_at)

    def test_short_or_fully_exposed_episode_fails_without_filtering(self):
        with self.assertRaises(ValueError):select(8,set(),self.prepared)
        with self.assertRaises(ValueError):select(30,{self.identity_at(i) for i in range(30)},self.prepared)


if __name__=='__main__':unittest.main()

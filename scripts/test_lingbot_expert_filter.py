import unittest

from scripts.collect_lingbot_robotwin_campaign_replay_v2 import released_expert_rejection


class ExpertFilterTests(unittest.TestCase):
    def test_released_demo_failure_is_rejected_without_hiding_capture_failure(self):
        response=dict(ok=False,error_type='AssertionError',error='target_pose cannot be None for move action.')
        self.assertTrue(released_expert_rejection({'op':'expert'},response,'Traceback:\n info = env.play_once()\n Action(...)'))
        self.assertFalse(released_expert_rejection({'op':'expert'},response,'Traceback:\n canonical=capture()'))
        self.assertFalse(released_expert_rejection({'op':'step'},response,'info = env.play_once()'))

    def test_only_expert_instability_is_filterable(self):
        response=dict(ok=False,error_type='UnStableError')
        self.assertTrue(released_expert_rejection({'op':'expert'},response,''))
        self.assertFalse(released_expert_rejection({'op':'reset'},response,''))
        self.assertFalse(released_expert_rejection({'op':'expert'},dict(ok=True),''))


if __name__=='__main__':unittest.main()

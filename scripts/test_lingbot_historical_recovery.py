import copy
import unittest
import torch
from scripts.recover_lingbot_historical_pilot_inputs import check_request
from scripts.audit_lingbot_campaign_corpus_v2 import signature


class HistoricalRecovery(unittest.TestCase):
    def test_identical_bytes_match(self):
        value={'state':torch.tensor([0.,-0.]),'noise':torch.tensor([1.])}
        expected={k:signature(v) for k,v in value.items()}
        self.assertEqual(check_request(value,expected),expected)

    def test_changed_noise_state_signed_zero_or_missing_fields_are_rejected(self):
        baseline={'state':torch.tensor([0.,-0.]),'noise':torch.tensor([1.])}
        expected={k:signature(v) for k,v in baseline.items()}
        for key in baseline:
            value=copy.deepcopy(baseline);value[key][-1]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):check_request(value,expected)
        value=copy.deepcopy(baseline);value['state'][1]=0.
        with self.assertRaises(ValueError):check_request(value,expected)
        del value['noise']
        with self.assertRaises(ValueError):check_request(value,expected)


if __name__=='__main__':unittest.main()

import unittest
import torch
from scripts.audit_lingbot_historical_pilot_inputs import tensor_signatures,verify_request


class HistoricalAudit(unittest.TestCase):
    def test_archived_signatures_control_acceptance(self):
        value={'state':torch.tensor([0.,-0.]),'noise':torch.ones(3)}
        expected=tensor_signatures(value)
        self.assertEqual(verify_request(value,expected),expected)
        value['state'][1]=0.
        with self.assertRaises(ValueError):verify_request(value,expected)

    def test_changed_dtype_or_missing_fields_cannot_match(self):
        value={'state':torch.ones(2),'noise':torch.ones(3)}
        expected=tensor_signatures(value)
        with self.assertRaises(ValueError):verify_request({'state':value['state'].double(),'noise':value['noise']},expected)
        with self.assertRaises(ValueError):verify_request({'state':value['state']},expected)
        with self.assertRaises(ValueError):verify_request({'state':[1.,1.],'noise':value['noise']},expected)


if __name__=='__main__':unittest.main()

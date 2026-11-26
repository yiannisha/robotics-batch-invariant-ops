"""Regression checks for the scientific replay equality criterion."""
import unittest

import torch

from scripts.run_model_replay_invariance import metrics


class ReplayMetricsTests(unittest.TestCase):
    def test_signed_zero_is_a_bitwise_violation_with_zero_numeric_error(self):
        result = metrics(torch.tensor([0.]), torch.tensor([-0.]))
        self.assertFalse(result["exact"])
        self.assertEqual(result["unequal_element_count"], 1)
        self.assertEqual(result["max_absolute_error"], 0.)

    def test_distinct_int64_values_survive_the_equality_check(self):
        result = metrics(torch.tensor([2**60]), torch.tensor([2**60+1]))
        self.assertFalse(result["exact"])
        self.assertEqual(result["unequal_element_count"], 1)

    def test_identical_nonfinite_values_never_qualify(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                result = metrics(torch.tensor([value]), torch.tensor([value]))
                self.assertFalse(result["exact"])
                self.assertEqual(result["nonfinite_count"], 1)

    def test_scalar_and_empty_outputs(self):
        self.assertTrue(metrics(torch.tensor(3.), torch.tensor(3.))["exact"])
        self.assertTrue(metrics(torch.empty(0), torch.empty(0))["exact"])

    def test_matching_values_with_distinct_dtypes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            metrics(torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64))

    def test_finite_equal_nested_outputs_have_identical_hashes(self):
        value = {"normalized": torch.tensor([[1., 2.]]), "physical": torch.tensor([[3.]])}
        result = metrics(value, value)
        self.assertTrue(result["exact"])
        for item in result["tensors"].values():
            self.assertEqual(item["reference_sha256"], item["candidate_sha256"])


if __name__ == "__main__":
    unittest.main()

import unittest
import torch
from scripts.leaf_request_projection import project, source_blocks


class RequestProjection(unittest.TestCase):
    def test_conditional_unconditional_target(self):
        reference = torch.tensor([[1.], [10.]])
        candidate = torch.tensor([[1.], [2.], [10.], [20.]])
        self.assertTrue(torch.equal(project(reference, candidate, batch_size=2, blocks=2), reference))

    def test_nonfirst_request_and_multiple_rows_per_block(self):
        candidate = torch.arange(24).reshape(12, 2)
        reference = torch.cat([candidate[4:6], candidate[10:12]])
        self.assertTrue(torch.equal(project(reference, candidate, batch_size=3, target_index=2, blocks=2), reference))

    def test_camera_group_is_contiguous_inside_request(self):
        candidate = torch.arange(18).reshape(6, 3)
        reference = candidate[:3]
        self.assertTrue(torch.equal(project(reference, candidate, batch_size=2), reference))

    def test_shared_or_unknown_shape_is_not_silently_compared(self):
        reference = torch.ones(2, 4)
        self.assertIsNone(project(reference, reference, batch_size=2, blocks=2))
        self.assertIsNone(project(reference, torch.ones(4, 5), batch_size=2, blocks=2))

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            project(torch.ones(1), torch.ones(2), batch_size=2, target_index=2)

    def test_cfg_contract_only_matches_action_network(self):
        adapter = 'scripts.model_invariance.adapters.cogact_simpler_exact'
        self.assertEqual(source_blocks(adapter, 'model.action_model.net.blocks.0.mlp.fc1'), 2)
        self.assertEqual(source_blocks(adapter, 'model.vlm.vision_backbone.featurizer'), 1)


if __name__ == '__main__':
    unittest.main()

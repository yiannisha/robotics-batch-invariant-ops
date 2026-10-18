"""CPU integration checks for UVA's pinned real contexts and action noises."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_invariance.adapters.uva_libero10 import UVALibero10Adapter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cross_model/smolvla/full/replay_manifest.json"


class UVARealReplayTests(unittest.TestCase):
    def test_first_and_last_contexts_match_real_anchor_pixels_and_noise_shapes(self) -> None:
        adapter = UVALibero10Adapter()
        adapter.config = object()
        adapter.num_diffusion_steps = 100
        adapter._policy_tokenizer = lambda tasks, **kwargs: {
            "input_ids": torch.ones(1, 30, dtype=torch.long),
            "attention_mask": torch.ones(1, 30, dtype=torch.long),
        }
        source = json.loads(SOURCE.read_text())
        inventory = json.loads(adapter.context_inventory_path.read_text())
        for index, noise_id in ((0, 0), (999, 2)):
            sample = adapter.load_replay_input(index, noise_id)
            self.assertEqual(sample["images"].shape, (1, 16, 3, 256, 256))
            self.assertEqual(sample["vae_noise"].shape, (1, 4, 16, 16, 16))
            self.assertEqual(sample["action_initial_noise"].shape, (1, 16, 10))
            self.assertEqual(sample["action_step_noise"].shape, (1, 100, 16, 10))
            window = inventory["windows"][index]
            offset = window["anchor_frame_index"] - window["start_frame_index"]
            image_path = Path(source["fixture_root"]) / source["observations"][index]["images"]["image"]
            with Image.open(image_path) as image:
                expected = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1)
            self.assertTrue(torch.equal(sample["images"][0, offset], expected))
        self.assertFalse(torch.equal(adapter.load_replay_input(0, 0)["vae_noise"],
                                     adapter.load_replay_input(0, 1)["vae_noise"]))


if __name__ == "__main__":
    unittest.main()

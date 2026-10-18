"""CPU-only checks for the exact OpenVLA-OFT real-input replay adapter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from model_invariance.adapters.openvla_oft import OpenVLAOFTAdapter


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/workspace/assets/openvla_oft/checkpoint")
MANIFEST = ROOT / "results/cross_model/openvla_oft/full/replay_manifest.json"


class OpenVLAOFTReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        cls.adapter = OpenVLAOFTAdapter()
        cls.adapter.checkpoint = CHECKPOINT
        cls.adapter.replay_manifest_path = MANIFEST
        cls.adapter.processor = AutoProcessor.from_pretrained(
            CHECKPOINT, trust_remote_code=True, local_files_only=True,
        )
        cls.adapter.stats = json.loads((CHECKPOINT / "dataset_statistics.json").read_text())[
            "libero_spatial_no_noops"
        ]
        cls.adapter.replay_manifest = json.loads(MANIFEST.read_text())
        tasks = {row["task"] for row in cls.adapter.replay_manifest["observations"]}
        cls.adapter.canonical_prompt_length = max(
            len(cls.adapter.processor.tokenizer(
                f"In: What action should the robot take to {task.lower()}?\nOut:"
            )["input_ids"]) for task in tasks
        )

    def test_real_observations_and_dynamic_fixed_padding(self) -> None:
        rows = self.adapter.replay_manifest["observations"]
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({row["id"] for row in rows}), 1000)
        first = self.adapter.load_replay_input(0)
        last = self.adapter.load_replay_input(999)
        self.assertEqual(first["pixel_values"].shape, (1, 12, 224, 224))
        self.assertEqual(last["proprio"].shape, (1, 8))
        native = self.adapter.compose_batch([first, last])
        fixed = self.adapter.compose_fixed_batch([first, last])
        self.assertEqual(native["input_ids"].shape[1], max(
            first["input_ids"].shape[1], last["input_ids"].shape[1]
        ))
        self.assertEqual(fixed["input_ids"].shape[1], self.adapter.canonical_prompt_length)
        self.assertTrue(torch.equal(native["input_ids"][0, :first["input_ids"].shape[1]],
                                    first["input_ids"][0]))
        self.assertTrue(torch.equal(fixed["input_ids"][1, :last["input_ids"].shape[1]],
                                    last["input_ids"][0]))
        with self.assertRaises(IndexError):
            self.adapter.load_replay_input(0, noise_id=1)


if __name__ == "__main__":
    unittest.main()

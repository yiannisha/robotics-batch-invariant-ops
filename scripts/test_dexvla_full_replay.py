from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from audit_full_cross_model_campaign import dexvla_source_limit_audit
from model_invariance.adapters.dexvla import DexVLAAdapter
from prepare_dexvla_replay import companion_indices, request_noise, split_membership


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cross_model/dexvla/full"
SOURCE = Path("/workspace/assets/dexvla/sample/episode_47.hdf5")


class _Processor:
    class _Tokenizer:
        eos_token_id = 151645

    tokenizer = _Tokenizer()

    @staticmethod
    def apply_chat_template(messages, **_kwargs):
        assert messages[0]["content"][-1]["text"] == "Cook rice."
        return "processed prompt"

    @staticmethod
    def __call__(*, text, images, **_kwargs):
        assert text == "processed prompt"
        assert len(images) == 3
        return {
            "input_ids": torch.ones((1, 12), dtype=torch.long),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
            "pixel_values": torch.stack([
                image[0, 0].to(torch.float32) for image in images
            ]),
            "image_grid_thw": torch.tensor([[1, 30, 40]] * 3),
        }


def test_source_limited_split_companions_and_noises_are_deterministic():
    splits = split_membership()
    assert list(splits.values()).count("diagnostic") == 200
    assert list(splits.values()).count("heldout") == 196
    for target in (0, 197, 395):
        companions = companion_indices(target)
        assert len(companions) == len(set(companions)) == 31
        assert target not in companions
    assert np.array_equal(request_noise(17, 2), request_noise(17, 2))
    assert not np.array_equal(request_noise(17, 1), request_noise(17, 2))


def test_lazy_adapter_reads_exact_hdf5_frame_and_frozen_noise():
    adapter = DexVLAAdapter()
    adapter.processor = _Processor()
    adapter.stats = {
        "qpos_mean": np.zeros(14, dtype=np.float32),
        "qpos_std": np.ones(14, dtype=np.float32),
    }
    sample = adapter.load_replay_input(395, 2)
    frozen_noise = np.load(FULL / "request_noise.npy", mmap_mode="r")
    with h5py.File(SOURCE, "r") as episode:
        expected_state = np.asarray(episode["observations/qpos"][395])
    assert torch.equal(sample["states"], torch.from_numpy(expected_state).unsqueeze(0))
    assert torch.equal(sample["noise"], torch.from_numpy(np.array(frozen_noise[395, 2])).unsqueeze(0))
    assert tuple(sample["noise"].shape) == (1, 50, 14)
    assert adapter.execution_configuration()["source_limit_observations"] == 396


def test_independent_source_limit_audit_accepts_frozen_replay():
    manifest = json.loads((FULL / "replay_manifest.json").read_text())
    issues = []
    dexvla_source_limit_audit(manifest, manifest["observations"], issues)
    assert issues == []

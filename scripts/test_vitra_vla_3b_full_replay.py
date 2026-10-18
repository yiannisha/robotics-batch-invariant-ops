from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from batch_invariant_ops import set_batch_invariant_mode
from model_invariance.adapters.vitra_vla_3b import VitraVLA3BAdapter
from prepare_vitra_vla_3b_replay import frame_indices, request_noise, task_class


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cross_model/vitra_vla_3b/full"


class _Normalizer:
    state_mean = np.zeros(122, dtype=np.float32)
    state_std = np.ones(122, dtype=np.float32)

    @staticmethod
    def normalize_state(value):
        return np.asarray(value, dtype=np.float32)


class _Processor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    @staticmethod
    def __call__(text, images, return_tensors):
        length = {
            "Pick up mango.": 270,
            "Place mango into white box.": 272,
            "Place mango onto white box.": 272,
            "Pick up mango.Place mango into white box.": 276,
        }[text.split("Right hand: ", 1)[1]]
        return {
            "pixel_values": torch.zeros((1, 3, 224, 224)),
            "input_ids": torch.arange(length).view(1, -1),
            "attention_mask": torch.ones((1, length), dtype=torch.bool),
        }


def test_vitra_selection_noise_and_task_classes_are_deterministic():
    assert len(frame_indices(112)) == 20
    assert frame_indices(112) == sorted(set(frame_indices(112)))
    assert np.array_equal(request_noise(17, 2), request_noise(17, 2))
    assert not np.array_equal(request_noise(17, 1), request_noise(17, 2))
    assert task_class("Place mango onto white box.") == "place_mango"
    assert task_class("Pick up mango.Place mango into white box.") == "pick_mango"


def test_vitra_manifest_has_full_real_replay_contract():
    manifest = json.loads((FULL / "replay_manifest.json").read_text())
    observations = manifest["observations"]
    assert len(observations) == 1000
    assert len({row["physical_observation_sha256"] for row in observations}) == 1000
    assert manifest["split_counts"] == {"diagnostic": 200, "heldout": 800}
    assert manifest["noise_shape"] == [1000, 3, 16, 192]
    assert len({row["episode_id"] for row in observations}) == 50
    assert {row["task"] for row in observations} == {"pick_mango", "place_mango"}
    assert all(len(row["companion_replay_indices"]) == 31 for row in observations)


def test_vitra_lazy_replay_preserves_state_image_fov_and_frozen_noise():
    adapter = VitraVLA3BAdapter()
    adapter.processor = _Processor()
    adapter.normalizer = _Normalizer()
    sample = adapter.load_replay_input(999, 2)
    manifest = json.loads((FULL / "replay_manifest.json").read_text())
    row = manifest["observations"][999]
    xhand = np.asarray(row["xhand_state"], dtype=np.float32)
    raw = adapter._raw_human_state(xhand)
    assert np.array_equal(raw[61:67], xhand[:6])
    assert np.count_nonzero(raw[:61]) == 0
    assert sample["state"].shape == (212,)
    assert sample["state_mask"].sum().item() == 51
    assert sample["pixel_values"].shape == (3, 224, 224)
    assert torch.isfinite(sample["fov"]).all()
    frozen = np.load(FULL / "request_noise.npy", mmap_mode="r")
    assert torch.equal(sample["noise"], torch.from_numpy(np.array(frozen[999, 2])))


def test_vitra_dynamic_and_invariant_language_composition_contracts():
    adapter = VitraVLA3BAdapter()
    adapter.processor = _Processor()
    adapter.normalizer = _Normalizer()
    shorter = adapter.load_replay_input(120, 0)
    longer = adapter.load_replay_input(940, 0)
    native_singleton = adapter.compose_batch([shorter])
    native_pair = adapter.compose_batch([shorter, longer])
    assert native_singleton["input_ids"].shape[1] < 276
    assert native_pair["input_ids"].shape == (2, 276)
    with set_batch_invariant_mode():
        invariant_singleton = adapter.compose_batch([shorter])
        invariant_pair = adapter.compose_batch([shorter, longer])
    assert invariant_singleton["input_ids"].shape == (1, 276)
    assert invariant_pair["input_ids"].shape == (2, 276)
    fixed = adapter.compose_fixed_batch([shorter, longer])
    assert adapter.fixed_batch_shape_metadata(fixed) == {
        "language_padding": "canonical_right",
        "language_tokens": 276,
        "canonical_length": 276,
    }


def test_vitra_execution_configuration_is_exact():
    configuration = VitraVLA3BAdapter().execution_configuration()
    assert configuration["real_xhand_state_dim"] == 18
    assert configuration["padded_model_state_dim"] == 212
    assert configuration["ddim_steps"] == 10
    assert configuration["action_chunk_size"] == 16

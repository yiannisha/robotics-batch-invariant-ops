from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from batch_invariant_ops import set_batch_invariant_mode
from audit_full_cross_model_campaign import xiaomi_source_audit
from model_invariance.adapters.xiaomi_robotics_1_robocasa import (
    XiaomiRobotics1RoboCasaAdapter,
)
from prepare_xiaomi_robotics_1_replay import frame_indices, request_noise


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cross_model/xiaomi_robotics_1/full"


def _sample(tokens: int, marker: int) -> dict:
    return {
        "input_ids": torch.full((1, tokens), marker, dtype=torch.long),
        "attention_mask": torch.ones((1, tokens), dtype=torch.long),
        "pixel_values": torch.zeros((768, 1536)),
        "image_grid_thw": torch.ones((3, 3), dtype=torch.long),
        "state": torch.zeros((1, 1, 60), dtype=torch.bfloat16),
        "action_mask": torch.ones((1, 10, 60), dtype=torch.bfloat16),
        "noise": torch.zeros((10, 60), dtype=torch.bfloat16),
        "seed": marker,
    }


def test_xiaomi_selection_and_explicit_noise_are_deterministic():
    assert frame_indices(268) == sorted(set(frame_indices(268)))
    assert len(frame_indices(268)) == 10
    assert frame_indices(268)[0] == 0
    assert frame_indices(268)[-1] == 267
    assert np.array_equal(request_noise(17, 2), request_noise(17, 2))
    assert not np.array_equal(request_noise(17, 1), request_noise(17, 2))


def test_xiaomi_dynamic_and_invariant_language_composition_contracts():
    adapter = XiaomiRobotics1RoboCasaAdapter()
    adapter.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    shorter, longer = _sample(259, 1), _sample(260, 2)
    native_singleton = adapter.compose_batch([shorter])
    native_pair = adapter.compose_batch([shorter, longer])
    assert native_singleton["input_ids"].shape == (1, 259)
    assert native_pair["input_ids"].shape == (2, 260)
    with set_batch_invariant_mode():
        invariant_singleton = adapter.compose_batch([shorter])
        invariant_pair = adapter.compose_batch([shorter, longer])
    assert invariant_singleton["input_ids"].shape == (1, 260)
    assert invariant_pair["input_ids"].shape == (2, 260)
    fixed = adapter.compose_fixed_batch([shorter, longer])
    assert adapter.fixed_batch_shape_metadata(fixed) == {
        "language_padding": "canonical_right",
        "language_tokens": 260,
        "canonical_length": 260,
    }


def test_xiaomi_execution_configuration_preserves_real_state_contract():
    adapter = XiaomiRobotics1RoboCasaAdapter()
    configuration = adapter.execution_configuration()
    assert configuration["real_state_dim"] == 25
    assert configuration["padded_state_dim"] == 60
    assert configuration["flow_steps"] == 5
    assert configuration["action_chunk_size"] == 10


def test_xiaomi_lazy_replay_preserves_real_state_and_frozen_noise():
    captured = {}

    class _Processor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            captured["messages"] = messages
            captured["state"] = kwargs["state"].clone()
            return {
                "input_ids": torch.ones((1, 260), dtype=torch.long),
                "attention_mask": torch.ones((1, 260), dtype=torch.long),
                "pixel_values": torch.zeros((768, 1536)),
                "image_grid_thw": torch.ones((3, 3), dtype=torch.long),
                "state": kwargs["state"].to(torch.bfloat16),
                "action_mask": torch.ones((1, 10, 60), dtype=torch.bfloat16),
            }

    adapter = XiaomiRobotics1RoboCasaAdapter()
    adapter.processor = _Processor()
    sample = adapter.load_replay_input(999, 2)
    manifest = json.loads((FULL / "replay_manifest.json").read_text())
    real_state = torch.tensor(manifest["observations"][999]["state"])
    assert torch.equal(captured["state"][0, 0, :25], real_state)
    assert torch.count_nonzero(captured["state"][0, 0, 25:]) == 0
    frozen = np.load(FULL / "request_noise.npy", mmap_mode="r")
    assert torch.equal(
        sample["noise"], torch.from_numpy(np.array(frozen[999, 2])).to(torch.bfloat16)
    )
    assert manifest["observations"][999]["task"] in str(captured["messages"])


def test_xiaomi_independent_source_audit_accepts_frozen_replay():
    manifest = json.loads((FULL / "replay_manifest.json").read_text())
    issues: list[str] = []
    xiaomi_source_audit(manifest, manifest["observations"], issues)
    assert issues == []

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from audit_full_cross_model_campaign import lingbot_vla2_source_limit_audit
from model_invariance.adapters.lingbot_vla2 import LingBotVLA2Adapter
from prepare_lingbot_vla2_replay import companion_indices, request_noise, split_membership


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cross_model/lingbot_vla2/full"
SOURCE = Path("/workspace/assets/dexvla/sample/episode_47.hdf5")


class _Server:
    @staticmethod
    def _prepare_model_input(observation):
        assert observation["task"] == "Cook rice."
        assert observation["observation.state"].shape == (14,)
        assert all(
            observation[f"observation.images.{camera}"].shape == (480, 640, 3)
            for camera in LingBotVLA2Adapter.camera_names
        )
        return {
            "images": torch.zeros((3, 3, 2, 2)),
            "img_masks": torch.ones(3, dtype=torch.bool),
            "lang_tokens": torch.zeros(8, dtype=torch.long),
            "lang_masks": torch.ones(8, dtype=torch.bool),
            "state": torch.from_numpy(observation["observation.state"]),
            "image_grid_thw": torch.ones((3, 3), dtype=torch.long),
        }


def test_source_limited_schedule_and_noise_are_deterministic():
    splits = split_membership()
    assert list(splits.values()).count("diagnostic") == 200
    assert list(splits.values()).count("heldout") == 196
    for target in (0, 197, 395):
        companions = companion_indices(target)
        assert len(companions) == len(set(companions)) == 31
        assert target not in companions
    assert np.array_equal(request_noise(17, 2), request_noise(17, 2))
    assert not np.array_equal(request_noise(17, 1), request_noise(17, 2))


def test_lazy_adapter_reads_exact_frame_and_frozen_noise():
    adapter = LingBotVLA2Adapter()
    adapter.server = _Server()
    sample = adapter.load_replay_input(395, 2)
    frozen_noise = np.load(FULL / "request_noise.npy", mmap_mode="r")
    with h5py.File(SOURCE, "r") as episode:
        expected_state = np.asarray(episode["observations/qpos"][395])
    assert torch.equal(sample["state"], torch.from_numpy(expected_state))
    assert torch.equal(sample["noise"], torch.from_numpy(np.array(frozen_noise[395, 2])))
    assert tuple(sample["noise"].shape) == (50, 55)
    assert adapter.execution_configuration()["source_limit_observations"] == 396


def test_independent_source_limit_audit_accepts_frozen_replay():
    manifest = json.loads((FULL / "replay_manifest.json").read_text())
    issues: list[str] = []
    lingbot_vla2_source_limit_audit(manifest, manifest["observations"], issues)
    assert issues == []


def test_adapter_traces_every_flow_prediction_and_updated_state():
    class _Flow:
        config = SimpleNamespace(num_steps=2)

        def predict_velocity(self, state, _mask, _cache, x_t, _time):
            return torch.ones_like(x_t)

        def sample_actions(self, **inputs):
            state = inputs["state"]
            x_t = inputs["noise"]
            for step in range(2):
                velocity = self.predict_velocity(state, None, None, x_t, step)
                x_t += -0.5 * velocity
            return x_t

    adapter = LingBotVLA2Adapter()
    traced = {}
    adapter.trace_callback = lambda label, value: traced.setdefault(label, value.detach().clone())
    batch = {
        "images": torch.zeros((1, 3, 3, 2, 2)),
        "img_masks": torch.ones((1, 3), dtype=torch.bool),
        "lang_tokens": torch.zeros((1, 8), dtype=torch.long),
        "lang_masks": torch.ones((1, 8), dtype=torch.bool),
        "state": torch.zeros((1, 14)),
        "image_grid_thw": torch.ones((1, 3, 3), dtype=torch.long),
        "noise": torch.zeros((1, 50, 55)),
    }
    result = adapter.run_model(_Flow(), batch)
    assert set(traced) == {
        "flow.step_00.prediction", "flow.step_00.state",
        "flow.step_01.prediction", "flow.step_01.state",
    }
    assert torch.equal(
        traced["flow.step_00.state"], torch.full_like(traced["flow.step_00.state"], -0.5)
    )
    assert torch.equal(
        traced["flow.step_01.state"], torch.full_like(traced["flow.step_01.state"], -1.0)
    )
    assert torch.equal(result, torch.full((1, 50, 55), -1.0))

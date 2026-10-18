"""CPU-only contracts for UniVLA's full replay adapter."""

from __future__ import annotations

from types import SimpleNamespace
from collections import deque

import numpy as np
import torch

from model_invariance.adapters.univla_baai import EOA_TOKEN_ID, UniVLAAdapter


class _Processor:
    def __init__(self) -> None:
        self.calls = []

    def video_process(self, **kwargs):
        self.calls.append(kwargs)
        batch = len(kwargs["text"])
        length = kwargs.get("max_length", 5)
        return SimpleNamespace(
            input_ids=torch.ones(batch, length, dtype=torch.long),
            attention_mask=torch.ones(batch, length, dtype=torch.long),
        )


class _Model:
    def generate(self, input_ids, generation_config, **kwargs):
        batch = input_ids.shape[0]
        generated = torch.tensor([[9, EOA_TOKEN_ID]] * batch, dtype=torch.long)
        return SimpleNamespace(
            sequences=torch.cat([input_ids, generated], dim=1),
            scores=(torch.arange(batch * 12).reshape(batch, 12).float(),) * 2,
        )


class _ActionTokenizer:
    @staticmethod
    def decode(tokens, *, time_horizon, action_dim):
        assert tuple(tokens.shape) == (1, 1)
        return np.zeros((1, time_horizon, action_dim), dtype=np.float32)


def _samples():
    return [
        {
            "video_tokens": torch.full((1, 2, 2), index, dtype=torch.long),
            "gripper_tokens": torch.full((1, 2, 2), index + 10, dtype=torch.long),
            "task": f"task {index}",
        }
        for index in range(2)
    ]


def test_compose_preserves_task_order_and_fixed_shape_contract() -> None:
    adapter = UniVLAAdapter()
    adapter.canonical_prompt_length = 17
    dynamic = adapter.compose_batch(_samples())
    fixed = adapter.compose_fixed_batch(_samples())
    assert dynamic["tasks"] == ["task 0", "task 1"]
    assert tuple(dynamic["video_tokens"].shape) == (2, 2, 2)
    assert dynamic.fixed_language is False
    assert fixed.fixed_language is True
    assert adapter.fixed_batch_shape_metadata(fixed) == {
        "language_padding": "canonical_right",
        "language_tokens": 17,
        "canonical_length": 17,
    }


def test_greedy_run_traces_each_autoregressive_score_step() -> None:
    adapter = UniVLAAdapter()
    adapter.device = torch.device("cpu")
    adapter.canonical_prompt_length = 7
    processor = _Processor()
    policy = SimpleNamespace(
        model=_Model(), processor=processor, generation_config=object(),
        logits_processor=object(), allowed_token_ids=torch.tensor([1, 3]),
        last_action_token_id=10, action_tokenizer=_ActionTokenizer(),
    )
    traced = {}
    adapter.trace_callback = lambda label, value: traced.setdefault(label, value.clone())
    output = adapter.run_model(policy, adapter.compose_fixed_batch(_samples()))
    assert tuple(output.shape) == (2, 10, 7)
    assert set(label for label in traced if label.startswith("flow.step_")) == {
        "flow.step_000.scores", "flow.step_001.scores",
    }
    assert tuple(traced["flow.step_000.scores"].shape) == (2, 2)
    assert processor.calls[-1]["text"] == ["task 0", "task 1"]
    assert processor.calls[-1]["padding"] == "max_length"
    assert processor.calls[-1]["max_length"] == 7


def test_deterministic_replay_rejects_noise_ids() -> None:
    adapter = UniVLAAdapter()
    adapter.image_processor = object()
    adapter.image_tokenizer = object()
    try:
        adapter.load_replay_input(0, 1)
    except IndexError as error:
        assert "no request-local noise" in str(error)
    else:
        raise AssertionError("UniVLA accepted a nonexistent stochastic request")


def test_runtime_image_conversion_preserves_uint8_pixels() -> None:
    pixels = torch.tensor(
        [[[0.0, 1.0], [0.5, 0.25]], [[1.0, 0.0], [0.5, 0.75]], [[0.0, 0.5], [1.0, 0.25]]]
    )
    image = UniVLAAdapter._runtime_image_to_pil(pixels)
    array = np.asarray(image)
    assert array.shape == (2, 2, 3)
    assert array.dtype == np.uint8
    assert array[0, 0].tolist() == [0, 255, 0]
    assert array[0, 1].tolist() == [255, 0, 128]


def test_univla_paired_facade_reuses_ten_action_chunk() -> None:
    from scripts.run_univla_paired_libero import UniVLAPairedPolicy

    class FakeAdapter:
        def __init__(self):
            self.runs = 0

        def prepare_runtime_sample(self, *, image, wrist_image, task):
            return {"image": image, "wrist": wrist_image, "task": task}

        def compose_batch(self, samples):
            return samples

        def run_model(self, _model, samples):
            self.runs += 1
            batch = len(samples)
            values = torch.arange(batch * 10 * 7).reshape(batch, 10, 7).float()
            return values

    policy = UniVLAPairedPolicy.__new__(UniVLAPairedPolicy)
    torch.nn.Module.__init__(policy)
    policy.adapter = FakeAdapter()
    policy.model = object()
    policy.config = SimpleNamespace(device="cpu", n_action_steps=10)
    policy._action_queue = deque(maxlen=10)
    batch = {
        "observation.state": torch.zeros(2, 8),
        "observation.images.image": torch.zeros(2, 3, 4, 4),
        "observation.images.image2": torch.ones(2, 3, 4, 4),
        "task": ["task zero", "task one"],
    }
    first = policy.select_action(batch)
    second = policy.select_action(batch)
    assert tuple(first.shape) == (2, 7)
    assert torch.equal(first, torch.tensor([[0, 1, 2, 3, 4, 5, 6], [70, 71, 72, 73, 74, 75, 76]]).float())
    assert torch.equal(second, first + 7)
    assert policy.adapter.runs == 1
    policy.reset()
    assert len(policy._action_queue) == 0

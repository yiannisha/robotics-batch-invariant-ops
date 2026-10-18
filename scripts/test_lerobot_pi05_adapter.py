from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from model_invariance.adapters.lerobot_pi05 import LeRobotPi05Adapter


def test_pi05_replay_uses_two_real_views_and_official_missing_camera_contract(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    Image.fromarray(np.full((5, 7, 3), 17, dtype=np.uint8)).save(fixture / "base.png")
    Image.fromarray(np.full((5, 7, 3), 31, dtype=np.uint8)).save(fixture / "wrist.png")

    captured = {}

    def preprocess(value):
        captured.update(value)
        return {
            "observation.images.base_0_rgb": value["observation.images.base_0_rgb"].unsqueeze(0),
            "observation.images.left_wrist_0_rgb": value[
                "observation.images.left_wrist_0_rgb"
            ].unsqueeze(0),
            "observation.state": value["observation.state"].unsqueeze(0),
            "observation.language.tokens": torch.zeros((1, 200), dtype=torch.long),
            "observation.language.attention_mask": torch.ones((1, 200), dtype=torch.bool),
        }

    adapter = LeRobotPi05Adapter()
    adapter.config = SimpleNamespace(
        image_features=(
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ),
        tokenizer_max_length=200,
        num_inference_steps=10,
        chunk_size=50,
        max_action_dim=32,
    )
    adapter.preprocessor = preprocess
    adapter.fixture_root = fixture
    adapter.replay_manifest = {
        "observations": [{
            "images": {"image": "base.png", "image2": "wrist.png"},
            "state": [0.0] * 8,
            "task": "put the mug on the plate",
        }]
    }

    processed = adapter._processed_replay_observation(0)
    assert tuple(processed["observation.language.tokens"].shape) == (1, 200)
    assert "observation.images.right_wrist_0_rgb" not in captured
    assert captured["task"] == "put the mug on the plate"
    assert tuple(captured["observation.images.base_0_rgb"].shape) == (3, 5, 7)


def test_pi05_composition_and_execution_shape_contracts():
    adapter = LeRobotPi05Adapter()
    adapter.config = SimpleNamespace(
        image_features=(
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ),
        tokenizer_max_length=200,
        num_inference_steps=10,
        chunk_size=50,
        max_action_dim=32,
    )
    sample = {
        "observation.language.tokens": torch.zeros((1, 200), dtype=torch.long),
        "noise": torch.zeros((1, 50, 32)),
    }
    batch = adapter.compose_fixed_batch([sample, sample])
    assert tuple(batch["noise"].shape) == (2, 50, 32)
    assert adapter.fixed_batch_shape_metadata(batch) == {
        "language_padding": "max_length_right",
        "language_tokens": 200,
        "canonical_length": 200,
    }
    assert adapter.execution_configuration() == {
        "language_padding": "max_length_right",
        "language_max_length": 200,
        "camera_features": [
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ],
        "missing_camera_contract": "official_zero_image_false_mask",
        "flow_steps": 10,
        "chunk_size": 50,
        "max_action_dim": 32,
    }

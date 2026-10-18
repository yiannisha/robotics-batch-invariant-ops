from types import SimpleNamespace

import torch

from scripts.model_invariance.adapters.lerobot_pi0fast import LeRobotPi0FastAdapter


class FakeTextTokenizer:
    vocab_size = 1000

    def encode(self, text, add_special_tokens=False):
        assert text == "Action: "
        return [1, 2, 3]

    def convert_ids_to_tokens(self, values):
        mapping = {1: "Action", 2: ":", 3: "_", 99: "invalid", 0: "<pad>"}
        return [mapping.get(value, "x") for value in values]

    def convert_tokens_to_ids(self, values):
        reverse = {"Action": 1, ":": 2, "_": 3, "invalid": 99, "<pad>": 0, "x": 4}
        return [reverse[value] for value in values]


def test_invalid_official_prefix_is_retained_as_categorical_robot_failure() -> None:
    adapter = LeRobotPi0FastAdapter()
    adapter.config = SimpleNamespace(
        validate_action_token_prefix=True,
        n_action_steps=2,
        output_features={"action": SimpleNamespace(shape=(1,))},
        fast_skip_tokens=0,
    )
    adapter.text_tokenizer = FakeTextTokenizer()
    adapter.action_tokenizer = SimpleNamespace(
        bpe_tokenizer=SimpleNamespace(decode=lambda values: ""),
        min_token=0,
        scale=1.0,
    )
    adapter.action_mean = torch.zeros(1)
    adapter.action_std = torch.ones(1)
    actions, valid = adapter._decode_actions(torch.tensor([[99, 0, 0]]))
    assert valid.tolist() == [False]
    assert actions.shape == (1, 2, 1)
    assert torch.isnan(actions).all()

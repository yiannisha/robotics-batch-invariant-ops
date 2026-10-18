"""CPU checks for deterministic request identity in PI0-FAST paired rollouts."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from run_lerobot_pi0fast_paired_libero import (
    CONTRACT_SHA256,
    attach_pi0fast_policy_hooks,
    install_verified_processor_overrides,
    verify_checkpoint_loaded,
)


class DummyPolicy:
    def __init__(self, temperature: float = 0.0) -> None:
        self.config = SimpleNamespace(
            temperature=temperature, use_kv_cache=True, max_decoding_steps=256,
        )
        self._paired_first_sensor_hashes = []
        self._paired_first_noise_hashes = []
        self._paired_state_trajectory = []

    def select_action(self, batch):
        return torch.zeros(batch["observation.state"].shape[0], 7)


class PI0FastPairedTests(unittest.TestCase):
    def test_serialized_processors_use_verified_local_tokenizers(self) -> None:
        calls = []

        def make_pre_post_processors(*args, **kwargs):
            calls.append((args, kwargs))
            return "pre", "post"

        evaluator = SimpleNamespace(make_pre_post_processors=make_pre_post_processors)
        text = Path("/verified/text"); action = Path("/verified/action")
        install_verified_processor_overrides(evaluator, text, action)
        self.assertEqual(
            evaluator.make_pre_post_processors(
                policy_cfg="cfg", preprocessor_overrides={"device_processor": {"device": "cuda"}},
            ),
            ("pre", "post"),
        )
        overrides = calls[0][1]["preprocessor_overrides"]
        self.assertEqual(overrides["device_processor"], {"device": "cuda"})
        self.assertEqual(
            overrides["tokenizer_processor"]["tokenizer_name"], str(text.resolve())
        )
        self.assertEqual(
            overrides["action_tokenizer_processor"]["action_tokenizer_name"],
            str(action.resolve()),
        )
        self.assertEqual(
            overrides["action_tokenizer_processor"]["paligemma_tokenizer_name"],
            str(text.resolve()),
        )
        with self.assertRaisesRegex(RuntimeError, "conflicting PI0-FAST"):
            evaluator.make_pre_post_processors(
                preprocessor_overrides={
                    "tokenizer_processor": {"tokenizer_name": "/unverified"},
                }
            )

    def test_checkpoint_verification_rejects_extra_and_wrong_tensors(self) -> None:
        class Policy:
            def __init__(self, state):
                self.state = state

            def state_dict(self):
                return self.state

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            save_file({"model.weight": torch.tensor([1.0, 2.0])}, path)
            report = verify_checkpoint_loaded(
                Policy({"model.weight": torch.tensor([1.0, 2.0])}), path, "test-hash"
            )
            self.assertEqual(report["tensors_checked"], 1)
            self.assertEqual(report["source_bytes_checked"], 8)
            with self.assertRaisesRegex(RuntimeError, "key sets differ"):
                verify_checkpoint_loaded(
                    Policy({"model.weight": torch.tensor([1.0, 2.0]),
                            "extra": torch.tensor([0.0])}), path, "test-hash"
                )
            with self.assertRaisesRegex(RuntimeError, "did not load pinned"):
                verify_checkpoint_loaded(
                    Policy({"model.weight": torch.tensor([1.0, 3.0])}), path, "test-hash"
                )

    def test_greedy_contract_is_shared_across_batch_positions(self) -> None:
        policy = DummyPolicy()
        attach_pi0fast_policy_hooks(policy, "pi_fast", False)
        self.assertEqual(policy._paired_deterministic_contract_sha256, CONTRACT_SHA256)

    def test_stochastic_checkpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "deterministic decoding"):
            attach_pi0fast_policy_hooks(DummyPolicy(temperature=0.5), "pi_fast", False)


if __name__ == "__main__":
    unittest.main()

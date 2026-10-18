import torch

from batch_invariant_ops import set_batch_invariant_mode
from model_invariance.adapters.cogact_small import (
    CANONICAL_LANGUAGE_TOKENS,
    CogACTSmallAdapter,
)


def sample(value: float):
    return {
        "image": object(), "instruction": f"instruction-{value}",
        "noise": torch.full((1, 16, 7), value),
    }


def test_cogact_composition_marks_only_repaired_and_fixed_batches_canonical():
    adapter = CogACTSmallAdapter()
    native = adapter.compose_batch([sample(1), sample(2)])
    with set_batch_invariant_mode():
        repaired = adapter.compose_batch([sample(1), sample(2)])
    fixed = adapter.compose_fixed_batch([sample(1), sample(2)])
    assert native.fixed_language is False
    assert repaired.fixed_language is True
    assert fixed.fixed_language is True
    assert native["noise"].shape == (2, 16, 7)


def test_cogact_execution_and_fixed_shape_contracts_are_pinned():
    adapter = CogACTSmallAdapter()
    assert adapter.execution_configuration()["canonical_language_tokens"] == 27
    batch = adapter.compose_fixed_batch([sample(1)])
    assert adapter.fixed_batch_shape_metadata(batch) == {
        "language_padding": "canonical_right",
        "language_tokens": CANONICAL_LANGUAGE_TOKENS,
        "canonical_length": CANONICAL_LANGUAGE_TOKENS,
    }

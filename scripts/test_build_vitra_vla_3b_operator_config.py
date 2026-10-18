from __future__ import annotations

import build_vitra_vla_3b_operator_config as builder
from model_invariance.adapters.vitra_vla_3b import VitraVLA3BAdapter


def test_vitra_operator_builder_matches_adapter_execution_contract():
    assert builder.EXPECTED_ADAPTER == "scripts.model_invariance.adapters.vitra_vla_3b"
    assert (
        builder.EXPECTED_EXECUTION_CONFIGURATION
        == VitraVLA3BAdapter().execution_configuration()
    )


def test_vitra_operator_builder_pins_exact_checkpoint():
    assert builder.CHECKPOINT_BYTES == 15_067_578_877
    assert builder.CHECKPOINT_SHA256 == (
        "7f57dfd4451d68f41accab8a06bb388d6fdc2cce87ee8b31fafa7c8f01d83796"
    )

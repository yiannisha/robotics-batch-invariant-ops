from pathlib import Path

from build_pi0fast_operator_config import sdpa_configurations
from build_internvla_a15_operator_config import BACKENDS, validate_trace


def test_grouped_query_trace_preserves_independent_qkv_heads_and_strides() -> None:
    trace = {
        "operations": [{
            "aten_operation": "aten::scaled_dot_product_attention",
            "arguments": [
                {"shape": [8, 8, 1, 256], "stride": [2048, 256, 256, 1]},
                {"shape": [8, 1, 970, 256], "stride": [248320, 248320, 256, 1]},
                {"shape": [8, 1, 970, 256], "stride": [248320, 248320, 256, 1]},
            ],
        }],
    }
    config = sdpa_configurations(trace, Path("/tmp/trace.json"))[0]
    assert config["enable_gqa"] is True
    assert (config["query_heads"], config["key_heads"], config["value_heads"]) == (8, 1, 1)
    assert config["source_q_stride"] == [2048, 256, 256, 1]
    assert config["source_k_stride"] == [248320, 248320, 256, 1]
    assert config["source_v_stride"] == [248320, 248320, 256, 1]


def test_shared_trace_builder_stamps_model_specific_identity() -> None:
    trace = {
        "operations": [{
            "aten_operation": "aten::scaled_dot_product_attention",
            "arguments": [
                {"shape": [8, 16, 261, 64], "stride": [267264, 64, 1024, 1]},
                {"shape": [8, 16, 261, 64], "stride": [267264, 64, 1024, 1]},
                {"shape": [8, 16, 261, 64], "stride": [267264, 64, 1024, 1]},
            ],
        }],
    }
    config = sdpa_configurations(
        trace, Path("/tmp/openvla-trace.json"),
        id_prefix="openvla_oft", source_name="OpenVLA-OFT",
    )[0]
    assert config["id"] == "openvla_oft_sdpa_trace_00"
    assert config["source_operation"].startswith("OpenVLA-OFT traced")


def test_internvla_trace_backend_identity_is_fail_closed(tmp_path: Path) -> None:
    _, optimized = BACKENDS["internvla_a15_optimized"]
    trace = tmp_path / "trace.json"
    trace.write_text(__import__("json").dumps({
        "schema_version": 1,
        "adapter": "scripts.model_invariance.adapters.internvla_a15",
        "adapter_execution_configuration": optimized,
        "batch_size": 8,
        "request_ids": [f"request-{index}" for index in range(8)],
        "replay_manifest_sha256": "abc",
        "gpu": "gpu",
        "torch": "torch",
    }))
    assert validate_trace(trace, 8, optimized)["adapter_execution_configuration"] == optimized
    try:
        validate_trace(trace, 8, None)
    except ValueError:
        pass
    else:
        raise AssertionError("optimized trace was accepted as standard backend evidence")

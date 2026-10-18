import json
import sys
from pathlib import Path

import build_pi05_operator_config as builder


def trace(batch_size: int, operations: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "adapter": builder.EXPECTED_ADAPTER,
        "adapter_execution_configuration": builder.EXPECTED_EXECUTION_CONFIGURATION,
        "batch_size": batch_size,
        "request_ids": [f"request-{index}" for index in range(batch_size)],
        "replay_manifest_sha256": "manifest-hash",
        "gpu": "gpu",
        "torch": "torch",
        "operations": operations,
    }


def test_pi05_builder_carries_traced_attention_and_real_architecture_shapes(
    tmp_path: Path, monkeypatch,
) -> None:
    operation = {
        "aten_operation": "aten::scaled_dot_product_attention",
        "arguments": [
            {"shape": [8, 8, 50, 256], "stride": [102400, 12800, 256, 1]},
            {"shape": [8, 1, 968, 256], "stride": [247808, 247808, 256, 1]},
            {"shape": [8, 1, 968, 256], "stride": [247808, 247808, 256, 1]},
        ],
    }
    b1, b8 = tmp_path / "b1.json", tmp_path / "b8.json"
    b1.write_text(json.dumps(trace(1, [])))
    b8.write_text(json.dumps(trace(8, [operation])))
    checkpoint, output = tmp_path / "model.safetensors", tmp_path / "config.json"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_pi05_operator_config.py",
            "--trace-b1",
            str(b1),
            "--trace-b8",
            str(b8),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )
    builder.main()
    report = json.loads(output.read_text())
    configs = {item["id"]: item for item in report["configurations"]}
    assert configs["siglip_patch_conv_contiguous"]["weight_shape"] == [1152, 3, 14, 14]
    assert configs["action_input_projection_addmm"]["m_per_request"] == 50
    assert configs["pi05_sdpa_trace_00"]["enable_gqa"] is True
    assert configs["pi05_sdpa_trace_00"]["key_length"] == 968
    assert report["adapter_execution_configuration"] == builder.EXPECTED_EXECUTION_CONFIGURATION


from __future__ import annotations

import json
import sys

import build_xiaomi_robotics_1_operator_config as builder


def _tensor(shape, stride):
    return {"shape": shape, "stride": stride, "dtype": "torch.bfloat16", "device": "cuda"}


def _trace(batch_size: int, implementation: str, operations: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "adapter": builder.EXPECTED_ADAPTER,
        "adapter_execution_configuration": builder.EXPECTED_EXECUTION_CONFIGURATION,
        "implementation": implementation,
        "batch_size": batch_size,
        "request_ids": [f"request-{index}" for index in range(batch_size)],
        "replay_manifest_sha256": "2" * 64,
        "gpu": "H100",
        "torch": "test",
        "operations": operations,
    }


def test_xiaomi_builder_retains_real_state_projection_shape(tmp_path, monkeypatch):
    b1, b8 = tmp_path / "b1.json", tmp_path / "b8.json"
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "configurations.json"
    checkpoint.mkdir()
    for index in range(1, 4):
        (checkpoint / f"model-{index:05d}-of-00003.safetensors").write_bytes(bytes([index]))
    b1.write_text(json.dumps(_trace(1, "native", [])))
    matrix = {
        "aten_operation": "aten::addmm",
        "arguments": [
            _tensor([1024], [1]), _tensor([8, 60], [60, 1]),
            _tensor([60, 1024], [1024, 1]),
        ],
        "keyword_arguments": {}, "count": 1,
    }
    convolution = {
        "aten_operation": "aten::convolution",
        "arguments": [
            _tensor([24, 3, 256, 256], [196608, 65536, 256, 1]),
            _tensor([1152, 3, 14, 14], [588, 196, 14, 1]),
            _tensor([1152], [1]), [14, 14], [0, 0], [1, 1], False, [0, 0], 1,
        ],
        "keyword_arguments": {}, "count": 1,
    }
    b8.write_text(json.dumps(_trace(8, "batch_invariant", [matrix, convolution])))
    monkeypatch.setattr(sys, "argv", [
        "build_xiaomi_robotics_1_operator_config.py", "--trace-b1", str(b1),
        "--trace-b8", str(b8), "--checkpoint", str(checkpoint),
        "--output", str(output),
    ])
    builder.main()
    report = json.loads(output.read_text())
    addmm = next(row for row in report["configurations"] if row["operation"] == "addmm")
    assert addmm["k"] == 60
    assert addmm["n"] == 1024
    assert len(report["checkpoint_weight_shards"]) == 3

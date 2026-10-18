from __future__ import annotations

import json
import sys

import build_lingbot_vla2_operator_config as builder


def _tensor(shape, stride):
    return {"shape": shape, "stride": stride, "dtype": "torch.float32", "device": "cuda"}


def _trace(batch_size: int, implementation: str, operations: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "adapter": builder.EXPECTED_ADAPTER,
        "adapter_execution_configuration": builder.EXPECTED_EXECUTION_CONFIGURATION,
        "implementation": implementation,
        "batch_size": batch_size,
        "request_ids": [f"request-{index}" for index in range(batch_size)],
        "replay_manifest_sha256": "1" * 64,
        "gpu": "H100",
        "torch": "test",
        "operations": operations,
    }


def test_builder_retains_every_traced_operator_class(tmp_path, monkeypatch):
    b1 = tmp_path / "b1.json"
    b8 = tmp_path / "b8.json"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "configurations.json"
    checkpoint.mkdir()
    for index in range(1, 7):
        (checkpoint / f"model-{index:05d}-of-00006.safetensors").write_bytes(bytes([index]))
    b1.write_text(json.dumps(_trace(1, "native", [])))
    qkv = _tensor([8, 16, 50, 80], [64000, 4000, 80, 1])
    operations = [
        {
            "aten_operation": "aten::convolution",
            "arguments": [
                _tensor([24, 3, 480, 640], [921600, 307200, 640, 1]),
                _tensor([1280, 3, 14, 14], [588, 196, 14, 1]),
                _tensor([1280], [1]), [14, 14], [0, 0], [1, 1], False, [0, 0], 1,
            ],
            "keyword_arguments": {}, "count": 1,
        },
        {
            "aten_operation": "aten::addmm",
            "arguments": [
                _tensor([768], [1]), _tensor([8, 55], [55, 1]),
                _tensor([55, 768], [768, 1]),
            ],
            "keyword_arguments": {}, "count": 1,
        },
        {
            "aten_operation": "aten::mean",
            "arguments": [_tensor([8, 50, 1280], [64000, 1280, 1]), [2], True],
            "keyword_arguments": {}, "count": 1,
        },
        {
            "aten_operation": "aten::scaled_dot_product_attention",
            "arguments": [qkv, qkv, qkv, None, 0.0, False],
            "keyword_arguments": {}, "count": 1,
        },
    ]
    b8.write_text(json.dumps(_trace(8, "batch_invariant", operations)))
    monkeypatch.setattr(sys, "argv", [
        "build_lingbot_vla2_operator_config.py", "--trace-b1", str(b1),
        "--trace-b8", str(b8), "--checkpoint", str(checkpoint),
        "--output", str(output),
    ])
    builder.main()
    report = json.loads(output.read_text())
    operations = {row["operation"] for row in report["configurations"]}
    assert {"conv2d", "addmm", "mean", "sdpa"}.issubset(operations)
    assert len(report["checkpoint_weight_shards"]) == 6

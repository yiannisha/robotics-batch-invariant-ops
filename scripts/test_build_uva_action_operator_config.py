import json
from pathlib import Path

from build_uva_action_operator_config import main


def tensor(shape, stride=None):
    if stride is None:
        stride = [1] * len(shape)
        for index in range(len(shape) - 2, -1, -1):
            stride[index] = stride[index + 1] * shape[index + 1]
    return {"shape": shape, "stride": stride, "dtype": "torch.float32", "device": "cuda"}


def trace(batch_size: int, operations: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "adapter": "scripts.model_invariance.adapters.uva_libero10_full",
        "adapter_execution_configuration": None,
        "batch_size": batch_size,
        "request_ids": [f"request-{index}" for index in range(batch_size)],
        "replay_manifest_sha256": "a" * 64,
        "gpu": "test-gpu", "torch": "test-torch", "operations": operations,
    }


def test_builder_preserves_uva_transposed_convolution(monkeypatch, tmp_path: Path) -> None:
    addmm = {
        "aten_operation": "aten::addmm",
        "arguments": [tensor([2048]), tensor([240, 512]), tensor([512, 2048])],
        "keyword_arguments": {}, "count": 1,
    }
    transposed = {
        "aten_operation": "aten::convolution",
        "arguments": [
            tensor([8, 768, 4, 16, 16]), tensor([768, 768, 4, 1, 1]), tensor([768]),
            [4, 1, 1], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1,
        ],
        "keyword_arguments": {}, "count": 1,
    }
    b1 = tmp_path / "b1.json"
    b8 = tmp_path / "b8.json"
    checkpoint = tmp_path / "libero10.ckpt"
    output = tmp_path / "configurations.json"
    b1.write_text(json.dumps(trace(1, [])))
    b8.write_text(json.dumps(trace(8, [addmm, transposed])))
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("sys.argv", [
        "build_uva_action_operator_config.py", "--trace-b1", str(b1),
        "--trace-b8", str(b8), "--checkpoint", str(checkpoint),
        "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text())
    configuration = next(
        item for item in report["configurations"]
        if item["operation"] == "conv_transpose3d"
    )
    assert configuration["m_per_request"] == 1
    assert configuration["weight_shape"] == [768, 768, 4, 1, 1]
    assert configuration["stride"] == [4, 1, 1]
    assert configuration["source_state_key"] == "model.diffactloss.conv_transpose3d.weight"


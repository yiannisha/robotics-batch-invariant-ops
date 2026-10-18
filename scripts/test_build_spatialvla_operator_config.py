import pytest

from build_spatialvla_operator_config import (
    convolution_configurations,
    matrix_configurations,
    reduction_configurations,
)


def tensor(shape, stride):
    return {"shape": shape, "stride": stride, "dtype": "torch.bfloat16", "device": "cuda"}


def test_spatialvla_builder_preserves_traced_operator_shapes(tmp_path) -> None:
    trace_path = tmp_path / "aten_trace_B8.json"
    trace = {"operations": [
        {
            "aten_operation": "aten::convolution",
            "arguments": [
                tensor([8, 3, 224, 224], [150528, 50176, 224, 1]),
                tensor([1152, 3, 14, 14], [588, 196, 14, 1]),
                tensor([1152], [1]), [14, 14], [0, 0], [1, 1],
                False, [0, 0], 1,
            ],
            "count": 1,
        },
        {
            "aten_operation": "aten::convolution",
            "arguments": [
                tensor([8, 256, 24, 24], [147456, 576, 24, 1]),
                tensor([256, 256, 4, 4], [4096, 16, 4, 1]),
                tensor([256], [1]), [4, 4], [0, 0], [1, 1],
                True, [0, 0], 1,
            ],
            "count": 1,
        },
        {
            "aten_operation": "aten::mean",
            "arguments": [tensor([8, 10, 2304], [23040, 2304, 1]), [-1], True],
            "count": 24,
        },
        {
            "aten_operation": "aten::_softmax",
            "arguments": [tensor([8, 4, 10, 10], [400, 100, 10, 1]), 1, False],
            "count": 7,
        },
        {
            "aten_operation": "aten::addmm",
            "arguments": [
                tensor([1152], [1]),
                tensor([2048, 1152], [1152, 1]),
                tensor([1152, 1152], [1, 1152]),
            ],
            "count": 1,
        },
        {
            "aten_operation": "aten::bmm",
            "arguments": [
                tensor([256, 10, 64], [640, 64, 1]),
                tensor([256, 64, 10], [640, 1, 64]),
            ],
            "count": 12,
        },
    ]}
    convolutions = convolution_configurations(trace, trace_path)
    assert {item["operation"] for item in convolutions} == {
        "conv2d", "conv_transpose2d",
    }
    transpose = next(item for item in convolutions if item["operation"] == "conv_transpose2d")
    assert transpose["input_per_unit"] == [256, 24, 24]
    assert transpose["output_padding"] == [0, 0]
    reductions = reduction_configurations(trace, trace_path)
    assert {item["operation"] for item in reductions} == {"mean", "softmax"}
    assert next(item for item in reductions if item["operation"] == "softmax")["dim"] == 1
    matrices = matrix_configurations(trace)
    assert [item["operation"] for item in matrices] == ["addmm", "bmm"]
    assert matrices[1]["m_per_request"] == 32
    assert matrices[1]["layout"] == "right_transposed"


def test_spatialvla_builder_rejects_overlapping_transposed_convolution(tmp_path) -> None:
    trace = {"operations": [{
        "aten_operation": "aten::convolution",
        "arguments": [
            tensor([8, 256, 24, 24], [147456, 576, 24, 1]),
            tensor([256, 256, 4, 4], [4096, 16, 4, 1]),
            None, [2, 2], [0, 0], [1, 1], True, [0, 0], 1,
        ],
        "count": 1,
    }]}
    with pytest.raises(ValueError, match="non-overlap"):
        convolution_configurations(trace, tmp_path / "trace.json")

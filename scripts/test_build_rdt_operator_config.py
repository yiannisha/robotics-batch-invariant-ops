from collections import Counter

from build_rdt_operator_config import matrix_configurations, reduction_configurations
from run_model_causal_ablation import SelectiveOperators


def traced(shape, stride):
    return {"shape": shape, "stride": stride, "dtype": "torch.bfloat16", "device": "cuda"}


def test_rdt_matrix_and_reduction_configurations_use_request_axis(tmp_path):
    trace = {
        "operations": [
            {
                "aten_operation": "aten::addmm",
                "arguments": [traced([32], [1]), traced([80, 16], [16, 1]), traced([16, 32], [1, 16])],
                "count": 2,
            },
            {
                "aten_operation": "aten::mean",
                "arguments": [traced([8, 4, 16], [64, 16, 1]), [2], True],
                "count": 3,
            },
            {
                "aten_operation": "aten::_softmax",
                "arguments": [traced([8, 4, 16], [64, 16, 1]), -1, False],
                "count": 4,
            },
        ]
    }
    matrices = matrix_configurations(trace)
    assert [(row["operation"], row["m_per_request"], row["layout"]) for row in matrices] == [
        ("addmm", 10, "right_transposed")
    ]
    reductions = reduction_configurations(trace, tmp_path / "trace.json")
    assert {(row["operation"], row["m_per_request"], row["dim"]) for row in reductions} == {
        ("mean", 1, 2), ("softmax", 1, -1)
    }


def test_rdt_convolution_may_be_explicitly_inapplicable():
    operators = SelectiveOperators(("convolution",))
    operators.counts = Counter()
    operators.assert_coverage(
        "operators_plus_patch_projection",
        inapplicable_operations=frozenset({"convolution"}),
    )

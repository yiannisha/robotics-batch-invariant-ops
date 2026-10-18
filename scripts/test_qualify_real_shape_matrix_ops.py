import torch

from qualify_real_shape_matrix_ops import explicit_per_matrix_reference


def test_conv1d_reference_preserves_request_chunks() -> None:
    source = torch.randn(6, 2, 9)
    weight = torch.randn(4, 2, 3)
    bias = torch.randn(4)
    config = {
        "m_per_request": 3, "stride": [1], "padding": [1],
        "dilation": [1], "groups": 1,
    }
    expected = torch.nn.functional.conv1d(source, weight, bias, 1, 1)
    actual = explicit_per_matrix_reference("conv1d", source, weight, bias, config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_conv3d_reference_preserves_request_chunks() -> None:
    source = torch.randn(4, 2, 3, 5, 5)
    weight = torch.randn(3, 2, 2, 3, 3)
    bias = torch.randn(3)
    config = {
        "m_per_request": 2, "stride": [1, 1, 1], "padding": [0, 1, 1],
        "dilation": [1, 1, 1], "groups": 1,
    }
    expected = torch.nn.functional.conv3d(source, weight, bias, (1, 1, 1), (0, 1, 1))
    actual = explicit_per_matrix_reference("conv3d", source, weight, bias, config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_conv_transpose3d_reference_preserves_request_chunks() -> None:
    source = torch.randn(4, 2, 3, 5, 5)
    weight = torch.randn(2, 3, 2, 1, 1)
    bias = torch.randn(3)
    config = {
        "m_per_request": 2, "stride": [2, 1, 1], "padding": [0, 0, 0],
        "output_padding": [0, 0, 0], "dilation": [1, 1, 1], "groups": 1,
    }
    expected = torch.nn.functional.conv_transpose3d(
        source, weight, bias, (2, 1, 1), (0, 0, 0), (0, 0, 0), 1, (1, 1, 1),
    )
    actual = explicit_per_matrix_reference(
        "conv_transpose3d", source, weight, bias, config,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_conv_transpose2d_reference_preserves_request_chunks() -> None:
    source = torch.randn(4, 2, 5, 5)
    weight = torch.randn(2, 3, 2, 2)
    bias = torch.randn(3)
    config = {
        "m_per_request": 2, "stride": [2, 2], "padding": [0, 0],
        "output_padding": [0, 0], "dilation": [1, 1], "groups": 1,
    }
    expected = torch.nn.functional.conv_transpose2d(
        source, weight, bias, (2, 2), (0, 0), (0, 0), 1, (1, 1),
    )
    actual = explicit_per_matrix_reference(
        "conv_transpose2d", source, weight, bias, config,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mean_reference_preserves_request_chunks() -> None:
    source = torch.randn(6, 5, 7)
    config = {"m_per_request": 2, "dim": -1, "keepdim": True}
    expected = torch.mean(source, dim=-1, keepdim=True)
    actual = explicit_per_matrix_reference("mean", source, None, None, config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_softmax_reference_preserves_request_chunks() -> None:
    source = torch.randn(6, 5, 7)
    config = {"m_per_request": 2, "dim": -1}
    expected = torch.softmax(source, dim=-1)
    actual = explicit_per_matrix_reference("softmax", source, None, None, config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

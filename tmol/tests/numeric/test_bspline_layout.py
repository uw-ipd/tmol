"""Interpolation depends on logical coordinates, not a tensor's storage layout."""

import math

import pytest
import torch

from tmol.numeric import BSplineInterpolation
from tmol.numeric.bspline_compiled import _compiled


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("layout", ["padded", "transposed"])
def test_strided_coefficients_preserve_values_and_derivatives(ndim, layout):
    shape = tuple(range(5, 5 + ndim))
    source = torch.sin(torch.arange(math.prod(shape), dtype=torch.float32)).reshape(
        shape
    )
    fitted = BSplineInterpolation.from_coordinates(source)
    if layout == "padded":
        storage = torch.full(tuple(n + 3 for n in shape), 137.0)
        view = storage[tuple(slice(1, n + 1) for n in shape)]
    else:
        storage = torch.empty(tuple(reversed(shape)))
        view = storage.permute(*reversed(range(ndim)))
    view.copy_(fitted.coeffs)
    assert not view.is_contiguous()
    points = torch.tensor([[0.125, -0.75, 3.25, 5.9], [4.875, 6.25, 7.5, -1.2]])[
        :, :ndim
    ].contiguous()
    evaluate = getattr(_compiled, f"interpolate{ndim}")
    expected = evaluate(fitted.coeffs, points)
    actual = evaluate(view, points)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_fitting_transposed_coordinates_preserves_input_and_grid_values(ndim):
    shape = tuple(range(5, 5 + ndim))
    source = torch.cos(torch.arange(math.prod(shape), dtype=torch.float32)).reshape(
        shape
    )
    source = source.permute(*reversed(range(ndim)))
    before = source.clone()
    fitted = BSplineInterpolation.from_coordinates(source)
    expected = BSplineInterpolation.from_coordinates(source.contiguous())
    torch.testing.assert_close(fitted.coeffs, expected.coeffs, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    axes = [torch.arange(n) for n in source.shape]
    points = (
        torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        .reshape(-1, ndim)
        .float()
    )
    torch.testing.assert_close(
        fitted.interpolate(points).reshape(source.shape), source, rtol=1e-5, atol=2e-6
    )

"""Interpolation depends on logical coordinates, not a tensor's storage layout."""

import math

import pytest
import torch

from tmol.numeric import BSplineInterpolation
from tmol.numeric.bspline_compiled import _compiled
from tmol.tests._torch import requires_cuda


def _strided_copy(source, layout):
    shape = source.shape
    if layout == "padded":
        storage = torch.full(
            tuple(n + 3 for n in shape),
            137.0,
            dtype=source.dtype,
            device=source.device,
        )
        view = storage[tuple(slice(1, n + 1) for n in shape)]
    else:
        storage = torch.empty(
            tuple(reversed(shape)), dtype=source.dtype, device=source.device
        )
        view = storage.permute(*reversed(range(source.ndim)))
    view.copy_(source)
    assert not view.is_contiguous()
    return view


@pytest.fixture(scope="module")
def _cuda_bspline():
    from tmol._load_ext import load_module

    return load_module(
        __name__,
        __file__,
        ["bspline.cuda.cu"],
        "tmol.tests.numeric._bspline_cuda",
    )


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("layout", ["padded", "transposed"])
def test_strided_coefficients_preserve_values_and_derivatives(ndim, layout):
    shape = tuple(range(5, 5 + ndim))
    source = torch.sin(torch.arange(math.prod(shape), dtype=torch.float32)).reshape(
        shape
    )
    fitted = BSplineInterpolation.from_coordinates(source)
    view = _strided_copy(fitted.coeffs, layout)
    points = torch.tensor([[0.125, -0.75, 3.25, 5.9], [4.875, 6.25, 7.5, -1.2]])[
        :, :ndim
    ].contiguous()
    evaluate = getattr(_compiled, f"interpolate{ndim}")
    expected = evaluate(fitted.coeffs, points)
    actual = evaluate(view, points)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("layout", ["padded", "transposed"])
def test_cuda_strided_coefficients_match_cpu_reference(_cuda_bspline, ndim, layout):
    shape = tuple(range(5, 5 + ndim))
    source = torch.sin(torch.arange(math.prod(shape), dtype=torch.float32)).reshape(
        shape
    )
    fitted = BSplineInterpolation.from_coordinates(source)
    points = torch.tensor(
        [[0.125, -0.75, 3.25, 5.9], [4.875, 6.25, 7.5, -1.2]],
        dtype=torch.float32,
    )[:, :ndim].contiguous()

    expected = getattr(_compiled, f"interpolate{ndim}")(fitted.coeffs, points)
    coeffs = _strided_copy(fitted.coeffs.cuda(), layout)
    actual = getattr(_cuda_bspline, f"interpolate{ndim}")(coeffs, points.cuda())

    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0, atol=6e-8)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=0, atol=5e-7)


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
    batch = BSplineInterpolation._coefficients_from_coordinate_tables(
        [source, source[:2]]
    )
    torch.testing.assert_close(batch[0], expected.coeffs, rtol=0, atol=0)
    torch.testing.assert_close(
        batch[1],
        BSplineInterpolation.from_coordinates(source[:2]).coeffs,
        rtol=0,
        atol=0,
    )
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


@pytest.mark.parametrize(
    "shape", [(1, 2), (2, 3), (5, 8), (12, 18), (1, 2, 3), (3, 4, 5, 6)]
)
def test_periodic_coefficients_match_independent_linear_solve(shape):
    # Cardinal cubic interpolation at grid points is convolution by (1,4,1)/6.
    # Solve that periodic linear system independently along each tensor axis.
    source = torch.cos(torch.arange(math.prod(shape), dtype=torch.float32)).reshape(
        shape
    )
    expected = source.double()
    for axis, width in enumerate(shape):
        identity = torch.eye(width, dtype=torch.float64)
        system = (4 * identity + identity.roll(1, 0) + identity.roll(-1, 0)) / 6
        rows = expected.movedim(axis, 0)
        expected = (
            torch.linalg.solve(system, rows.reshape(width, -1))
            .reshape(rows.shape)
            .movedim(0, axis)
        )
    actual = BSplineInterpolation.from_coordinates(source)
    torch.testing.assert_close(actual.coeffs.double(), expected, atol=2e-5, rtol=2e-6)

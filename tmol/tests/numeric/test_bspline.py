import torch
import numpy
import pytest
from tmol.numeric.bspline import BSplineInterpolation


@pytest.mark.parametrize("input", [torch.tensor([2.0, 5.0]), torch.tensor([3.0, 4.0])])
def test_2d_bspline(input):
    x = torch.arange(-5, 6, dtype=torch.float).unsqueeze(1)
    y = torch.arange(-5, 6, dtype=torch.float).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z)
    zint = zspline.interpolate(torch.tensor(input, dtype=torch.float))

    zgold = z[int(input[0].item()), int(input[1].item())]
    numpy.testing.assert_allclose(zint, zgold, atol=1e-5)


@pytest.mark.parametrize(
    "input, expected",
    [(torch.tensor([3.125, 4.5]), -0.406765), (torch.tensor([3.25, 4.75]), -0.414712)],
)
def test_2d_bspline_off_grid(input, expected):
    x = torch.arange(-5, 6, dtype=torch.float).unsqueeze(1)
    y = torch.arange(-5, 6, dtype=torch.float).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z)
    zint = zspline.interpolate(torch.tensor(input, dtype=torch.float))
    numpy.testing.assert_allclose(zint, expected, atol=1e-5)


@pytest.mark.parametrize(
    "input, expected",
    [
        (torch.tensor([0.15625, 6 - 0.15625]), -0.004631),
        (torch.tensor([0.140625, 0.125]), -0.004126),
        (torch.tensor([5.140625, 0.125]), -0.019515),
        (torch.tensor([6.140625, 6.140625]), -0.000967),
    ],
)
def test_2d_bspline_off_grid_at_edges(input, expected):
    x = torch.arange(-3, 4, dtype=torch.float).unsqueeze(1)
    y = torch.arange(-3, 4, dtype=torch.float).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z)
    zint = zspline.interpolate(torch.tensor(input, dtype=torch.float))
    numpy.testing.assert_allclose(zint, expected, atol=1e-5)


@pytest.mark.parametrize("input", [torch.tensor([2.0, 5.0]), torch.tensor([3.0, 4.0])])
def test_2d_bspline_not_square(input):
    # 2d
    x = torch.arange(-5, 6, dtype=torch.float).unsqueeze(1)
    y = torch.arange(-8, 9, dtype=torch.float).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z)
    zint = zspline.interpolate(torch.tensor(input, dtype=torch.float))

    zgold = z[int(input[0].item()), int(input[1].item())]
    numpy.testing.assert_allclose(zint, zgold, atol=1e-5)


@pytest.mark.parametrize(
    "input", [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([3.0, 4.0, 1.0])]
)
def test_3d_bspline(input):
    # 3d
    x = torch.arange(-5, 6, dtype=torch.float).reshape(-1, 1, 1)
    y = torch.arange(-5, 6, dtype=torch.float).reshape(1, -1, 1)
    z = torch.arange(-5, 6, dtype=torch.float).reshape(1, 1, -1)
    w = (1 - x * x - y * y - z * z) * torch.exp(-0.5 * (x * x + y * y + z * z))
    w = w.type(torch.float)

    wspline = BSplineInterpolation.from_coordinates(w)
    wint = wspline.interpolate(torch.tensor(input, dtype=torch.float))

    wgold = w[int(input[0].item()), int(input[1].item()), int(input[2].item())]
    numpy.testing.assert_allclose(wint, wgold, atol=1e-5)


@pytest.mark.parametrize(
    "input", [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([3.0, 4.0, 1.0])]
)
def test_3d_bspline_not_square(input):
    # 3d
    x = torch.arange(-4, 5, dtype=torch.float).reshape(-1, 1, 1)
    y = torch.arange(-5, 6, dtype=torch.float).reshape(1, -1, 1)
    z = torch.arange(-6, 7, dtype=torch.float).reshape(1, 1, -1)
    w = (1 - x * x - y * y - z * z) * torch.exp(-0.5 * (x * x + y * y + z * z))
    w = w.type(torch.float)

    wspline = BSplineInterpolation.from_coordinates(w)
    wint = wspline.interpolate(torch.tensor(input, dtype=torch.float))

    wgold = w[int(input[0].item()), int(input[1].item()), int(input[2].item())]
    numpy.testing.assert_allclose(wint, wgold, atol=1e-5)


@pytest.mark.parametrize(
    "input", [torch.tensor([2.0, 3.0, 5.0, 7.0]), torch.tensor([3.0, 4.0, 1.0, 7.0])]
)
def test_4d_bspline(input):
    # 3d
    x = torch.arange(-5, 6, dtype=torch.float).reshape(-1, 1, 1, 1)
    y = torch.arange(-5, 6, dtype=torch.float).reshape(1, -1, 1, 1)
    z = torch.arange(-5, 6, dtype=torch.float).reshape(1, 1, -1, 1)
    w = torch.arange(-5, 6, dtype=torch.float).reshape(1, 1, 1, -1)
    u = (1 - x * x - y * y - z * z - w * w) * torch.exp(
        -0.5 * (x * x + y * y + z * z + w * w)
    )
    u = u.type(torch.float)

    uspline = BSplineInterpolation.from_coordinates(u)
    uint = uspline.interpolate(torch.tensor(input, dtype=torch.float))

    ugold = u[
        int(input[0].item()),
        int(input[1].item()),
        int(input[2].item()),
        int(input[3].item()),
    ]
    numpy.testing.assert_allclose(uint, ugold, atol=1e-5)

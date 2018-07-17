import torch
import numpy
import pytest
from tmol.numeric.bspline import BSplineInterpolation


#@pytest.fixture(params=[3])
@pytest.fixture(params=[2, 3, 4, 5])
def bspline_degree(request):
    return request.param


def test_2d_bspline(bspline_degree):
    # 2d
    x = torch.arange(-5, 6).unsqueeze(1)
    y = torch.arange(-5, 6).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(torch.Tensor([[2, 5], [3, 4]]))

    zgold = torch.Tensor([z[2, 5], z[3, 4]])
    numpy.testing.assert_allclose(zint.numpy(), zgold.numpy(), atol=1e-5)


def test_2d_bspline_not_square(bspline_degree):
    # 2d
    x = torch.arange(-5, 6).unsqueeze(1)
    y = torch.arange(-8, 9).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(torch.Tensor([[2, 5], [3, 4]]))

    zgold = torch.Tensor([z[2, 5], z[3, 4]])
    numpy.testing.assert_allclose(zint.numpy(), zgold.numpy(), atol=1e-5)


def test_barely_3d_bspline(bspline_degree):
    # 2d
    x = torch.arange(-5, 6).unsqueeze(1)
    y = torch.arange(-5, 6).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float).reshape(1, 11, 11)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(torch.Tensor([[0, 2, 5], [0, 3, 4]]))

    zgold = torch.Tensor([z[0, 2, 5], z[0, 3, 4]])
    numpy.testing.assert_allclose(zint.numpy(), zgold.numpy(), atol=1e-5)


def test_2d_bspline_everywhere(bspline_degree):
    # 2d
    x = torch.arange(-5, 6).unsqueeze(1)
    y = torch.arange(-5, 6).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)

    phi_vals = torch.arange(11).reshape(-1, 1).repeat(1, 11).reshape(-1, 1)
    psi_vals = torch.arange(11).repeat(1, 11).reshape(-1, 1)
    xs = torch.cat((phi_vals, psi_vals), dim=1)

    xlong = xs.type(torch.long)
    inds = xlong[:, 0] * 11 + xlong[:, 1]

    orig_vals = z.reshape(-1)[inds]
    zint = zspline.interpolate(xs)
    numpy.testing.assert_allclose(zint.numpy(), orig_vals.numpy(), atol=1e-4)


def test_2d_bspline_x1(bspline_degree):
    # 2d
    x = torch.arange(-5, 6).unsqueeze(1)
    y = torch.arange(-5, 6).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(torch.Tensor([[2, 5]]))

    zgold = torch.Tensor([z[2, 5]])
    numpy.testing.assert_allclose(zint.numpy(), zgold.numpy(), atol=1e-5)


def test_5x2d_bspline(bspline_degree):
    # 2d
    zs = torch.full((5, 11, 11), 0, dtype=torch.float)
    for i in range(5):
        x = torch.arange(-5, 6).unsqueeze(1)
        y = torch.arange(-5, 6).unsqueeze(0)
        z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y)) + i
        z = z.type(torch.float)
        zs[i, :, :] = z
    zspline = BSplineInterpolation.from_coordinates(zs, bspline_degree, 1)
    zint = zspline.interpolate(
        torch.Tensor([[2, 5], [3, 4]]), torch.LongTensor([[3], [0]])
    )

    zgold = torch.Tensor([zs[3, 2, 5], zs[0, 3, 4]])
    numpy.testing.assert_allclose(zint.numpy(), zgold.numpy(), atol=1e-5)


def test_5x3x2d_bspline(bspline_degree):
    # 2d
    zs = torch.full((5, 3, 11, 11), 0, dtype=torch.float)
    for i in range(5):
        for j in range(3):
            x = torch.arange(-5, 6).unsqueeze(1)
            y = torch.arange(-5, 6).unsqueeze(0)
            z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y)
                                                ) + 0.1 * i * 3 + 0.1 * j
            z = z.type(torch.float)
            zs[i, j, :, :] = z
    zspline = BSplineInterpolation.from_coordinates(zs, bspline_degree, 2)

    zint = zspline.interpolate(
        torch.Tensor([[2, 5], [3, 4]]), torch.LongTensor([[3, 1], [0, 2]])
    )

    zgold = torch.Tensor([zs[3, 1, 2, 5], zs[0, 2, 3, 4]])
    numpy.testing.assert_allclose(zint.numpy(), zgold.numpy(), atol=1e-5)


def test_3d_bspline(bspline_degree):
    # 3d
    x = torch.arange(-5, 6).reshape(11, 1, 1)
    y = torch.arange(-5, 6).reshape(1, 11, 1)
    z = torch.arange(-5, 6).reshape(1, 1, 11)
    w = (1 - x * x - y * y - z * z) * torch.exp(-0.5 * (x * x + y * y + z * z))
    w = w.type(torch.float)

    wspline = BSplineInterpolation.from_coordinates(w, bspline_degree)
    wint = wspline.interpolate(torch.Tensor([[2, 3, 5], [3, 4, 1]]))
    wgold = torch.Tensor([w[2, 3, 5], w[3, 4, 1]])
    numpy.testing.assert_allclose(wint.numpy(), wgold.numpy(), atol=1e-5)


def test_3d_bspline_not_square(bspline_degree):
    # 3d
    x = torch.arange(-5, 6).reshape(11, 1, 1)
    y = torch.arange(-8, 9).reshape(1, 17, 1)
    z = torch.arange(-5, 6).reshape(1, 1, 11)
    w = (1 - x * x - y * y - z * z) * torch.exp(-0.5 * (x * x + y * y + z * z))
    w = w.type(torch.float)

    wspline = BSplineInterpolation.from_coordinates(w, bspline_degree)
    wint = wspline.interpolate(torch.Tensor([[2, 3, 5], [3, 4, 1]]))
    wgold = torch.Tensor([w[2, 3, 5], w[3, 4, 1]])
    numpy.testing.assert_allclose(wint.numpy(), wgold.numpy(), atol=1e-5)


def test_4d_bspline(bspline_degree):
    # 4d
    x = torch.arange(-5, 6).reshape(11, 1, 1, 1)
    y = torch.arange(-5, 6).reshape(1, 11, 1, 1)
    z = torch.arange(-5, 6).reshape(1, 1, 11, 1)
    w = torch.arange(-5, 6).reshape(1, 1, 1, 11)
    u = (1 - x * x - y * y - z * z - w * w
         ) * torch.exp(-0.5 * (x * x + y * y + z * z + w * w))
    u = u.type(torch.float)

    uspline = BSplineInterpolation.from_coordinates(u, bspline_degree)
    uint = uspline.interpolate(torch.Tensor([[2, 3, 5, 7], [3, 4, 1, 7]]))

    ugold = torch.Tensor((u[2, 3, 5, 7], u[3, 4, 1, 7]))
    numpy.testing.assert_allclose(uint.numpy(), ugold.numpy(), atol=1e-5)

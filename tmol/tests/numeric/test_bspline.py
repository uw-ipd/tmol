import torch
import numpy
import pytest
from tmol.numeric.bspline import BSplineInterpolation


@pytest.fixture(params=[2, 3, 4, 5])
def bspline_degree(request):
    return request.param


def test_2d_bspline(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
    y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(
        torch.tensor([[2, 5], [3, 4]], dtype=torch.float, device=torch_device)
    )

    zgold = torch.tensor([z[2, 5], z[3, 4]])
    numpy.testing.assert_allclose(zint.cpu().numpy(), zgold.numpy(), atol=1e-5)


def test_2d_bspline_off_grid(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
    y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    offgrid = torch.tensor(
        [[3.125, 4.5], [3.25, 4.75]], dtype=torch.float, device=torch_device
    )
    x_og = -5 + offgrid[:, 0]
    y_og = -5 + offgrid[:, 1]
    z_offgrid = (1 - x_og * x_og - y_og * y_og) * torch.exp(
        -0.5 * (x_og * x_og + y_og * y_og)
    )

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(offgrid)

    # empirically observed increase in quality-of-fit for this landscape for the chosen
    # spline degrees for the particular choices of off-grid points. Totally detached
    # from any numerical analysis or theory. Unlikely to apply to other cases.
    # Duplicate these tolerances at your own risk!
    atol = 3 * pow(10, -0.5 * bspline_degree)

    numpy.testing.assert_allclose(
        zint.cpu().numpy(), z_offgrid.cpu().numpy(), atol=atol
    )


def test_2d_bspline_off_grid_at_edges(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
    y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    offgrid = torch.tensor(
        [[0.15625, 10 - 0.15625], [0.140625, 0.125], [5.140625, 0.125]],
        dtype=torch.float,
        device=torch_device,
    )
    x_og = -5 + offgrid[:, 0]
    y_og = -5 + offgrid[:, 1]
    z_offgrid = (1 - x_og * x_og - y_og * y_og) * torch.exp(
        -0.5 * (x_og * x_og + y_og * y_og)
    )

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(offgrid)

    # empirically observed "increase" in quality-of-fit for this landscape
    # for the chosen spline degrees for the particular choices of off-grid
    # points. Totally detached from any numerical analysis or theory.
    # Unlikely to apply to other cases. Duplicate these tolerances at your
    # own risk!
    # The quality of fit actually decreases as the spline degree increases
    # as the chosen landscape is not inherrently periodic. As the degree
    # increases and the spline reaches for more and more data, it has to
    # distort the potential toward the edges more and more. Eww.
    atol = 4 * pow(10, -5 + 0.5 * bspline_degree)

    numpy.testing.assert_allclose(
        zint.cpu().numpy(), z_offgrid.cpu().numpy(), atol=atol
    )


def test_2d_bspline_off_grid_periodic(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-20, 20, device=torch_device).unsqueeze(1)
    y = torch.arange(-20, 20, device=torch_device).unsqueeze(0)
    z = torch.sin(numpy.pi / 10 * x) + torch.cos(numpy.pi / 10 * y)
    z = z.type(torch.float)

    offgrid = torch.tensor(
        [[11.15625, 22.15625], [15.140625, 8.125], [23.140625, 17.125]],
        dtype=torch.float,
        device=torch_device,
    )
    x_og = -20 + offgrid[:, 0]
    y_og = -20 + offgrid[:, 1]
    z_offgrid = torch.sin(numpy.pi / 10 * x_og) + torch.cos(numpy.pi / 10 * y_og)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(offgrid)

    # empirically observed "increase" in quality-of-fit for this landscape
    # for the chosen spline degrees for the particular choices of off-grid
    # points. Totally detached from any numerical analysis or theory.
    # Unlikely to apply to other cases. Duplicate these tolerances at your
    # own risk!
    atol = 5 * pow(10, -2 + -1 * bspline_degree)

    numpy.testing.assert_allclose(
        zint.cpu().numpy(), z_offgrid.cpu().numpy(), atol=atol
    )


def test_2d_bspline_off_grid_at_edges_periodic(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-20, 20, device=torch_device).unsqueeze(1)
    y = torch.arange(-20, 20, device=torch_device).unsqueeze(0)
    z = torch.sin(numpy.pi / 10 * x) + torch.cos(numpy.pi / 10 * y)
    z = z.type(torch.float)

    offgrid = torch.tensor(
        [[0.15625, 40 - 0.15625], [0.140625, 0.125], [5.140625, 0.125]],
        dtype=torch.float,
        device=torch_device,
    )
    x_og = -20 + offgrid[:, 0]
    y_og = -20 + offgrid[:, 1]
    z_offgrid = torch.sin(numpy.pi / 10 * x_og) + torch.cos(numpy.pi / 10 * y_og)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(offgrid)

    # empirically observed increase in quality-of-fit for this landscape
    # for the chosen spline degrees for the particular choices of off-grid
    # points. Totally detached from any numerical analysis or theory.
    # Unlikely to apply to other cases. Duplicate these tolerances at your
    # own risk!
    atol = 5 * pow(10, -2 + -1 * bspline_degree)

    numpy.testing.assert_allclose(
        zint.cpu().numpy(), z_offgrid.cpu().numpy(), atol=atol
    )


def test_request_unsupported_bspline_degree(torch_device):
    x = torch.arange(-20, 20, device=torch_device).unsqueeze(1)
    y = torch.arange(-20, 20, device=torch_device).unsqueeze(0)
    z = torch.sin(numpy.pi / 10 * x) + torch.cos(numpy.pi / 10 * y)
    z = z.type(torch.float)

    with pytest.raises(ValueError):
        BSplineInterpolation.from_coordinates(z, 6)


def test_2d_bspline_not_square(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
    y = torch.arange(-8, 9, device=torch_device).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(
        torch.tensor([[2, 5], [3, 4]], dtype=torch.float, device=torch_device)
    )

    zgold = torch.tensor([z[2, 5], z[3, 4]])
    numpy.testing.assert_allclose(zint.cpu().numpy(), zgold.numpy(), atol=1e-5)


def test_barely_3d_bspline(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
    y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float).reshape(1, 11, 11)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(
        torch.tensor([[0, 2, 5], [0, 3, 4]], dtype=torch.float, device=torch_device)
    )

    zgold = torch.tensor([z[0, 2, 5], z[0, 3, 4]])
    numpy.testing.assert_allclose(zint.cpu().numpy(), zgold.numpy(), atol=1e-5)


def test_2d_bspline_everywhere(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
    y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)

    phi_vals = (
        torch.arange(11, dtype=torch.float, device=torch_device)
        .reshape(-1, 1)
        .repeat(1, 11)
        .reshape(-1, 1)
    )
    psi_vals = (
        torch.arange(11, dtype=torch.float, device=torch_device)
        .repeat(1, 11)
        .reshape(-1, 1)
    )
    xs = torch.cat((phi_vals, psi_vals), dim=1)

    xlong = xs.type(torch.long)
    inds = xlong[:, 0] * 11 + xlong[:, 1]

    orig_vals = z.reshape(-1)[inds]
    zint = zspline.interpolate(xs)
    numpy.testing.assert_allclose(
        zint.cpu().numpy(), orig_vals.cpu().numpy(), atol=1e-4
    )


def test_2d_bspline_x1(bspline_degree, torch_device):
    # 2d
    x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
    y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zspline = BSplineInterpolation.from_coordinates(z, bspline_degree)
    zint = zspline.interpolate(
        torch.tensor([[2, 5]], dtype=torch.float, device=torch_device)
    )

    zgold = torch.tensor([z[2, 5]])
    numpy.testing.assert_allclose(zint.cpu().numpy(), zgold.numpy(), atol=1e-5)


def test_5x2d_bspline(bspline_degree, torch_device):
    # 2d
    zs = torch.full((5, 11, 11), 0, dtype=torch.float, device=torch_device)
    for i in range(5):
        x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
        y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
        z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y)) + i
        z = z.type(torch.float)
        zs[i, :, :] = z
    zspline = BSplineInterpolation.from_coordinates(zs, bspline_degree, 1)
    zint = zspline.interpolate(
        torch.tensor([[2, 5], [3, 4]], dtype=torch.float, device=torch_device),
        torch.tensor([[3], [0]], dtype=torch.long, device=torch_device),
    )

    zgold = torch.tensor([zs[3, 2, 5], zs[0, 3, 4]])
    numpy.testing.assert_allclose(zint.cpu().numpy(), zgold.numpy(), atol=1e-5)


def test_5x3x2d_bspline(bspline_degree, torch_device):
    # 2d
    zs = torch.full((5, 3, 11, 11), 0, dtype=torch.float, device=torch_device)
    for i in range(5):
        for j in range(3):
            x = torch.arange(-5, 6, device=torch_device).unsqueeze(1)
            y = torch.arange(-5, 6, device=torch_device).unsqueeze(0)
            z = (
                (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
                + 0.1 * i * 3
                + 0.1 * j
            )
            z = z.type(torch.float)
            zs[i, j, :, :] = z
    zspline = BSplineInterpolation.from_coordinates(zs, bspline_degree, 2)

    zint = zspline.interpolate(
        torch.tensor([[2, 5], [3, 4]], dtype=torch.float, device=torch_device),
        torch.tensor([[3, 1], [0, 2]], dtype=torch.long, device=torch_device),
    )

    zgold = torch.tensor([zs[3, 1, 2, 5], zs[0, 2, 3, 4]])
    numpy.testing.assert_allclose(zint.cpu().numpy(), zgold.numpy(), atol=1e-5)


def test_3d_bspline(bspline_degree, torch_device):
    # 3d
    x = torch.arange(-5, 6, device=torch_device).reshape(11, 1, 1)
    y = torch.arange(-5, 6, device=torch_device).reshape(1, 11, 1)
    z = torch.arange(-5, 6, device=torch_device).reshape(1, 1, 11)
    w = (1 - x * x - y * y - z * z) * torch.exp(-0.5 * (x * x + y * y + z * z))
    w = w.type(torch.float)

    wspline = BSplineInterpolation.from_coordinates(w, bspline_degree)
    wint = wspline.interpolate(
        torch.tensor([[2, 3, 5], [3, 4, 1]], dtype=torch.float, device=torch_device)
    )
    wgold = torch.tensor([w[2, 3, 5], w[3, 4, 1]])
    numpy.testing.assert_allclose(wint.cpu().numpy(), wgold.numpy(), atol=1e-5)


def test_3d_bspline_not_square(bspline_degree, torch_device):
    # 3d
    x = torch.arange(-5, 6, device=torch_device).reshape(11, 1, 1)
    y = torch.arange(-8, 9, device=torch_device).reshape(1, 17, 1)
    z = torch.arange(-5, 6, device=torch_device).reshape(1, 1, 11)
    w = (1 - x * x - y * y - z * z) * torch.exp(-0.5 * (x * x + y * y + z * z))
    w = w.type(torch.float)

    wspline = BSplineInterpolation.from_coordinates(w, bspline_degree)
    wint = wspline.interpolate(
        torch.tensor([[2, 3, 5], [3, 4, 1]], dtype=torch.float, device=torch_device)
    )
    wgold = torch.tensor([w[2, 3, 5], w[3, 4, 1]])
    numpy.testing.assert_allclose(wint.cpu().numpy(), wgold.numpy(), atol=1e-5)


def test_4d_bspline(bspline_degree, torch_device):
    # 4d
    x = torch.arange(-5, 6, device=torch_device).reshape(11, 1, 1, 1)
    y = torch.arange(-5, 6, device=torch_device).reshape(1, 11, 1, 1)
    z = torch.arange(-5, 6, device=torch_device).reshape(1, 1, 11, 1)
    w = torch.arange(-5, 6, device=torch_device).reshape(1, 1, 1, 11)
    u = (1 - x * x - y * y - z * z - w * w) * torch.exp(
        -0.5 * (x * x + y * y + z * z + w * w)
    )
    u = u.type(torch.float)

    uspline = BSplineInterpolation.from_coordinates(u, bspline_degree)
    uint = uspline.interpolate(
        torch.tensor(
            [[2, 3, 5, 7], [3, 4, 1, 7]], dtype=torch.float, device=torch_device
        )
    )

    ugold = torch.tensor((u[2, 3, 5, 7], u[3, 4, 1, 7]))
    numpy.testing.assert_allclose(uint.cpu().numpy(), ugold.numpy(), atol=1e-5)

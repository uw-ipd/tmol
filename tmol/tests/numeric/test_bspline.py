import torch
import numpy
from tmol.numeric.bspline import compute_coeffs, interpolate


def test_2d_bspline():
    # 2d
    x = torch.arange(-5, 6).unsqueeze(1)
    y = torch.arange(-5, 6).unsqueeze(0)
    z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y))
    z = z.type(torch.float)

    zcoeff = compute_coeffs(z, 3)
    zint = interpolate(zcoeff, 3, torch.Tensor([[2, 5], [3, 4]]))

    zgold = torch.Tensor([z[2, 5], z[3, 4]])
    numpy.testing.assert_allclose(zgold.numpy(), zint.numpy(), rtol=1e-6)


def test_5x2d_bspline():
    # 2d
    zs = torch.full((5, 11, 11), 0, dtype=torch.float)
    zscoeff = torch.full((5, 11, 11), 0, dtype=torch.float)
    for i in range(5):
        x = torch.arange(-5, 6).unsqueeze(1)
        y = torch.arange(-5, 6).unsqueeze(0)
        z = (1 - x * x - y * y) * torch.exp(-0.5 * (x * x + y * y)) + i
        z = z.type(torch.float)
        zcoeff = compute_coeffs(z, 3)
        zs[i, :, :] = z
        zscoeff[i, :, :] = zcoeff

    zint = interpolate(
        zscoeff, 3, torch.Tensor([[2, 5], [3, 4]]),
        torch.LongTensor((3, 0)).reshape(-1, 1)
    )

    zgold = torch.Tensor([zs[3, 2, 5], zs[0, 3, 4]])
    numpy.testing.assert_allclose(zgold.numpy(), zint.numpy(), rtol=1e-6)


def test_3d_bspline():
    # 3d
    x = torch.arange(-5, 6).reshape(11, 1, 1)
    y = torch.arange(-5, 6).reshape(1, 11, 1)
    z = torch.arange(-5, 6).reshape(1, 1, 11)
    w = (1 - x * x - y * y - z * z) * torch.exp(-0.5 * (x * x + y * y + z * z))
    w = w.type(torch.float)

    wcoeff = compute_coeffs(w, 3)
    wint = interpolate(wcoeff, 3, torch.Tensor([[2, 3, 5], [3, 4, 1]]))
    wgold = torch.Tensor([w[2, 3, 5], w[3, 4, 1]])
    numpy.testing.assert_allclose(wgold.numpy(), wint.numpy(), rtol=1e-6)


def test_4d_bspline():
    # 4d
    x = torch.arange(-5, 6).reshape(11, 1, 1, 1)
    y = torch.arange(-5, 6).reshape(1, 11, 1, 1)
    z = torch.arange(-5, 6).reshape(1, 1, 11, 1)
    w = torch.arange(-5, 6).reshape(1, 1, 1, 11)
    u = (1 - x * x - y * y - z * z - w * w
         ) * torch.exp(-0.5 * (x * x + y * y + z * z + w * w))
    u = u.type(torch.float)

    ucoeff = compute_coeffs(u, 3)
    uint = interpolate(ucoeff, 3, torch.Tensor([[2, 3, 5, 7], [3, 4, 1, 7]]))

    ugold = torch.Tensor((u[2, 3, 5, 7], u[3, 4, 1, 7]))
    numpy.testing.assert_allclose(ugold.numpy(), uint.numpy(), rtol=1e-6)

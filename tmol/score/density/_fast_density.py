"""Rosetta-style ``elec_dens_fast`` score grid.

The score of an atom at ``x`` is a precomputed, normalized, zero-mean correlation
of the observed map with a single atom-sized Gaussian, read at ``x`` by a cubic
B-spline::

    S      = IFFT( FFT(rho - mean) * FFT(K - mean) )              periodic convolution
    S_norm = S / ( 4 * (S_(5th largest) - S_(5th smallest)) )
    E_atom = - (a_elt / 6) * S_norm(x)

``K = k^1.5 exp(-k d^2)`` is truncated where it falls below ``density_cutoff``,
with ``k = min(pi^2 / sigma_C, 4 pi^2 / B_eff)`` and
``B_eff = 4 max(2 voxel, resolution)^2``. This is a linear correlation, not a
Pearson CC: there is no normalization by the model density.

This reimplements the algorithm of Rosetta's ``elec_dens_fast``; no Rosetta source
code is included.
"""

from __future__ import annotations

import math

import torch

from .density import ElectronDensityMap, _CRYOEM_SCATTERERS


class FastDensityScore:
    """Differentiable ``elec_dens_fast`` score grid, B-spline interpolated in position.

    Args:
        density_map: Observed map (``[z, y, x]`` values; Cartesian ``origin`` and
            ``voxel_size`` in ``[x, y, z]``).
        resolution: Map resolution in Angstrom; it sets the kernel width as
            Rosetta's ``-edensity::mapreso`` does.
        fastdens_params: Rosetta ``-edensity::fastdens_params``, a constant scale of
            the kernel.
        normalisation_cut: The score range is taken between this-th smallest and
            this-th largest grid value.
        scalefactor: Rosetta's normalization scale (the score is divided by
            ``scalefactor / 2`` times the range).
        density_cutoff: The atom kernel is truncated where it falls below this value.
    """

    def __init__(
        self,
        density_map: ElectronDensityMap,
        resolution: float,
        *,
        fastdens_params: tuple[float, float] = (0.4, 0.6),
        normalisation_cut: int = 5,
        scalefactor: float = 8.0,
        density_cutoff: float = 1e-4,
    ):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        rho = density_map.density.permute(
            2, 1, 0
        ).contiguous()  # [z, y, x] -> [x, y, z]
        dev, dt = rho.device, rho.dtype
        self.shape = tuple(int(n) for n in rho.shape)
        self.origin = density_map.origin.to(dev, dt).reshape(3)
        self.voxel = density_map.voxel_size.to(dev, dt).reshape(3)
        self.resolution = float(resolution)

        sigma_carbon = _CRYOEM_SCATTERERS[6][1]
        effective_b = 4.0 * max(2.0 * float(self.voxel.max()), self.resolution) ** 2
        self.k = min(math.pi**2 / sigma_carbon, 4.0 * math.pi**2 / effective_b)
        amp = self.k**1.5
        kernel_radius_sq = (math.log(amp) - math.log(density_cutoff)) / self.k

        # periodic minimal-image squared distance from the origin voxel
        d2 = None
        for axis, n in enumerate(self.shape):
            idx = torch.arange(n, device=dev, dtype=dt)
            off = ((idx + n // 2) % n - n // 2) * self.voxel[axis]
            view = [1, 1, 1]
            view[axis] = n
            term = (off * off).reshape(view)
            d2 = term if d2 is None else d2 + term
        kernel = torch.where(
            d2 <= kernel_radius_sq, amp * torch.exp(-self.k * d2), torch.zeros_like(d2)
        )
        kernel = kernel * (self.k ** (-fastdens_params[0]) - fastdens_params[1])

        # zero-mean both maps (the DC term of each spectrum is zero); multiplying
        # the spectra is a periodic convolution
        f_rho = torch.fft.rfftn(rho)
        f_rho[0, 0, 0] = 0
        f_ker = torch.fft.rfftn(kernel)
        f_ker[0, 0, 0] = 0
        score = torch.fft.irfftn(f_rho * f_ker, s=self.shape)

        flat = score.reshape(-1)
        lo = torch.kthvalue(flat, normalisation_cut).values
        hi = torch.kthvalue(flat, flat.numel() - normalisation_cut + 1).values
        sigma = 0.5 * scalefactor * (hi - lo)
        if not torch.isfinite(sigma) or float(sigma) <= 0:
            raise ValueError("density score grid is degenerate (map is constant?)")
        self.score = score / sigma

        # interpolating cubic B-spline: coefficients c with
        # (c[i-1] + 4 c[i] + c[i+1]) / 6 = score[i] (periodic)
        response = None
        for axis, n in enumerate(self.shape):
            freq = (
                torch.fft.rfftfreq(n, dtype=dt, device=dev)
                if axis == 2
                else torch.fft.fftfreq(n, dtype=dt, device=dev)
            )
            resp = (4.0 + 2.0 * torch.cos(2.0 * math.pi * freq)) / 6.0
            view = [1, 1, 1]
            view[axis] = resp.numel()
            resp = resp.reshape(view)
            response = resp if response is None else response * resp
        self.coeffs = torch.fft.irfftn(
            torch.fft.rfftn(self.score) / response, s=self.shape
        ).contiguous()
        self._dims = torch.tensor(self.shape, device=dev)

    @staticmethod
    def amplitude(atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Rosetta's per-element amplitude ``trunc(a) / 6``; unknown ones are carbon."""
        out = torch.full(
            atomic_numbers.shape,
            float(math.trunc(_CRYOEM_SCATTERERS[6][0])),
            device=atomic_numbers.device,
        )
        for z, (a, _) in _CRYOEM_SCATTERERS.items():
            out = torch.where(
                atomic_numbers == z, torch.full_like(out, float(math.trunc(a))), out
            )
        return out / 6.0

    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        """Normalized score at points ``xyz [..., 3]`` (periodic), differentiable."""
        u = (xyz - self.origin.to(xyz.dtype)) / self.voxel.to(xyz.dtype)
        n = self._dims.to(u.device)
        u = torch.remainder(u, n.to(u.dtype))
        i0 = torch.floor(u).detach()
        t = u - i0
        i0 = i0.long()
        t2, t3 = t * t, t * t * t
        w = torch.stack(
            [
                (1 - t) ** 3 / 6.0,
                (3 * t3 - 6 * t2 + 4) / 6.0,
                (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0,
                t3 / 6.0,
            ],
            dim=-1,
        )  # [..., 3 (axis), 4 (offset)]
        offs = torch.arange(-1, 3, device=u.device)
        idx = torch.remainder(
            i0.unsqueeze(-1) + offs, n.view(*([1] * (i0.dim() - 1)), 3, 1)
        )
        ix, iy, iz = (
            idx[..., 0, :, None, None],
            idx[..., 1, None, :, None],
            idx[..., 2, None, None, :],
        )
        c = self.coeffs[ix, iy, iz].to(w.dtype)
        return torch.einsum(
            "...abc,...a,...b,...c->...", c, w[..., 0, :], w[..., 1, :], w[..., 2, :]
        )

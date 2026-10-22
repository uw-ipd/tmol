"""Rosetta-style elec_dens_fast score grid.

The score of an atom at x is a precomputed, normalized, zero-mean correlation
of the observed map with a single atom-sized Gaussian, read at x by a cubic
B-spline whose coefficients are the grid values themselves (a smoothing, not an
interpolating, spline; this matches what Rosetta computes):

    S      = IFFT( FFT(rho - mean) * FFT(K - mean) )
    S_norm = S / ( 4 * (S_(5th largest) - S_(5th smallest)) )
    E_atom = - (trunc(a_elt) / 6) * S_norm(x)

K = k^1.5 exp(-k d^2) is truncated where it falls below density_cutoff, with
k = min(pi^2 / sigma_C, 4 pi^2 / B_eff) and B_eff = 4 max(2 voxel, resolution)^2.
This is a linear correlation, not a Pearson CC.

This reimplements the algorithm of Rosetta's elec_dens_fast; no Rosetta source
code is included.
"""

from __future__ import annotations

import math

import torch

from .density import ElectronDensityMap
from .potentials import _compiled as compiled

# cubic B-spline support beyond the containing voxel
_SPLINE_MARGIN = 2


class FastDensityScore:
    """Differentiable elec_dens_fast score grid, read by a cubic B-spline.

    Args:
        density_map: Observed map; its resolution sets the kernel width.
        carbon_sigma: Intrinsic width of the carbon scatterer; caps the kernel
            sharpness at high resolution.
        periodic: Treat the map as one period of an infinite lattice. If False,
            the map is zero-padded by the kernel radius and atoms off the padded
            grid score zero.
        normalization_cut: The score range is taken between the this-th
            smallest and this-th largest grid value.
        scalefactor: The score is divided by scalefactor / 2 times the range.
        density_cutoff: The atom kernel is truncated where it falls below this.
    """

    def __init__(
        self,
        density_map: ElectronDensityMap,
        carbon_sigma: float,
        *,
        periodic: bool = False,
        normalization_cut: int = 5,
        scalefactor: float = 8.0,
        density_cutoff: float = 1e-4,
    ):
        resolution = density_map.resolution
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        rho = density_map.density.permute(2, 1, 0)  # [z, y, x] -> [x, y, z]
        dev, dt = rho.device, rho.dtype
        basis = density_map.voxel_basis.to(dev, dt).reshape(3, 3)
        self.origin = density_map.origin.to(dev, dt).reshape(3)
        self.inv_basis = torch.linalg.inv(basis)
        self.periodic = periodic

        max_spacing = float(basis.norm(dim=0).max())
        effective_b = 4.0 * max(2.0 * max_spacing, float(resolution)) ** 2
        k = min(math.pi**2 / carbon_sigma, 4.0 * math.pi**2 / effective_b)
        amp = k**1.5
        kernel_radius_sq = (math.log(amp) - math.log(density_cutoff)) / k

        if periodic:
            self.pad = (0, 0, 0)
            rho = rho - rho.mean()
        else:
            # grid steps covering the kernel radius along each axis: the
            # distance between adjacent lattice planes is 1 / |row of inv_basis|
            reach = math.sqrt(kernel_radius_sq) * self.inv_basis.norm(dim=1)
            self.pad = tuple(int(math.ceil(r)) + _SPLINE_MARGIN + 1 for r in reach)
            rho = torch.nn.functional.pad(
                rho - rho.mean(),
                (self.pad[2],) * 2 + (self.pad[1],) * 2 + (self.pad[0],) * 2,
            )
        shape = tuple(int(n) for n in rho.shape)

        # squared Cartesian length of the minimal-image grid offset of each voxel
        metric = basis.T @ basis
        offsets = []
        for axis, n in enumerate(shape):
            idx = torch.arange(n, device=dev, dtype=dt)
            view = [1, 1, 1]
            view[axis] = n
            offsets.append(((idx + n // 2) % n - n // 2).reshape(view))
        d2 = torch.zeros(shape, device=dev, dtype=dt)
        for i in range(3):
            for j in range(i, 3):
                d2 += (1.0 if i == j else 2.0) * metric[i, j] * offsets[i] * offsets[j]
        kernel = torch.where(d2 <= kernel_radius_sq, amp * torch.exp(-k * d2), 0.0)
        del d2, offsets

        # zero-mean both maps (zero DC term); multiplying spectra is a convolution
        spectrum = torch.fft.rfftn(rho)
        del rho
        f_ker = torch.fft.rfftn(kernel)
        del kernel
        spectrum *= f_ker
        del f_ker
        spectrum[0, 0, 0] = 0
        score = torch.fft.irfftn(spectrum, s=shape)
        del spectrum

        flat = score.reshape(-1)
        lo = torch.topk(flat, normalization_cut, largest=False).values[-1]
        hi = torch.topk(flat, normalization_cut).values[-1]
        sigma = 0.5 * scalefactor * (hi - lo)
        if not torch.isfinite(sigma) or float(sigma) <= 0:
            raise ValueError("density score grid is degenerate (map is constant?)")
        score /= sigma
        self.coeffs = score.contiguous()
        self.shape = shape
        self._table_cache = {}

    def _tables(self, dtype: torch.dtype, device: torch.device):
        """Grid, origin, inverse basis and padding in the given dtype and device."""
        key = (dtype, device)
        if key not in self._table_cache:
            self._table_cache[key] = tuple(
                t.to(dtype=dtype, device=device).contiguous()
                for t in (
                    self.coeffs,
                    self.origin,
                    self.inv_basis,
                    torch.tensor(self.pad, dtype=torch.float64),
                )
            )
        return self._table_cache[key]

    def score(
        self,
        coords: torch.Tensor,
        atom_index: torch.Tensor,
        atom_group: torch.Tensor,
        atom_weight: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        """Sum of -atom_weight * S(coords[atom_index]) into n_groups slots.

        coords is [n, 3]; atom_index entries must be distinct. Differentiable in
        coords.
        """
        coeffs, origin, inv_basis, pad = self._tables(coords.dtype, coords.device)
        return compiled.density_score(
            coords.contiguous(),
            coeffs,
            origin,
            inv_basis,
            pad,
            self.periodic,
            atom_index,
            atom_group,
            atom_weight.to(coords.dtype),
            n_groups,
        )

    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        """Normalized score at points xyz [..., 3], differentiable."""
        points = xyz.reshape(-1, 3)
        index = torch.arange(points.shape[0], device=xyz.device)
        weight = torch.full((points.shape[0],), -1.0, device=xyz.device)
        value = self.score(points, index, index, weight, points.shape[0])
        return value.reshape(xyz.shape[:-1])

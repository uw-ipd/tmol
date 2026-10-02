"""Rosetta-style differentiable cryo-EM density scoring.

This module reimplements the algorithm used by Rosetta's electron-density
score: atoms are represented by element-specific single Gaussians and the
calculated and observed maps are compared with a masked Pearson correlation.
The implementation is native PyTorch, batched, and differentiable with respect
to Cartesian coordinates.

The Rosetta source is published under its own license.  No Rosetta source code
is included here; the equations and cryo-EM scattering parameters are
independently expressed in PyTorch from the published implementation.
"""

from __future__ import annotations

import gzip
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy
import torch


# atomic number -> (single-Gaussian weight, intrinsic sigma)
# These are Rosetta's cryo-EM/electron-scattering parameters fitted over the
# 20--2 A resolution range. Unknown heavy elements use carbon, as in Rosetta.
_CRYOEM_SCATTERERS = {
    6: (6.00000, 7.10668),
    7: (5.28737, 6.03448),
    8: (4.74213, 5.17616),
    11: (11.42607, 6.58734),
    12: (12.45197, 7.34364),
    15: (13.12395, 7.48955),
    16: (12.34197, 7.05366),
    19: (21.48425, 7.09360),
    20: (23.70586, 7.47775),
    26: (17.13431, 5.96932),
    27: (15.70905, 5.56662),
    28: (15.70905, 5.56662),
    30: (15.70905, 5.56662),
}


@dataclass(frozen=True)
class ElectronDensityMap:
    """An MRC/CCP4 map in Cartesian axis order.

    Attributes:
        density: Map values with shape ``[z, y, x]``.
        origin: Cartesian coordinate of voxel ``[0, 0, 0]``, in ``[x,y,z]``.
        voxel_size: Voxel spacing in Angstrom, in ``[x,y,z]``.
    """

    density: torch.Tensor
    origin: torch.Tensor
    voxel_size: torch.Tensor

    def to(self, *args, **kwargs) -> "ElectronDensityMap":
        """Return a copy whose tensors have been passed through ``Tensor.to``."""
        return ElectronDensityMap(
            self.density.to(*args, **kwargs),
            self.origin.to(*args, **kwargs),
            self.voxel_size.to(*args, **kwargs),
        )


def _open_binary(path: str | Path) -> BinaryIO:
    path = Path(path)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _mrc_endian(header: bytes) -> str:
    """Infer MRC byte order from dimensions, mode, and the machine stamp."""
    little = struct.unpack_from("<4i", header, 0)
    big = struct.unpack_from(">4i", header, 0)

    def plausible(values):
        return all(0 < n < 10_000_000 for n in values[:3]) and values[3] in {
            0,
            1,
            2,
            6,
        }

    if plausible(little) and not plausible(big):
        return "<"
    if plausible(big) and not plausible(little):
        return ">"
    # 0x44 0x41 is the standard little-endian IEEE machine stamp.
    return "<" if header[212:214] in (b"DA", b"DD", b"\x44\x41") else ">"


def read_mrc(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ElectronDensityMap:
    """Read an MRC/CCP4 density map, including gzip-compressed maps.

    Axis permutations (MAPC/MAPR/MAPS), extended headers, integer map modes,
    Cartesian origins, and start-index origins are handled.  Skewed unit cells
    are rejected because the scorer currently assumes an orthogonal voxel grid.
    """
    with _open_binary(path) as inp:
        header = inp.read(1024)
        if len(header) != 1024:
            raise ValueError(f"{path}: truncated MRC header")
        if header[208:212] != b"MAP ":
            raise ValueError(f"{path}: missing MRC MAP signature")

        endian = _mrc_endian(header)
        nc, nr, ns, mode = struct.unpack_from(f"{endian}4i", header, 0)
        starts = struct.unpack_from(f"{endian}3i", header, 16)
        mx, my, mz = struct.unpack_from(f"{endian}3i", header, 28)
        cell = struct.unpack_from(f"{endian}6f", header, 40)
        mapc, mapr, maps = struct.unpack_from(f"{endian}3i", header, 64)
        nsymbt = struct.unpack_from(f"{endian}i", header, 92)[0]
        cart_origin = struct.unpack_from(f"{endian}3f", header, 196)

        if sorted((mapc, mapr, maps)) != [1, 2, 3]:
            raise ValueError(f"{path}: invalid MRC axis order {(mapc, mapr, maps)}")
        if min(mx, my, mz) <= 0:
            raise ValueError(f"{path}: invalid unit-cell sampling {(mx, my, mz)}")
        if any(abs(angle - 90.0) > 1e-3 for angle in cell[3:]):
            raise ValueError(f"{path}: non-orthogonal MRC cells are not supported")
        if nsymbt < 0:
            raise ValueError(f"{path}: invalid extended-header size {nsymbt}")

        mode_dtype = {
            0: numpy.dtype("i1"),
            1: numpy.dtype(f"{endian}i2"),
            2: numpy.dtype(f"{endian}f4"),
            6: numpy.dtype(f"{endian}u2"),
        }.get(mode)
        if mode_dtype is None:
            raise ValueError(f"{path}: unsupported MRC mode {mode}")

        inp.seek(1024 + nsymbt)
        count = nc * nr * ns
        payload = inp.read(count * mode_dtype.itemsize)
        if len(payload) != count * mode_dtype.itemsize:
            raise ValueError(f"{path}: truncated MRC density payload")

    raw = numpy.frombuffer(payload, dtype=mode_dtype, count=count).reshape(ns, nr, nc)
    # Raw dimensions are section, row, column. Return conventional [z, y, x].
    raw_axis_for_cart = {maps: 0, mapr: 1, mapc: 2}
    zyx = raw.transpose(
        raw_axis_for_cart[3], raw_axis_for_cart[2], raw_axis_for_cart[1]
    ).copy()

    voxel_size = numpy.asarray(cell[:3], dtype=numpy.float64) / numpy.asarray(
        (mx, my, mz), dtype=numpy.float64
    )
    if numpy.any(voxel_size <= 0):
        raise ValueError(f"{path}: invalid voxel size {voxel_size.tolist()}")

    origin = numpy.asarray(cart_origin, dtype=numpy.float64)
    if numpy.allclose(origin, 0.0) and any(start != 0 for start in starts):
        start_xyz = numpy.zeros(3, dtype=numpy.float64)
        start_xyz[mapc - 1] = starts[0]
        start_xyz[mapr - 1] = starts[1]
        start_xyz[maps - 1] = starts[2]
        origin = start_xyz * voxel_size

    return ElectronDensityMap(
        density=torch.as_tensor(zyx, dtype=dtype, device=device),
        origin=torch.as_tensor(origin, dtype=dtype, device=device),
        voxel_size=torch.as_tensor(voxel_size, dtype=dtype, device=device),
    )


def _element_atomic_numbers() -> dict[str, int]:
    # Keep this independent of a specific ParameterDatabase so the MRC scorer
    # can also be used directly on model outputs.
    return {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "NA": 11,
        "MG": 12,
        "P": 15,
        "S": 16,
        "K": 19,
        "CA": 20,
        "FE": 26,
        "CO": 27,
        "NI": 28,
        "ZN": 30,
    }


def block_type_atomic_numbers(packed_block_types, device=None) -> torch.Tensor:
    """Return atomic numbers per active block type, shape ``[n_block_types, max_n_atoms]``.

    Padding atoms receive zero. Element identity comes from TMol atom types;
    name parsing is only a fallback for custom types lacking an element entry.
    """
    pbt = packed_block_types
    atom_type_elements = {atom.name: atom.element for atom in pbt.chem_db.atom_types}
    element_z = {
        element.name.upper(): element.atomic_number
        for element in pbt.chem_db.element_types
    }

    block_z = []
    for block_type in pbt.active_block_types:
        values = []
        for atom in block_type.atoms:
            element = atom_type_elements.get(atom.atom_type)
            if element is None:
                from tmol.chemical import get_element_from_atom_name

                element = get_element_from_atom_name(atom.name)
            values.append(element_z.get(element.upper(), 6))
        values.extend([0] * (pbt.max_n_atoms - len(values)))
        block_z.append(values)
    return torch.tensor(block_z, dtype=torch.int64, device=device)


def atomic_numbers_for_pose_stack(pose_stack) -> torch.Tensor:
    """Return atomic numbers aligned to ``pose_stack.coords``.

    Padding atoms receive zero.
    """
    pbt = pose_stack.packed_block_types
    block_z = block_type_atomic_numbers(pbt, pose_stack.device)

    result = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_pose_atoms),
        dtype=torch.int64,
        device=pose_stack.device,
    )
    for pose_index in range(pose_stack.n_poses):
        real_blocks = torch.nonzero(
            pose_stack.block_type_ind[pose_index] >= 0, as_tuple=False
        ).flatten()
        for block_index_tensor in real_blocks:
            block_index = int(block_index_tensor)
            block_type_index = int(pose_stack.block_type_ind[pose_index, block_index])
            offset = int(pose_stack.block_coord_offset[pose_index, block_index])
            n_atoms = int(pbt.n_atoms[block_type_index])
            result[pose_index, offset : offset + n_atoms] = block_z[
                block_type_index, :n_atoms
            ]
    return result


def _scattering_tensors(
    atomic_numbers: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.full_like(atomic_numbers, 6.0, dtype=dtype)
    sigmas = torch.full_like(atomic_numbers, 7.10668, dtype=dtype)
    for atomic_number, (weight, sigma) in _CRYOEM_SCATTERERS.items():
        selected = atomic_numbers == atomic_number
        weights = torch.where(selected, weight, weights)
        sigmas = torch.where(selected, sigma, sigmas)
    return weights, sigmas


class DensityCorrelation(torch.nn.Module):
    """Synthesize density and evaluate a Rosetta-style masked correlation.

    ``forward`` returns ``-CC`` so lower is better and the result can be added
    directly to a TMol energy. ``correlation`` returns the corresponding CC.

    Args:
        density_map: Observed map in ``[z,y,x]`` order with Cartesian metadata.
        resolution: Map resolution in Angstrom. It controls the effective
            B-factor exactly as in Rosetta's map reader.
        atom_mask_radius: Radius of the smooth atom-union scoring mask.
        mask_padding: Extra stencil support beyond the mask radius.
        density_cutoff: Gaussian density below this value is omitted.
        atom_chunk_size: Atoms processed at once; lower values reduce memory.
    """

    def __init__(
        self,
        density_map: ElectronDensityMap,
        resolution: float,
        *,
        atom_mask_radius: float = 3.0,
        mask_padding: float = 2.0,
        density_cutoff: float = 1e-4,
        atom_chunk_size: int = 128,
    ):
        super().__init__()
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if atom_mask_radius <= 0 or mask_padding < 0 or density_cutoff <= 0:
            raise ValueError("invalid density mask parameters")
        if atom_chunk_size <= 0:
            raise ValueError("atom_chunk_size must be positive")

        self.register_buffer("observed_density", density_map.density)
        self.register_buffer("origin", density_map.origin)
        self.register_buffer("voxel_size", density_map.voxel_size)
        self.resolution = float(resolution)
        self.atom_mask_radius = float(atom_mask_radius)
        self.mask_padding = float(mask_padding)
        self.density_cutoff = float(density_cutoff)
        self.atom_chunk_size = int(atom_chunk_size)

    @classmethod
    def from_mrc(
        cls,
        path: str | Path,
        resolution: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "DensityCorrelation":
        """Construct a scorer directly from an MRC or MRC.GZ file."""
        return cls(read_mrc(path, device=device, dtype=dtype), resolution, **kwargs)

    def _gaussian_parameters(
        self, atomic_numbers: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, sigmas = _scattering_tensors(atomic_numbers, dtype)
        max_voxel = float(self.voxel_size.max())
        effective_resolution = max(2.0 * max_voxel, self.resolution)
        effective_b = 4.0 * effective_resolution * effective_resolution
        k_from_b = 4.0 * math.pi * math.pi / effective_b
        k = torch.minimum(
            (math.pi * math.pi) / sigmas, torch.full_like(sigmas, k_from_b)
        )
        amplitude = weights * k.pow(1.5)
        density_radius_sq = (
            torch.log(amplitude.clamp_min(self.density_cutoff))
            - math.log(self.density_cutoff)
        ) / k
        return k, amplitude, density_radius_sq

    def _stencil_offsets(self, support_radius: float, device) -> torch.Tensor:
        radii = torch.ceil(
            torch.as_tensor(support_radius, dtype=self.voxel_size.dtype, device=device)
            / self.voxel_size
        ).to(torch.int64)
        x = torch.arange(-radii[0], radii[0] + 1, device=device)
        y = torch.arange(-radii[1], radii[1] + 1, device=device)
        z = torch.arange(-radii[2], radii[2] + 1, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)

    def synthesize_density(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Render element-specific Gaussian atoms onto the observed map grid.

        Args:
            coords: Coordinates shaped ``[pose, atom, 3]`` or ``[atom, 3]``.
            atomic_numbers: Atomic numbers shaped ``[pose, atom]`` or ``[atom]``.
            atom_mask: Optional boolean mask with the atomic-number shape.
            return_mask: Also return the smooth union of atom-centered masks.
        """
        squeeze = coords.ndim == 2
        if squeeze:
            coords = coords.unsqueeze(0)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [pose, atom, 3] or [atom, 3]")

        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0).expand(coords.shape[0], -1)
        if atomic_numbers.shape != coords.shape[:2]:
            raise ValueError("atomic_numbers must align with coords")
        atomic_numbers = atomic_numbers.to(device=coords.device, dtype=torch.int64)

        valid_atoms = (atomic_numbers > 1) & torch.isfinite(coords).all(dim=-1)
        if atom_mask is not None:
            if atom_mask.ndim == 1:
                atom_mask = atom_mask.unsqueeze(0).expand(coords.shape[0], -1)
            if atom_mask.shape != coords.shape[:2]:
                raise ValueError("atom_mask must align with coords")
            valid_atoms &= atom_mask.to(device=coords.device, dtype=torch.bool)

        k, amplitude, density_radius_sq = self._gaussian_parameters(
            atomic_numbers, coords.dtype
        )
        maximum_density_radius = float(torch.sqrt(density_radius_sq.max()).detach())
        support_radius = max(
            self.atom_mask_radius + self.mask_padding, maximum_density_radius
        )
        offsets = self._stencil_offsets(support_radius, coords.device)

        nz, ny, nx = self.observed_density.shape
        n_voxels = nz * ny * nx
        calculated = []
        scoring_masks = []
        for pose_index in range(coords.shape[0]):
            flat_density = torch.zeros(
                n_voxels, dtype=coords.dtype, device=coords.device
            )
            flat_log_complement = torch.zeros_like(flat_density)
            atom_indices = torch.nonzero(
                valid_atoms[pose_index], as_tuple=False
            ).flatten()

            for begin in range(0, atom_indices.numel(), self.atom_chunk_size):
                chosen = atom_indices[begin : begin + self.atom_chunk_size]
                atom_coords = coords[pose_index, chosen]
                center_grid = (atom_coords - self.origin) / self.voxel_size
                grid_xyz = (
                    torch.floor(center_grid).to(torch.int64)[:, None, :] + offsets
                )

                inside = (
                    (grid_xyz[..., 0] >= 0)
                    & (grid_xyz[..., 0] < nx)
                    & (grid_xyz[..., 1] >= 0)
                    & (grid_xyz[..., 1] < ny)
                    & (grid_xyz[..., 2] >= 0)
                    & (grid_xyz[..., 2] < nz)
                )
                voxel_xyz = self.origin + grid_xyz.to(coords.dtype) * self.voxel_size
                distance_sq = ((voxel_xyz - atom_coords[:, None, :]) ** 2).sum(dim=-1)
                inside &= distance_sq <= support_radius * support_radius

                linear_index = (
                    grid_xyz[..., 2] * (ny * nx)
                    + grid_xyz[..., 1] * nx
                    + grid_xyz[..., 0]
                )
                gaussian = amplitude[pose_index, chosen, None] * torch.exp(
                    -k[pose_index, chosen, None] * distance_sq
                )
                gaussian = torch.where(
                    inside
                    & (distance_sq <= density_radius_sq[pose_index, chosen, None]),
                    gaussian,
                    0.0,
                )

                smooth_mask = torch.sigmoid(
                    self.atom_mask_radius * self.atom_mask_radius - distance_sq
                )
                smooth_mask = torch.where(inside, smooth_mask, 0.0)

                valid_entries = inside.flatten()
                indices = linear_index.flatten()[valid_entries]
                flat_density = flat_density.scatter_add(
                    0, indices, gaussian.flatten()[valid_entries]
                )
                log_complement = torch.log1p(
                    -smooth_mask.clamp(max=1.0 - torch.finfo(coords.dtype).eps)
                )
                flat_log_complement = flat_log_complement.scatter_add(
                    0, indices, log_complement.flatten()[valid_entries]
                )

            calculated.append(flat_density.reshape(nz, ny, nx))
            scoring_masks.append(
                (1.0 - torch.exp(flat_log_complement)).reshape(nz, ny, nx)
            )

        calculated_tensor = torch.stack(calculated)
        scoring_mask_tensor = torch.stack(scoring_masks)
        if squeeze:
            calculated_tensor = calculated_tensor[0]
            scoring_mask_tensor = scoring_mask_tensor[0]
        if return_mask:
            return calculated_tensor, scoring_mask_tensor
        return calculated_tensor

    def correlation(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the masked Pearson map correlation for every input pose."""
        squeeze = coords.ndim == 2
        calculated, scoring_mask = self.synthesize_density(
            coords, atomic_numbers, atom_mask, return_mask=True
        )
        if squeeze:
            calculated = calculated.unsqueeze(0)
            scoring_mask = scoring_mask.unsqueeze(0)

        observed = self.observed_density.to(dtype=calculated.dtype).unsqueeze(0)
        reduce_dims = (1, 2, 3)
        volume = scoring_mask.sum(dim=reduce_dims).clamp_min(1.0)
        sum_calc = (scoring_mask * calculated).sum(dim=reduce_dims)
        sum_obs = (scoring_mask * observed).sum(dim=reduce_dims)
        centered_cross = (scoring_mask * calculated * observed).sum(
            dim=reduce_dims
        ) - sum_calc * sum_obs / volume
        variance_calc = (scoring_mask * calculated.square()).sum(
            dim=reduce_dims
        ) - sum_calc.square() / volume
        variance_obs = (scoring_mask * observed.square()).sum(
            dim=reduce_dims
        ) - sum_obs.square() / volume
        denominator = torch.sqrt(
            variance_calc.clamp_min(0.0) * variance_obs.clamp_min(0.0)
        ).clamp_min(torch.finfo(calculated.dtype).eps)
        result = centered_cross / denominator
        return result[0] if squeeze else result

    def forward(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return negative map correlation, suitable as a minimization energy."""
        return -self.correlation(coords, atomic_numbers, atom_mask)

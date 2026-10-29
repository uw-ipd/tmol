"""CPU construction and serialization of rigid-receptor energy grids.

The serialized representation is a ``torch.save`` dictionary containing only
tensors and primitive Python values.  It is intended to be loaded by later GPU
interpolation stages without requiring the original ``PoseStack``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from tmol.database import ParameterDatabase
from tmol.pose import PoseStack
from tmol.score.elec import ElecEnergyTerm
from tmol.score.ljlk import LJLKEnergyTerm
from tmol.score.lk_ball import LKBallEnergyTerm
from tmol.score import ScoreFunction, ScoreType, beta2016_score_function

from ._common import build_cell_list, cell_grid
from .live_lk_ball import _generate_reference_waters

DEFAULT_VOXEL_SPACING = 0.25
DEFAULT_BBOX_PADDING = 4.0
_SERIALIZATION_VERSION = 3
OUT_OF_BOUNDS_PENALTY = 50.0
GRID_LK_BALL_WATER_DISTANCE = 2.65
GRID_LK_BALL_FADE = 1.0


def xform_rep(rep: torch.Tensor) -> torch.Tensor:
    """Map raw fa_rep from ``[0, inf)`` to Rosetta's bounded representation."""
    return torch.pow(rep + 1, -0.25)


def ixform_rep(transformed_rep: torch.Tensor) -> torch.Tensor:
    """Invert Rosetta's fa_rep grid transform."""
    return torch.pow(transformed_rep, -4) - 1


def _mirror_spline_inverse(n: int) -> torch.Tensor:
    """Inverse of the cubic B-spline interpolation matrix with mirror boundaries.

    Row ``k`` is ``(c[k-1] + 4 c[k] + c[k+1]) / 6`` with ``c[-1] = c[1]`` and
    ``c[n] = c[n-2]`` (Rosetta's mirror boundary).
    """
    matrix = torch.zeros((n, n), dtype=torch.float64)
    index = torch.arange(n)
    matrix[index, index] = 4.0 / 6.0
    matrix[index[:-1], index[:-1] + 1] = 1.0 / 6.0
    matrix[index[1:], index[1:] - 1] = 1.0 / 6.0
    matrix[0, 1] *= 2
    matrix[-1, -2] *= 2
    return torch.linalg.inv(matrix)


def compute_cubic_bspline_coefficients(values: torch.Tensor) -> torch.Tensor:
    """Compute separable cubic B-spline coefficients with mirror boundaries.

    The final three tensor axes are the spatial ``(x, y, z)`` axes. Leading
    axes, such as the grid atom-type axis, index independent grids. Runs on the
    device of ``values`` (float64 internally, cast back) as three batched
    matrix products, and does not preserve autograd.
    """
    if values.ndim < 3:
        raise ValueError("grid values must have at least three dimensions")
    if any(size <= 0 for size in values.shape[-3:]):
        raise ValueError("spatial grid dimensions must be positive")

    coefficients = values.detach().to(torch.float64)
    for axis in range(coefficients.ndim - 3, coefficients.ndim):
        n = coefficients.shape[axis]
        if n > 1:
            inverse = _mirror_spline_inverse(n).to(values.device)
            coefficients = (coefficients.movedim(axis, -1) @ inverse.T).movedim(
                -1, axis
            )
    return coefficients.contiguous().to(values.dtype)


@dataclass(frozen=True)
class GridGeometry:
    """Cubic grid geometry with zero-based tensor indices."""

    origin: torch.Tensor
    dims: tuple[int, int, int]
    voxel_spacing: float
    ligand_com: torch.Tensor
    max_radius: float

    def __post_init__(self):
        if len(self.dims) != 3 or any(dim <= 0 for dim in self.dims):
            raise ValueError("dims must contain three positive integers")
        if self.voxel_spacing <= 0:
            raise ValueError("voxel_spacing must be positive")

    def cartesian_from_zero_based_index(
        self, index: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Return the Cartesian position of a zero-based ``(x, y, z)`` node."""
        index_tensor = torch.as_tensor(
            index, dtype=self.origin.dtype, device=self.origin.device
        )
        return self.origin + self.voxel_spacing * index_tensor

    def zero_based_index_from_cartesian(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return the continuous zero-based grid index of a Cartesian point."""
        return (xyz - self.origin) / self.voxel_spacing

    def as_dict(self) -> dict:
        return {
            "origin": self.origin,
            "dims": self.dims,
            "voxel_spacing": self.voxel_spacing,
            "ligand_com": self.ligand_com,
            "max_radius": self.max_radius,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "GridGeometry":
        return cls(
            origin=state["origin"],
            dims=tuple(state["dims"]),
            voxel_spacing=float(state["voxel_spacing"]),
            ligand_com=state["ligand_com"],
            max_radius=float(state["max_radius"]),
        )


@dataclass(frozen=True)
class GridSet:
    """All grids and metadata needed by later rigid-receptor scoring stages.

    ``fa_rep`` is stored transformed. ``fa_atr``, ``fa_sol``, and ``fa_elec``
    are raw. The corresponding ``*_coeffs`` tensors hold mirror-boundary cubic
    B-spline coefficients. The LK-ball fields contain Rosetta's single
    weighted gridded aggregate when built.
    """

    geometry: GridGeometry
    atom_type_names: tuple[str, ...]
    atom_type_indices: torch.Tensor
    fa_atr: torch.Tensor
    fa_rep: torch.Tensor
    fa_sol: torch.Tensor
    fa_elec: torch.Tensor
    lk_ball: torch.Tensor | None = None
    lk_ball_water: torch.Tensor | None = None
    lk_bridge: torch.Tensor | None = None
    lk_ball_enabled: bool = False
    lk_bridge_enabled: bool = False
    fa_atr_coeffs: torch.Tensor | None = None
    fa_rep_coeffs: torch.Tensor | None = None
    fa_sol_coeffs: torch.Tensor | None = None
    fa_elec_coeffs: torch.Tensor | None = None
    lk_ball_coeffs: torch.Tensor | None = None
    lk_ball_water_coeffs: torch.Tensor | None = None
    lk_bridge_coeffs: torch.Tensor | None = None

    def __post_init__(self):
        if self.atom_type_indices.shape != (len(self.atom_type_names),):
            raise ValueError("atom_type_indices must have one entry per grid type")

        def check_coefficients(grid_name: str, coefficient_name: str) -> None:
            grid = getattr(self, grid_name)
            coefficients = getattr(self, coefficient_name)
            if coefficients is None:
                coefficients = compute_cubic_bspline_coefficients(grid)
                object.__setattr__(self, coefficient_name, coefficients)
            if coefficients.dtype != grid.dtype or coefficients.device != grid.device:
                raise ValueError(
                    f"{coefficient_name} must match {grid_name} dtype and device"
                )

        for name in ("fa_atr", "fa_rep", "fa_sol", "fa_elec"):
            check_coefficients(name, f"{name}_coeffs")
        lk_fields = (
            ("lk_ball", "lk_ball_coeffs"),
            ("lk_ball_water", "lk_ball_water_coeffs"),
            ("lk_bridge", "lk_bridge_coeffs"),
        )
        lk_present = [getattr(self, name) is not None for name, _ in lk_fields]
        if any(lk_present) and not all(lk_present):
            raise ValueError("all three LK-ball grid families must be present together")
        for grid_name, coefficient_name in lk_fields:
            grid = getattr(self, grid_name)
            if grid is None:
                if getattr(self, coefficient_name) is not None:
                    raise ValueError(f"{coefficient_name} requires {grid_name}")
                continue
            check_coefficients(grid_name, coefficient_name)

    @property
    def atom_type_to_grid(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.atom_type_names)}

    @property
    def fa_lk(self) -> torch.Tensor:
        """Alias for the fa_sol grid computed by the isotropic LK potential."""
        return self.fa_sol

    def save(self, path: str | os.PathLike) -> None:
        """Serialize grids and geometry as a versioned ``torch.save`` dictionary."""
        state = {
            "version": _SERIALIZATION_VERSION,
            "geometry": self.geometry.as_dict(),
            "atom_type_names": self.atom_type_names,
            "atom_type_indices": self.atom_type_indices,
            "atom_type_to_grid": self.atom_type_to_grid,
            "fa_atr": self.fa_atr,
            "fa_rep": self.fa_rep,
            "fa_sol": self.fa_sol,
            "fa_elec": self.fa_elec,
            "fa_atr_coeffs": self.fa_atr_coeffs,
            "fa_rep_coeffs": self.fa_rep_coeffs,
            "fa_sol_coeffs": self.fa_sol_coeffs,
            "fa_elec_coeffs": self.fa_elec_coeffs,
            "lk_ball": self.lk_ball,
            "lk_ball_water": self.lk_ball_water,
            "lk_bridge": self.lk_bridge,
            "lk_ball_enabled": self.lk_ball_enabled,
            "lk_bridge_enabled": self.lk_bridge_enabled,
            "lk_ball_coeffs": self.lk_ball_coeffs,
            "lk_ball_water_coeffs": self.lk_ball_water_coeffs,
            "lk_bridge_coeffs": self.lk_bridge_coeffs,
        }
        torch.save(state, Path(path))

    @classmethod
    def load(
        cls, path: str | os.PathLike, map_location: str | torch.device = "cpu"
    ) -> "GridSet":
        """Load a grid file produced by :meth:`save`."""
        state = torch.load(Path(path), map_location=map_location, weights_only=True)
        if state.get("version") != _SERIALIZATION_VERSION:
            raise ValueError(
                f"unsupported grid serialization version: {state.get('version')}"
            )
        atom_type_names = tuple(state["atom_type_names"])
        expected_map = {name: index for index, name in enumerate(atom_type_names)}
        if state["atom_type_to_grid"] != expected_map:
            raise ValueError("serialized atom_type_to_grid is inconsistent")

        geometry = GridGeometry.from_dict(state["geometry"])
        return cls(
            geometry=geometry,
            atom_type_names=atom_type_names,
            atom_type_indices=state["atom_type_indices"],
            fa_atr=state["fa_atr"],
            fa_rep=state["fa_rep"],
            fa_sol=state["fa_sol"],
            fa_elec=state["fa_elec"],
            fa_atr_coeffs=state.get("fa_atr_coeffs"),
            fa_rep_coeffs=state.get("fa_rep_coeffs"),
            fa_sol_coeffs=state.get("fa_sol_coeffs"),
            fa_elec_coeffs=state.get("fa_elec_coeffs"),
            lk_ball=state["lk_ball"],
            lk_ball_water=state["lk_ball_water"],
            lk_bridge=state["lk_bridge"],
            lk_ball_enabled=bool(state.get("lk_ball_enabled", False)),
            lk_bridge_enabled=bool(state.get("lk_bridge_enabled", False)),
            lk_ball_coeffs=state.get("lk_ball_coeffs"),
            lk_ball_water_coeffs=state.get("lk_ball_water_coeffs"),
            lk_bridge_coeffs=state.get("lk_bridge_coeffs"),
        )

    def on(self, device: str | torch.device, dtype: torch.dtype) -> "GridSet":
        """``to(device, dtype)``, computed once per target and cached on the set.

        Lets every backend that shares this ``GridSet`` share one device copy.
        """
        cache = self.__dict__.setdefault("_device_copies", {})
        key = (torch.device(device), dtype)
        if key not in cache:
            cache[key] = self.to(device, dtype)
        return cache[key]

    def to(
        self, device: str | torch.device, dtype: torch.dtype | None = None
    ) -> "GridSet":
        """Return a copy of this grid set on ``device`` for interpolation."""
        device = torch.device(device)

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            target_dtype = dtype if value.dtype.is_floating_point else value.dtype
            return value.to(device=device, dtype=target_dtype)

        geometry = GridGeometry(
            origin=move(self.geometry.origin),
            dims=self.geometry.dims,
            voxel_spacing=self.geometry.voxel_spacing,
            ligand_com=move(self.geometry.ligand_com),
            max_radius=self.geometry.max_radius,
        )
        return GridSet(
            geometry=geometry,
            atom_type_names=self.atom_type_names,
            atom_type_indices=move(self.atom_type_indices),
            fa_atr=move(self.fa_atr),
            fa_rep=move(self.fa_rep),
            fa_sol=move(self.fa_sol),
            fa_elec=move(self.fa_elec),
            fa_atr_coeffs=move(self.fa_atr_coeffs),
            fa_rep_coeffs=move(self.fa_rep_coeffs),
            fa_sol_coeffs=move(self.fa_sol_coeffs),
            fa_elec_coeffs=move(self.fa_elec_coeffs),
            lk_ball=move(self.lk_ball),
            lk_ball_water=move(self.lk_ball_water),
            lk_bridge=move(self.lk_bridge),
            lk_ball_enabled=self.lk_ball_enabled,
            lk_bridge_enabled=self.lk_bridge_enabled,
            lk_ball_coeffs=move(self.lk_ball_coeffs),
            lk_ball_water_coeffs=move(self.lk_ball_water_coeffs),
            lk_bridge_coeffs=move(self.lk_bridge_coeffs),
        )


GRID_SCORE_TYPES = (
    ScoreType.fa_ljatr,
    ScoreType.fa_ljrep,
    ScoreType.fa_lk,
    ScoreType.fa_elec,
)

LK_BALL_SCORE_TYPES = (
    ScoreType.lk_ball_iso,
    ScoreType.lk_ball,
    ScoreType.lk_bridge,
    ScoreType.lk_bridge_uncpl,
)
LIVE_SCORE_TYPES = (ScoreType.hbond,) + LK_BALL_SCORE_TYPES


def compute_grid_geometry(
    ligand_coords: torch.Tensor,
    voxel_spacing: float = DEFAULT_VOXEL_SPACING,
    bbox_padding: float = DEFAULT_BBOX_PADDING,
) -> GridGeometry:
    """Compute the exact M2 ligand-centered cubic grid geometry."""
    if ligand_coords.device.type != "cpu":
        raise ValueError("grid geometry must be computed on CPU")
    if ligand_coords.shape[0] == 0:
        raise ValueError("at least one ligand atom is required")
    if not ligand_coords.dtype.is_floating_point:
        raise TypeError("ligand_coords must have a floating-point dtype")

    ligand_coords = ligand_coords.detach().to(dtype=torch.float64)
    ligand_com = torch.mean(ligand_coords, dim=0)
    measured_radius = torch.linalg.vector_norm(ligand_coords - ligand_com, dim=1).max()
    max_radius = max(float(measured_radius), 1.5 * bbox_padding)
    half_extent = max_radius + bbox_padding
    origin = ligand_com - half_extent
    dim = math.ceil(2 * half_extent / voxel_spacing)
    return GridGeometry(
        origin=origin,
        dims=(dim, dim, dim),
        voxel_spacing=float(voxel_spacing),
        ligand_com=ligand_com,
        max_radius=max_radius,
    )


def probe_atom_types_from_name3(
    pose_stack: PoseStack, ligand_name3: str = "LG1"
) -> tuple[str, ...]:
    """Return the sorted unique chemical atom types in matching ligand blocks."""
    names = set()
    pbt = pose_stack.packed_block_types
    for block_type_index_tensor in pose_stack.block_type_ind.reshape(-1):
        block_type_index = int(block_type_index_tensor)
        if block_type_index < 0:
            continue
        block_type = pbt.active_block_types[block_type_index]
        if block_type.name3 == ligand_name3:
            names.update(atom.atom_type for atom in block_type.atoms)
    if not names:
        raise ValueError(f"no ligand block type with name3={ligand_name3!r}")
    return tuple(sorted(names))


def _setup_scoring_parameters(
    pose_stack: PoseStack, param_db: ParameterDatabase
) -> tuple[LJLKEnergyTerm, ElecEnergyTerm]:
    device = torch.device("cpu")
    ljlk_term = LJLKEnergyTerm(param_db, device)
    elec_term = ElecEnergyTerm(param_db, device)
    for block_type in pose_stack.packed_block_types.active_block_types:
        ljlk_term.setup_block_type(block_type)
        elec_term.setup_block_type(block_type)
    ljlk_term.setup_packed_block_types(pose_stack.packed_block_types)
    elec_term.setup_packed_block_types(pose_stack.packed_block_types)
    return ljlk_term, elec_term


def _pack_parameter_tensors(
    ljlk_term: LJLKEnergyTerm, elec_term: ElecEnergyTerm
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ljlk_type_params = torch.stack(
        [
            value.to(dtype=torch.float64)
            for value in (
                ljlk_term.type_params.lj_radius,
                ljlk_term.type_params.lj_wdepth,
                ljlk_term.type_params.lk_dgfree,
                ljlk_term.type_params.lk_lambda,
                ljlk_term.type_params.lk_volume,
                ljlk_term.type_params.is_donor,
                ljlk_term.type_params.is_hydroxyl,
                ljlk_term.type_params.is_polarh,
                ljlk_term.type_params.is_acceptor,
                ljlk_term.type_params.is_carbon_lk,
            )
        ],
        dim=1,
    )
    ljlk_global_params = torch.stack(
        [
            value.to(dtype=torch.float64)
            for value in (
                ljlk_term.global_params.max_dis,
                ljlk_term.global_params.lj_dlin_sigma_factor,
                ljlk_term.global_params.lj_hbond_dis,
                ljlk_term.global_params.lj_hbond_OH_donor_dis,
                ljlk_term.global_params.lj_hbond_hdis,
            )
        ],
        dim=1,
    )
    elec_global_params = torch.stack(
        [
            elec_term.global_params.elec_sigmoidal_die_D,
            elec_term.global_params.elec_sigmoidal_die_D0,
            elec_term.global_params.elec_sigmoidal_die_S,
            elec_term.global_params.elec_min_dis,
            elec_term.global_params.elec_max_dis,
        ],
        dim=0,
    ).to(dtype=torch.float64)
    return tuple(
        tensor.contiguous()
        for tensor in (ljlk_type_params, ljlk_global_params, elec_global_params)
    )


def _extract_ligand_and_receptor_atoms(
    pose_stack: PoseStack,
    ligand_name3: str,
    reference_waters: torch.Tensor | None = None,
    excluded_receptor_blocks: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    pbt = pose_stack.packed_block_types
    ligand_coords = []
    receptor_coords = []
    receptor_atom_types = []
    receptor_charges = []
    receptor_waters = []

    for block_index in range(pose_stack.max_n_blocks):
        block_type_index = int(pose_stack.block_type_ind[0, block_index])
        if block_type_index < 0:
            continue
        block_type = pbt.active_block_types[block_type_index]
        n_atoms = int(pbt.n_atoms[block_type_index])
        offset = int(pose_stack.block_coord_offset64[0, block_index])
        coords = pose_stack.coords[0, offset : offset + n_atoms]
        if block_type.name3 == ligand_name3:
            ligand_coords.append(coords)
        elif block_index in excluded_receptor_blocks:
            continue
        else:
            receptor_coords.append(coords)
            receptor_atom_types.append(pbt.atom_types[block_type_index, :n_atoms])
            receptor_charges.append(pbt.elec_partial_charge[block_type_index, :n_atoms])
            if reference_waters is not None:
                receptor_waters.append(reference_waters[offset : offset + n_atoms])

    if not ligand_coords:
        raise ValueError(f"no ligand block with name3={ligand_name3!r} in pose")
    if not receptor_coords:
        raise ValueError("pose contains no receptor atoms")
    return (
        torch.cat(ligand_coords).detach().to(dtype=torch.float64).contiguous(),
        torch.cat(receptor_coords).detach().to(dtype=torch.float64).contiguous(),
        torch.cat(receptor_atom_types).to(dtype=torch.int64).contiguous(),
        torch.cat(receptor_charges).detach().to(dtype=torch.float64).contiguous(),
        (
            torch.cat(receptor_waters).detach().to(dtype=torch.float64).contiguous()
            if reference_waters is not None
            else None
        ),
    )


def compute_receptor_grids(
    receptor_coords: torch.Tensor,
    receptor_types: torch.Tensor,
    receptor_charges: torch.Tensor,
    probe_types: torch.Tensor,
    origin: torch.Tensor,
    dims: tuple[int, int, int],
    voxel_spacing: float,
    ljlk_type_params: torch.Tensor,
    ljlk_global_params: torch.Tensor,
    elec_global_params: torch.Tensor,
    *,
    receptor_waters: torch.Tensor | None = None,
    lk_ball_weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Receptor-vs-probe grids on ``device``: ``[fa_atr, fa_rep, fa_sol, fa_elec]``
    plus ``[lk_ball, lk_ball_water, lk_bridge]`` when ``lk_ball_weights`` is given.

    ``fa_rep`` is transformed, typed grids are ``[probe, x, y, z]``. Inputs are
    CPU tensors (float64 coordinates and charges, int64 types); a CSR cell list
    over the receptor atoms bounds the neighbor search on the device.
    """
    from .potentials.compiled import build_grid as build_grid_op

    device = torch.device(device)
    max_dis = float(max(ljlk_global_params[0, 0], elec_global_params[4]))
    cell_origin, cell_dims = cell_grid(receptor_coords, max_dis)
    cell_offsets, cell_indices = build_cell_list(
        receptor_coords, cell_origin, cell_dims, max_dis
    )
    build_lk_ball = lk_ball_weights is not None
    if receptor_waters is None:
        receptor_waters = torch.zeros((1, 4, 3), dtype=torch.float64)
    if lk_ball_weights is None:
        lk_ball_weights = torch.zeros(4, dtype=torch.float64)
    on = lambda t: t.contiguous().to(device)  # noqa: E731
    grids = build_grid_op(
        on(receptor_coords),
        on(receptor_types),
        on(receptor_charges),
        on(receptor_waters),
        on(probe_types),
        on(ljlk_type_params),
        ljlk_global_params.contiguous(),
        elec_global_params.contiguous(),
        lk_ball_weights.contiguous(),
        on(cell_offsets),
        on(cell_indices),
        on(cell_origin),
        on(cell_dims),
        on(origin),
        torch.tensor(dims, dtype=torch.int64),
        float(voxel_spacing),
        build_lk_ball,
    )
    typed = (probe_types.shape[0], *dims)
    shapes = [typed] * 3 + [tuple(dims)] + [typed] * 3
    return [grid.view(shape) for grid, shape in zip(grids, shapes)]


def build_grid(
    pose_stack: PoseStack,
    probe_atom_types: Sequence[str],
    param_db: ParameterDatabase,
    *,
    ligand_name3: str = "LG1",
    voxel_spacing: float = DEFAULT_VOXEL_SPACING,
    bbox_padding: float = DEFAULT_BBOX_PADDING,
    score_function: ScoreFunction | None = None,
    build_lk_ball: bool = False,
    excluded_receptor_blocks: tuple[int, ...] = (),
    device: str | torch.device = "cpu",
) -> GridSet:
    """Build beta16 all-atom grids and optional Rosetta-style LK-ball grids.

    The grids are computed (and spline-prefiltered) on ``device``.

    Probe atom-type names are deduplicated and sorted, making the grid-to-type
    mapping deterministic even when the caller passes a set.
    """
    if pose_stack.n_poses != 1:
        raise ValueError("M2b grid construction supports exactly one pose")
    if pose_stack.device.type != "cpu" or pose_stack.coords.device.type != "cpu":
        raise ValueError("M2b grid construction is CPU-only")
    if len(set(excluded_receptor_blocks)) != len(excluded_receptor_blocks):
        raise ValueError("excluded_receptor_blocks must not contain duplicates")
    if any(
        block < 0 or block >= pose_stack.max_n_blocks
        for block in excluded_receptor_blocks
    ):
        raise ValueError("excluded receptor block is out of range")
    atom_type_names = tuple(sorted(set(probe_atom_types)))
    if not atom_type_names:
        raise ValueError("at least one probe atom type is required")

    ljlk_term, elec_term = _setup_scoring_parameters(pose_stack, param_db)
    unknown = [
        name for name in atom_type_names if name not in ljlk_term.atom_type_index
    ]
    if unknown:
        raise ValueError(f"unknown probe atom types: {unknown}")
    probe_atom_type_indices = torch.tensor(
        [ljlk_term.atom_type_index.get_loc(name) for name in atom_type_names],
        dtype=torch.int64,
    )

    receptor_waters = None
    if build_lk_ball:
        lk_ball_term = LKBallEnergyTerm(param_db, torch.device("cpu"))
        for block_type in pose_stack.packed_block_types.active_block_types:
            lk_ball_term.setup_block_type(block_type)
        lk_ball_term.setup_packed_block_types(pose_stack.packed_block_types)
        reference_waters = _generate_reference_waters(
            pose_stack,
            lk_ball_term,
            water_distance=GRID_LK_BALL_WATER_DISTANCE,
        )
    else:
        reference_waters = None
    (
        ligand_coords,
        receptor_coords,
        receptor_types,
        receptor_charges,
        receptor_waters,
    ) = _extract_ligand_and_receptor_atoms(
        pose_stack,
        ligand_name3,
        reference_waters,
        tuple(excluded_receptor_blocks),
    )
    geometry = compute_grid_geometry(ligand_coords, voxel_spacing, bbox_padding)
    ljlk_type_params, ljlk_global_params, elec_global_params = _pack_parameter_tensors(
        ljlk_term, elec_term
    )

    lk_ball_weights = None
    if build_lk_ball:
        if score_function is None:
            score_function = beta2016_score_function(
                torch.device("cpu"), param_db=param_db
            )
        lk_ball_weights = torch.stack(
            [score_function.get_weight(t) for t in LK_BALL_SCORE_TYPES]
        ).to(dtype=torch.float64, device="cpu")
    device = torch.device(device)
    fa_atr, fa_rep, fa_sol, fa_elec, *lk = compute_receptor_grids(
        receptor_coords,
        receptor_types,
        receptor_charges,
        probe_atom_type_indices,
        geometry.origin,
        geometry.dims,
        geometry.voxel_spacing,
        ljlk_type_params,
        ljlk_global_params,
        elec_global_params,
        receptor_waters=receptor_waters,
        lk_ball_weights=lk_ball_weights,
        device=device,
    )
    lk_ball, lk_ball_water, lk_bridge = lk if build_lk_ball else (None,) * 3
    lk_ball_enabled = build_lk_ball and bool(torch.any(lk_ball_weights != 0))
    lk_bridge_enabled = build_lk_ball and bool(torch.any(lk_ball_weights[2:] != 0))
    on = lambda t: t.contiguous().to(device)  # noqa: E731
    geometry = GridGeometry(
        origin=on(geometry.origin),
        dims=geometry.dims,
        voxel_spacing=geometry.voxel_spacing,
        ligand_com=on(geometry.ligand_com),
        max_radius=geometry.max_radius,
    )
    return GridSet(
        geometry=geometry,
        atom_type_names=atom_type_names,
        atom_type_indices=on(probe_atom_type_indices),
        fa_atr=fa_atr,
        fa_rep=fa_rep,
        fa_sol=fa_sol,
        fa_elec=fa_elec,
        lk_ball=lk_ball,
        lk_ball_water=lk_ball_water,
        lk_bridge=lk_bridge,
        lk_ball_enabled=lk_ball_enabled,
        lk_bridge_enabled=lk_bridge_enabled,
    )

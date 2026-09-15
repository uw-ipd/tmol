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
from typing import TYPE_CHECKING, Sequence

import torch

from tmol.database import ParameterDatabase
from tmol.pose.pose_stack import PoseStack
from tmol.score.elec.elec_energy_term import ElecEnergyTerm
from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
from tmol.score.score_types import ScoreType

if TYPE_CHECKING:
    from tmol.score.score_function import ScoreFunction
    from .live_hbond import LiveHBondData
    from .live_ligand_intra import LiveLigandIntraData
    from .live_lk_ball import LiveLKBallData


DEFAULT_VOXEL_SPACING = 0.25
DEFAULT_BBOX_PADDING = 4.0
_SERIALIZATION_VERSION = 3
_CUBIC_BSPLINE_POLE = math.sqrt(3.0) - 2.0
OUT_OF_BOUNDS_PENALTY = 50.0
GRID_LK_BALL_WATER_DISTANCE = 2.65
GRID_LK_BALL_FADE = 1.0


def xform_rep(rep: torch.Tensor) -> torch.Tensor:
    """Map raw fa_rep from ``[0, inf)`` to Rosetta's bounded representation."""
    return torch.pow(rep + 1, -0.25)


def ixform_rep(transformed_rep: torch.Tensor) -> torch.Tensor:
    """Invert Rosetta's fa_rep grid transform."""
    return torch.pow(transformed_rep, -4) - 1


def _smooth_fa_rep_grid(
    fa_rep: torch.Tensor, smoothing: float, voxel_spacing: float
) -> torch.Tensor:
    """Apply the truncated spherical maximum filter used for transformed fa_rep."""
    if not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("fa_rep_smoothing must be finite and nonnegative")
    if smoothing == 0:
        return fa_rep

    window_width = math.ceil(smoothing / voxel_spacing)
    smoothing_squared = smoothing * smoothing
    smoothed = fa_rep.clone()
    spatial_shape = fa_rep.shape[-3:]
    for di in range(-window_width, window_width + 1):
        for dj in range(-window_width, window_width + 1):
            for dk in range(-window_width, window_width + 1):
                distance_squared = (
                    (di * voxel_spacing) ** 2
                    + (dj * voxel_spacing) ** 2
                    + (dk * voxel_spacing) ** 2
                )
                if distance_squared > smoothing_squared:
                    continue
                if di == 0 and dj == 0 and dk == 0:
                    continue

                destination = []
                source = []
                for size, offset in zip(spatial_shape, (di, dj, dk)):
                    if offset < 0:
                        destination.append(slice(-offset, size))
                        source.append(slice(0, max(0, size + offset)))
                    else:
                        destination.append(slice(0, max(0, size - offset)))
                        source.append(slice(offset, size))
                destination_index = (..., *destination)
                source_index = (..., *source)
                torch.maximum(
                    smoothed[destination_index],
                    fa_rep[source_index],
                    out=smoothed[destination_index],
                )
    return smoothed


def _prefilter_cubic_bspline_axis_(values: torch.Tensor, axis: int) -> None:
    """Apply Rosetta's mirror-boundary cubic B-spline recursion on one axis."""
    lines_tensor = values.movedim(axis, -1).contiguous()
    lines = lines_tensor.reshape(-1, lines_tensor.shape[-1])
    line_length = lines.shape[1]
    if line_length == 1:
        return

    pole = _CUBIC_BSPLINE_POLE
    gain = (1.0 - pole) * (1.0 - 1.0 / pole)
    lines.mul_(gain)

    tolerance = torch.finfo(values.dtype).eps
    horizon = min(line_length, math.ceil(math.log(tolerance) / math.log(abs(pole))))
    if horizon < line_length:
        causal = lines[:, 0].clone()
        pole_power = pole
        for index in range(1, horizon):
            causal.add_(lines[:, index], alpha=pole_power)
            pole_power *= pole
    else:
        pole_power = pole
        inverse_pole = 1.0 / pole
        opposite_power = pole ** (line_length - 1)
        causal = lines[:, 0] + opposite_power * lines[:, -1]
        opposite_power *= opposite_power * inverse_pole
        for index in range(1, line_length - 1):
            causal.add_(lines[:, index], alpha=pole_power + opposite_power)
            pole_power *= pole
            opposite_power *= inverse_pole
        causal.div_(1.0 - pole_power * pole_power)
    lines[:, 0] = causal

    for index in range(1, line_length):
        lines[:, index].add_(lines[:, index - 1], alpha=pole)

    lines[:, -1] = (pole / (pole * pole - 1.0)) * (pole * lines[:, -2] + lines[:, -1])
    for index in range(line_length - 2, -1, -1):
        lines[:, index] = pole * (lines[:, index + 1] - lines[:, index])

    values.copy_(lines_tensor.movedim(-1, axis))


def compute_cubic_bspline_coefficients(values: torch.Tensor) -> torch.Tensor:
    """Compute separable cubic B-spline coefficients with mirror boundaries.

    The final three tensor axes are the spatial ``(x, y, z)`` axes. Leading
    axes, such as the grid atom-type axis, index independent grids. This is a
    one-time CPU precomputation and intentionally does not preserve autograd.
    """
    if values.device.type != "cpu":
        raise ValueError("spline coefficients must be computed on CPU")
    if values.ndim < 3:
        raise ValueError("grid values must have at least three dimensions")
    if values.dtype not in (torch.float32, torch.float64):
        raise TypeError("grid values must be float32 or float64")
    if any(size <= 0 for size in values.shape[-3:]):
        raise ValueError("spatial grid dimensions must be positive")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("grid values must be finite")

    # Run the Thevenaz recursion in float64 regardless of input dtype: Rosetta's
    # prefilter always uses DBL_EPSILON for the causal-init horizon, so a float32
    # epsilon would truncate the horizon differently. Cast back at the end.
    # NOTE: .clone() is essential — the recursion is IN-PLACE, and .to(float64) on
    # a float64 input aliases the source, which would corrupt `values`.
    input_dtype = values.dtype
    coefficients = values.detach().clone().to(torch.float64).contiguous()
    for axis in range(coefficients.ndim - 3, coefficients.ndim):
        _prefilter_cubic_bspline_axis_(coefficients, axis)
    return coefficients.to(input_dtype)


@dataclass(frozen=True)
class GridGeometry:
    """Cubic grid geometry with zero-based tensor indices."""

    origin: torch.Tensor
    dims: tuple[int, int, int]
    voxel_spacing: float
    ligand_com: torch.Tensor
    max_radius: float

    def __post_init__(self):
        if self.origin.shape != (3,) or self.ligand_com.shape != (3,):
            raise ValueError("origin and ligand_com must have shape [3]")
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
        if index_tensor.shape != (3,):
            raise ValueError("index must have shape [3]")
        return self.origin + self.voxel_spacing * index_tensor

    def cartesian_from_one_based_index(
        self, index: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Return the Cartesian position for the M2 one-based node convention."""
        index_tensor = torch.as_tensor(
            index, dtype=self.origin.dtype, device=self.origin.device
        )
        return self.cartesian_from_zero_based_index(index_tensor - 1)

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
        typed_shape = (len(self.atom_type_names), *self.geometry.dims)
        shared_shape = self.geometry.dims
        if not isinstance(self.lk_ball_enabled, bool) or not isinstance(
            self.lk_bridge_enabled, bool
        ):
            raise TypeError("LK-ball enabled flags must be bool")
        if self.atom_type_indices.shape != (len(self.atom_type_names),):
            raise ValueError("atom_type_indices must have one entry per grid type")
        for name in ("fa_atr", "fa_rep", "fa_sol"):
            if tuple(getattr(self, name).shape) != typed_shape:
                raise ValueError(f"{name} must have shape {typed_shape}")
        if tuple(self.fa_elec.shape) != shared_shape:
            raise ValueError(f"fa_elec must have shape {shared_shape}")
        for grid_name, coefficient_name in (
            ("fa_atr", "fa_atr_coeffs"),
            ("fa_rep", "fa_rep_coeffs"),
            ("fa_sol", "fa_sol_coeffs"),
            ("fa_elec", "fa_elec_coeffs"),
        ):
            grid = getattr(self, grid_name)
            coefficients = getattr(self, coefficient_name)
            if coefficients is None:
                if grid.device.type != "cpu":
                    raise ValueError(
                        f"{coefficient_name} is required for a non-CPU GridSet"
                    )
                coefficients = compute_cubic_bspline_coefficients(grid)
                object.__setattr__(self, coefficient_name, coefficients)
            if coefficients.shape != grid.shape:
                raise ValueError(
                    f"{coefficient_name} must have shape {tuple(grid.shape)}"
                )
            if coefficients.dtype != grid.dtype or coefficients.device != grid.device:
                raise ValueError(
                    f"{coefficient_name} must match {grid_name} dtype and device"
                )
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
            coefficients = getattr(self, coefficient_name)
            if grid is None:
                if coefficients is not None:
                    raise ValueError(f"{coefficient_name} requires {grid_name}")
                continue
            if tuple(grid.shape) != typed_shape:
                raise ValueError(f"{grid_name} must have shape {typed_shape}")
            if coefficients is None:
                if grid.device.type != "cpu":
                    raise ValueError(
                        f"{coefficient_name} is required for a non-CPU GridSet"
                    )
                coefficients = compute_cubic_bspline_coefficients(grid)
                object.__setattr__(self, coefficient_name, coefficients)
            if coefficients.shape != grid.shape:
                raise ValueError(
                    f"{coefficient_name} must have shape {tuple(grid.shape)}"
                )
            if coefficients.dtype != grid.dtype or coefficients.device != grid.device:
                raise ValueError(
                    f"{coefficient_name} must match {grid_name} dtype and device"
                )

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
        version = state.get("version")
        if version not in (1, 2, _SERIALIZATION_VERSION):
            raise ValueError(f"unsupported grid serialization version: {version}")
        atom_type_names = tuple(state["atom_type_names"])
        expected_map = {name: index for index, name in enumerate(atom_type_names)}
        if state["atom_type_to_grid"] != expected_map:
            raise ValueError("serialized atom_type_to_grid is inconsistent")

        # Version 1 predates all spline coefficients. Version 2 could persist
        # the reserved LK grid fields but had no corresponding coefficients.
        # Migrate either case through CPU before honoring a GPU map location.
        migrate_coefficients = version == 1 or (
            version == 2
            and any(
                state.get(name) is not None
                for name in ("lk_ball", "lk_ball_water", "lk_bridge")
            )
        )

        def loaded_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None or not migrate_coefficients:
                return value
            return value.cpu()

        geometry_state = state["geometry"]
        geometry = GridGeometry.from_dict(
            {
                **geometry_state,
                "origin": loaded_tensor(geometry_state["origin"]),
                "ligand_com": loaded_tensor(geometry_state["ligand_com"]),
            }
        )
        grids = cls(
            geometry=geometry,
            atom_type_names=atom_type_names,
            atom_type_indices=loaded_tensor(state["atom_type_indices"]),
            fa_atr=loaded_tensor(state["fa_atr"]),
            fa_rep=loaded_tensor(state["fa_rep"]),
            fa_sol=loaded_tensor(state["fa_sol"]),
            fa_elec=loaded_tensor(state["fa_elec"]),
            fa_atr_coeffs=loaded_tensor(state.get("fa_atr_coeffs")),
            fa_rep_coeffs=loaded_tensor(state.get("fa_rep_coeffs")),
            fa_sol_coeffs=loaded_tensor(state.get("fa_sol_coeffs")),
            fa_elec_coeffs=loaded_tensor(state.get("fa_elec_coeffs")),
            lk_ball=loaded_tensor(state["lk_ball"]),
            lk_ball_water=loaded_tensor(state["lk_ball_water"]),
            lk_bridge=loaded_tensor(state["lk_bridge"]),
            lk_ball_enabled=bool(state.get("lk_ball_enabled", False)),
            lk_bridge_enabled=bool(state.get("lk_bridge_enabled", False)),
            lk_ball_coeffs=loaded_tensor(state.get("lk_ball_coeffs")),
            lk_ball_water_coeffs=loaded_tensor(state.get("lk_ball_water_coeffs")),
            lk_bridge_coeffs=loaded_tensor(state.get("lk_bridge_coeffs")),
        )
        return grids.to(map_location) if migrate_coefficients else grids

    def to(
        self, device: str | torch.device, dtype: torch.dtype | None = None
    ) -> "GridSet":
        """Return a copy of this grid set on ``device`` for interpolation."""
        device = torch.device(device)
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise TypeError("grid dtype must be float32 or float64")

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
GRID_AND_LIVE_SCORE_TYPES = GRID_SCORE_TYPES + LIVE_SCORE_TYPES


def _apply_live_weights(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Multiply live channels while making zero weights an exact mask."""

    return torch.where(weights != 0, values * weights, torch.zeros_like(values))


class _GridLKBallWaters(torch.autograd.Function):
    """Autograd bridge for ligand virtual-water construction."""

    @staticmethod
    def forward(ctx, ligand_coords, data):
        from .potentials.compiled import build_lk_ball_waters

        ctx.data = data
        ctx.save_for_backward(ligand_coords)
        return build_lk_ball_waters(
            ligand_coords.contiguous(),
            data.ligand_polar_atoms,
            data.ligand_water_kinds,
            data.ligand_water_atoms,
            data.ligand_water_params,
            data.ligand_water_is_ring,
            GRID_LK_BALL_WATER_DISTANCE,
        )

    @staticmethod
    def backward(ctx, grad_waters):
        from .potentials.compiled import propagate_lk_ball_water_gradients

        (ligand_coords,) = ctx.saved_tensors
        data = ctx.data
        gradients = propagate_lk_ball_water_gradients(
            ligand_coords.contiguous(),
            data.ligand_polar_atoms,
            data.ligand_water_kinds,
            data.ligand_water_atoms,
            data.ligand_water_params,
            data.ligand_water_is_ring,
            grad_waters.contiguous(),
            GRID_LK_BALL_WATER_DISTANCE,
        )
        return gradients, None


def _build_grid_lk_ball_waters(
    data: "LiveLKBallData", ligand_coords: torch.Tensor
) -> torch.Tensor:
    if ligand_coords.requires_grad:
        return _GridLKBallWaters.apply(ligand_coords, data)
    from .potentials.compiled import build_lk_ball_waters

    return build_lk_ball_waters(
        ligand_coords.contiguous(),
        data.ligand_polar_atoms,
        data.ligand_water_kinds,
        data.ligand_water_atoms,
        data.ligand_water_params,
        data.ligand_water_is_ring,
        GRID_LK_BALL_WATER_DISTANCE,
    )


def _validate_gridded_lk_ball(
    grids: GridSet, data: "LiveLKBallData", ligand_coords: torch.Tensor
) -> None:
    if any(
        value is None
        for value in (
            grids.lk_ball_coeffs,
            grids.lk_ball_water_coeffs,
            grids.lk_bridge_coeffs,
        )
    ):
        raise ValueError("GridSet does not contain gridded LK-ball coefficients")
    if data.n_ligand_atoms != ligand_coords.shape[1]:
        raise ValueError("ligand atom count does not match LK-ball topology")
    if (
        data.receptor_coords.device != ligand_coords.device
        or data.receptor_coords.dtype != ligand_coords.dtype
    ):
        raise ValueError("LK-ball topology must match coordinate dtype and device")
    for coefficients in (
        grids.lk_ball_coeffs,
        grids.lk_ball_water_coeffs,
        grids.lk_bridge_coeffs,
    ):
        if (
            coefficients.device != ligand_coords.device
            or coefficients.dtype != ligand_coords.dtype
        ):
            raise ValueError(
                "LK-ball coefficients must match coordinate dtype and device"
            )


def _score_gridded_lk_ball(
    grids: GridSet,
    data: "LiveLKBallData",
    ligand_coords: torch.Tensor,
    atom_grid_indices: torch.Tensor,
    coordinate_atom_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Rosetta's single weighted gridded LK-ball total per pose."""

    _validate_gridded_lk_ball(grids, data, ligand_coords)
    n_poses, n_atoms, _ = ligand_coords.shape
    if not grids.lk_ball_enabled:
        return torch.zeros(
            n_poses, dtype=ligand_coords.dtype, device=ligand_coords.device
        )
    if coordinate_atom_indices is None:
        grid_coords = ligand_coords
        n_grid_atoms = n_atoms
    else:
        grid_coords = ligand_coords.index_select(1, coordinate_atom_indices)
        n_grid_atoms = coordinate_atom_indices.shape[0]
    flat_coords = grid_coords.reshape(n_poses * n_grid_atoms, 3)
    atom_indices = atom_grid_indices.repeat(n_poses)
    fractional = grids.geometry.zero_based_index_from_cartesian(flat_coords)
    bounded, _ = _bound_grid_indices(grids.lk_ball_coeffs, fractional)
    heavy = _interpolate_bounded_grid(
        grids.lk_ball_coeffs, bounded, atom_indices
    ).reshape(n_poses, n_grid_atoms)
    total = torch.sum(heavy, dim=1)

    n_polar = data.ligand_polar_atoms.shape[0]
    if n_polar == 0:
        return total
    waters = _build_grid_lk_ball_waters(data, ligand_coords)
    active = data.ligand_water_kinds != 0
    has_water = torch.any(active, dim=1)
    safe_waters = torch.where(
        active[None, :, :, None],
        waters,
        grids.geometry.origin.reshape(1, 1, 1, 3),
    )
    water_fractional = grids.geometry.zero_based_index_from_cartesian(
        safe_waters.reshape(n_poses * n_polar * 4, 3)
    )
    water_bounded, _ = _bound_grid_indices(grids.lk_ball_water_coeffs, water_fractional)
    if coordinate_atom_indices is None:
        polar_grid_indices = atom_grid_indices[data.ligand_polar_atoms]
    else:
        coordinate_to_grid = torch.full(
            (n_atoms,), -1, dtype=torch.int64, device=ligand_coords.device
        )
        coordinate_to_grid[coordinate_atom_indices] = torch.arange(
            n_grid_atoms, dtype=torch.int64, device=ligand_coords.device
        )
        polar_grid_positions = coordinate_to_grid[data.ligand_polar_atoms]
        if bool(torch.any(polar_grid_positions < 0)):
            raise ValueError("LK-ball polar atom is absent from the grid atom set")
        polar_grid_indices = atom_grid_indices[polar_grid_positions]
    water_grid_indices = (
        polar_grid_indices[:, None].expand(n_polar, 4).reshape(-1).repeat(n_poses)
    )
    water_values = _interpolate_bounded_grid(
        grids.lk_ball_water_coeffs, water_bounded, water_grid_indices
    ).reshape(n_poses, n_polar, 4)
    if grids.lk_bridge_enabled:
        bridge_values = _interpolate_bounded_grid(
            grids.lk_bridge_coeffs, water_bounded, water_grid_indices
        ).reshape(n_poses, n_polar, 4)
    else:
        bridge_values = torch.zeros_like(water_values)
    neg_inf = torch.full_like(water_values, -torch.inf)
    pos_inf = torch.full_like(bridge_values, torch.inf)
    water_values = torch.where(active[None], water_values, neg_inf)
    bridge_values = torch.where(active[None], bridge_values, pos_inf)
    water_values = torch.where(
        has_water[None, :, None], water_values, torch.zeros_like(water_values)
    )
    bridge_values = torch.where(
        has_water[None, :, None], bridge_values, torch.zeros_like(bridge_values)
    )
    water_softmax = GRID_LK_BALL_FADE * torch.logsumexp(
        water_values / GRID_LK_BALL_FADE, dim=2
    )
    if grids.lk_bridge_enabled:
        bridge_softmin = -GRID_LK_BALL_FADE * torch.logsumexp(
            -bridge_values / GRID_LK_BALL_FADE, dim=2
        )
    else:
        bridge_softmin = torch.zeros_like(water_softmax)
    return total + torch.sum(
        torch.where(
            has_water[None],
            water_softmax + bridge_softmin,
            torch.zeros_like(water_softmax),
        ),
        dim=1,
    )


def _score_gridded_lk_ball_with_gradients(
    grids: GridSet,
    data: "LiveLKBallData",
    ligand_coords: torch.Tensor,
    atom_grid_indices: torch.Tensor,
    coordinate_atom_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gridded LK-ball total and graph-free Cartesian gradient."""

    _validate_gridded_lk_ball(grids, data, ligand_coords)
    from .potentials.compiled import (
        build_lk_ball_waters,
        interpolate_with_derivatives,
        propagate_lk_ball_water_gradients,
    )

    n_poses, n_atoms, _ = ligand_coords.shape
    if not grids.lk_ball_enabled:
        return (
            torch.zeros(
                n_poses, dtype=ligand_coords.dtype, device=ligand_coords.device
            ),
            torch.zeros_like(ligand_coords),
        )
    if coordinate_atom_indices is None:
        grid_coords = ligand_coords
        n_grid_atoms = n_atoms
    else:
        grid_coords = ligand_coords.index_select(1, coordinate_atom_indices)
        n_grid_atoms = coordinate_atom_indices.shape[0]
    flat_coords = grid_coords.detach().reshape(n_poses * n_grid_atoms, 3)
    atom_indices = atom_grid_indices.repeat(n_poses).contiguous()
    fractional = grids.geometry.zero_based_index_from_cartesian(
        flat_coords
    ).contiguous()
    heavy, dheavy, _, _ = interpolate_with_derivatives(
        grids.lk_ball_coeffs.contiguous(), fractional, atom_indices, 0.0
    )
    energies = torch.sum(heavy.reshape(n_poses, n_grid_atoms), dim=1)
    grid_gradients = (dheavy / grids.geometry.voxel_spacing).reshape(
        n_poses, n_grid_atoms, 3
    )
    if coordinate_atom_indices is None:
        gradients = grid_gradients
    else:
        gradients = torch.zeros_like(ligand_coords)
        gradients[:, coordinate_atom_indices] = grid_gradients

    n_polar = data.ligand_polar_atoms.shape[0]
    if n_polar == 0:
        return energies, gradients
    waters = build_lk_ball_waters(
        ligand_coords.detach().contiguous(),
        data.ligand_polar_atoms,
        data.ligand_water_kinds,
        data.ligand_water_atoms,
        data.ligand_water_params,
        data.ligand_water_is_ring,
        GRID_LK_BALL_WATER_DISTANCE,
    )
    active = data.ligand_water_kinds != 0
    has_water = torch.any(active, dim=1)
    safe_waters = torch.where(
        active[None, :, :, None],
        waters,
        grids.geometry.origin.reshape(1, 1, 1, 3),
    )
    water_fractional = grids.geometry.zero_based_index_from_cartesian(
        safe_waters.reshape(n_poses * n_polar * 4, 3)
    ).contiguous()
    if coordinate_atom_indices is None:
        polar_grid_indices = atom_grid_indices[data.ligand_polar_atoms]
    else:
        coordinate_to_grid = torch.full(
            (n_atoms,), -1, dtype=torch.int64, device=ligand_coords.device
        )
        coordinate_to_grid[coordinate_atom_indices] = torch.arange(
            n_grid_atoms, dtype=torch.int64, device=ligand_coords.device
        )
        polar_grid_positions = coordinate_to_grid[data.ligand_polar_atoms]
        if bool(torch.any(polar_grid_positions < 0)):
            raise ValueError("LK-ball polar atom is absent from the grid atom set")
        polar_grid_indices = atom_grid_indices[polar_grid_positions]
    water_grid_indices = (
        polar_grid_indices[:, None]
        .expand(n_polar, 4)
        .reshape(-1)
        .repeat(n_poses)
        .contiguous()
    )
    water_values, dwater, _, _ = interpolate_with_derivatives(
        grids.lk_ball_water_coeffs.contiguous(),
        water_fractional,
        water_grid_indices,
        0.0,
    )
    if grids.lk_bridge_enabled:
        bridge_values, dbridge, _, _ = interpolate_with_derivatives(
            grids.lk_bridge_coeffs.contiguous(),
            water_fractional,
            water_grid_indices,
            0.0,
        )
    else:
        bridge_values = torch.zeros_like(water_values)
        dbridge = torch.zeros_like(dwater)
    water_values = water_values.reshape(n_poses, n_polar, 4)
    bridge_values = bridge_values.reshape(n_poses, n_polar, 4)
    dwater = dwater.reshape(n_poses, n_polar, 4, 3)
    dbridge = dbridge.reshape(n_poses, n_polar, 4, 3)
    water_masked = torch.where(
        active[None], water_values, torch.full_like(water_values, -torch.inf)
    )
    bridge_masked = torch.where(
        active[None], bridge_values, torch.full_like(bridge_values, torch.inf)
    )
    water_masked = torch.where(
        has_water[None, :, None], water_masked, torch.zeros_like(water_masked)
    )
    bridge_masked = torch.where(
        has_water[None, :, None], bridge_masked, torch.zeros_like(bridge_masked)
    )
    water_energy = GRID_LK_BALL_FADE * torch.logsumexp(
        water_masked / GRID_LK_BALL_FADE, dim=2
    )
    if grids.lk_bridge_enabled:
        bridge_energy = -GRID_LK_BALL_FADE * torch.logsumexp(
            -bridge_masked / GRID_LK_BALL_FADE, dim=2
        )
    else:
        bridge_energy = torch.zeros_like(water_energy)
    energies = energies + torch.sum(
        torch.where(
            has_water[None],
            water_energy + bridge_energy,
            torch.zeros_like(water_energy),
        ),
        dim=1,
    )
    active_real = active.to(dtype=water_values.dtype)[None]
    water_weights = torch.softmax(water_masked / GRID_LK_BALL_FADE, dim=2) * active_real
    if grids.lk_bridge_enabled:
        bridge_weights = (
            torch.softmax(-bridge_masked / GRID_LK_BALL_FADE, dim=2) * active_real
        )
    else:
        bridge_weights = torch.zeros_like(bridge_values)
    water_coordinate_gradients = (
        water_weights[..., None] * dwater + bridge_weights[..., None] * dbridge
    ) / grids.geometry.voxel_spacing
    gradients = gradients + propagate_lk_ball_water_gradients(
        ligand_coords.detach().contiguous(),
        data.ligand_polar_atoms,
        data.ligand_water_kinds,
        data.ligand_water_atoms,
        data.ligand_water_params,
        data.ligand_water_is_ring,
        water_coordinate_gradients.contiguous(),
        GRID_LK_BALL_WATER_DISTANCE,
    )
    return energies, gradients


def interpolate_grid(
    coefficients: torch.Tensor,
    indices: torch.Tensor,
    grid_indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate boundary-clamped tricubic B-splines at zero-based indices.

    The receptor coefficients are fixed. Gradients are provided for ``indices``
    by the compiled analytic spatial derivative; coefficient gradients are not
    part of the rigid-receptor grid contract.
    """
    if coefficients.ndim != 4:
        raise ValueError("coefficients must have shape [n_grids, nx, ny, nz]")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape [n_points, 3]")
    if grid_indices.ndim != 1 or grid_indices.shape[0] != indices.shape[0]:
        raise ValueError("grid_indices must have shape [n_points]")
    if coefficients.dtype not in (torch.float32, torch.float64):
        raise TypeError("coefficients must be float32 or float64")
    if coefficients.requires_grad:
        raise NotImplementedError(
            "rigid receptor grid coefficient gradients are not supported"
        )
    if indices.dtype != coefficients.dtype:
        raise TypeError("indices and coefficients must have the same dtype")
    if grid_indices.dtype != torch.int64:
        raise TypeError("grid_indices must be int64")
    if not (coefficients.device == indices.device == grid_indices.device):
        raise ValueError("interpolation tensors must be on the same device")

    return _interpolate_grid_and_penalty(coefficients, indices, grid_indices)[0]


class _GridInterpolation(torch.autograd.Function):
    """Autograd bridge for the compiled value and spatial-derivative kernel."""

    @staticmethod
    def forward(ctx, coefficients, indices, grid_indices):
        from .potentials.compiled import interpolate_with_derivatives

        values, derivatives, penalties, penalty_derivatives = (
            interpolate_with_derivatives(
                coefficients.contiguous(),
                indices.contiguous(),
                grid_indices.contiguous(),
                OUT_OF_BOUNDS_PENALTY,
            )
        )
        ctx.save_for_backward(derivatives, penalty_derivatives)
        return values, penalties

    @staticmethod
    def backward(ctx, grad_values, grad_penalties):
        derivatives, penalty_derivatives = ctx.saved_tensors
        grad_indices = None
        if ctx.needs_input_grad[1]:
            grad_indices = torch.zeros_like(derivatives)
            if grad_values is not None:
                grad_indices.add_(derivatives * grad_values[:, None])
            if grad_penalties is not None:
                grad_indices.add_(penalty_derivatives * grad_penalties[:, None])
        return None, grad_indices, None


def _interpolate_grid_and_penalty(
    coefficients: torch.Tensor,
    indices: torch.Tensor,
    grid_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clamped spline values and per-point quadratic OOB penalties."""

    if indices.requires_grad:
        return _GridInterpolation.apply(coefficients, indices, grid_indices)

    # Keep inference/scoring on the cheaper value-only compiled path. This is
    # algebraically identical to the clamp and penalty in the derivative op.
    valid = torch.all(
        torch.isfinite(indices) & (torch.abs(indices) <= float(1 << 30)),
        dim=1,
    )
    safe_indices = torch.where(valid[:, None], indices, torch.zeros_like(indices))
    upper = indices.new_tensor(coefficients.shape[1:]) - 1
    bounded_indices = torch.minimum(
        torch.maximum(safe_indices, torch.zeros_like(indices)), upper
    )
    overshoot = safe_indices - bounded_indices
    penalties = OUT_OF_BOUNDS_PENALTY * torch.sum(overshoot * overshoot, dim=1)
    from .potentials.compiled import interpolate as interpolate_op

    values = interpolate_op(
        coefficients.contiguous(),
        bounded_indices.contiguous(),
        grid_indices.contiguous(),
    )
    nan = torch.full_like(values, float("nan"))
    values = torch.where(valid, values, nan)
    penalties = torch.where(valid, penalties, nan)
    return values, penalties


def _bound_grid_indices(
    coefficients: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp once for multi-channel scoring and return the OOB penalty."""

    upper = indices.new_tensor(coefficients.shape[1:]) - 1
    bounded = torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), upper)
    overshoot = indices - bounded
    penalty = OUT_OF_BOUNDS_PENALTY * torch.sum(overshoot * overshoot, dim=1)
    return bounded, penalty


def _interpolate_bounded_grid(
    coefficients: torch.Tensor,
    bounded_indices: torch.Tensor,
    grid_indices: torch.Tensor,
) -> torch.Tensor:
    """Interpolate already-bounded indices without repeating boundary work."""

    if bounded_indices.requires_grad:
        return _GridInterpolation.apply(coefficients, bounded_indices, grid_indices)[0]
    from .potentials.compiled import interpolate as interpolate_op

    return interpolate_op(
        coefficients.contiguous(),
        bounded_indices.contiguous(),
        grid_indices.contiguous(),
    )


def score_ligand_grid(
    grids: GridSet,
    ligand_coords: torch.Tensor,
    ligand_atom_types: Sequence[str],
    ligand_charges: torch.Tensor,
    *,
    score_function: "ScoreFunction | None" = None,
    live_hbond: "LiveHBondData | None" = None,
    live_lk_ball: "LiveLKBallData | None" = None,
    lk_ball_grid: bool = False,
) -> torch.Tensor:
    """Return weighted grid and optional live ligand-interface energies.

    ``ligand_atom_types`` contains chemical atom-type names in the same order as
    ``ligand_coords``. Live hbond and LK-ball channels are appended when their
    corresponding shared receptor structures are supplied.
    The channel-only API clamps OOB coordinates but does not assign the
    boundary penalty to a score type; the batched total scorer includes it.
    """
    if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3:
        raise ValueError("ligand_coords must have shape [n_atoms, 3]")
    if ligand_coords.shape[0] == 0:
        raise ValueError("at least one ligand atom is required")
    if ligand_charges.ndim != 1 or ligand_charges.shape[0] != ligand_coords.shape[0]:
        raise ValueError("ligand_charges must have shape [n_atoms]")
    if len(ligand_atom_types) != ligand_coords.shape[0]:
        raise ValueError("ligand_atom_types must have one entry per atom")
    if ligand_coords.dtype not in (torch.float32, torch.float64):
        raise TypeError("ligand_coords must be float32 or float64")
    if ligand_charges.dtype != ligand_coords.dtype:
        raise TypeError("ligand charges and coordinates must have the same dtype")
    if ligand_charges.device != ligand_coords.device:
        raise ValueError("ligand charges and coordinates must be on the same device")
    if (
        grids.geometry.origin.device != ligand_coords.device
        or grids.geometry.origin.dtype != ligand_coords.dtype
    ):
        raise ValueError("grid geometry must match ligand coordinate dtype and device")

    type_to_grid = grids.atom_type_to_grid
    unknown_types = sorted(set(ligand_atom_types) - set(type_to_grid))
    if unknown_types:
        raise ValueError(f"ligand atom types are absent from the grid: {unknown_types}")
    grid_indices = torch.tensor(
        [type_to_grid[name] for name in ligand_atom_types],
        dtype=torch.int64,
        device=ligand_coords.device,
    )
    zero_grid_indices = torch.zeros_like(grid_indices)

    coefficients = (
        grids.fa_atr_coeffs,
        grids.fa_rep_coeffs,
        grids.fa_sol_coeffs,
        grids.fa_elec_coeffs,
    )
    if any(value is None for value in coefficients):
        raise ValueError("GridSet does not contain spline coefficients")
    for value in coefficients:
        if value.device != ligand_coords.device or value.dtype != ligand_coords.dtype:
            raise ValueError(
                "grid coefficients must match ligand coordinate dtype and device"
            )

    fractional_indices = grids.geometry.zero_based_index_from_cartesian(ligand_coords)
    bounded_indices, _ = _bound_grid_indices(coefficients[0], fractional_indices)
    atr = _interpolate_bounded_grid(coefficients[0], bounded_indices, grid_indices)
    rep = ixform_rep(
        _interpolate_bounded_grid(coefficients[1], bounded_indices, grid_indices)
    )
    sol = _interpolate_bounded_grid(coefficients[2], bounded_indices, grid_indices)
    elec_coefficients = coefficients[3].unsqueeze(0)
    elec = ligand_charges * _interpolate_bounded_grid(
        elec_coefficients, bounded_indices, zero_grid_indices
    )
    unweighted_terms = torch.stack((atr, rep, sol, elec), dim=1)

    if score_function is None:
        from tmol.score import beta2016_score_function

        score_function = beta2016_score_function(ligand_coords.device)
    weights = torch.stack(
        [score_function.get_weight(score_type) for score_type in GRID_SCORE_TYPES]
    ).to(device=ligand_coords.device, dtype=ligand_coords.dtype)
    weighted_terms = torch.sum(unweighted_terms * weights, dim=0)
    live_terms = [weighted_terms]
    if live_hbond is not None:
        from .live_hbond import score_ligand_hbond

        hbond = score_ligand_hbond(live_hbond, ligand_coords.unsqueeze(0))[0]
        hbond_weight = score_function.get_weight(ScoreType.hbond).to(
            device=ligand_coords.device, dtype=ligand_coords.dtype
        )
        live_terms.append((hbond * hbond_weight).unsqueeze(0))
    if lk_ball_grid:
        if live_lk_ball is None:
            raise ValueError("gridded LK-ball requires ligand water topology")
        gridded_lk_ball = _score_gridded_lk_ball(
            grids,
            live_lk_ball,
            ligand_coords.unsqueeze(0),
            grid_indices,
        )[0]
        live_terms.append(gridded_lk_ball.unsqueeze(0))
    elif live_lk_ball is not None:
        from .live_lk_ball import score_ligand_lk_ball

        lk_ball = score_ligand_lk_ball(live_lk_ball, ligand_coords.unsqueeze(0))[0]
        lk_ball_weights = torch.stack(
            [
                score_function.get_weight(score_type)
                for score_type in LK_BALL_SCORE_TYPES
            ]
        ).to(device=ligand_coords.device, dtype=ligand_coords.dtype)
        live_terms.append(_apply_live_weights(lk_ball, lk_ball_weights))
    return torch.cat(live_terms)


def score_ligand_grid_batched(
    grids: GridSet,
    ligand_coords: torch.Tensor,
    ligand_atom_types: Sequence[str],
    ligand_charges: torch.Tensor,
    *,
    score_function: "ScoreFunction | None" = None,
    live_hbond: "LiveHBondData | None" = None,
    live_ligand_intra: "LiveLigandIntraData | None" = None,
    live_lk_ball: "LiveLKBallData | None" = None,
    lk_ball_grid: bool = False,
    grid_atom_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score many ligand poses against one shared receptor grid.

    ``ligand_coords`` has shape ``[n_poses, n_atoms, 3]``. Atom types and
    charges are shared by every pose. The result has shape ``[n_poses]`` and
    contains the atom-summed, weighted grid total plus live hbond, weighted
    ligand-intramolecular energy, and either the single gridded LK-ball total
    or the exact four-term live LK-ball total.

    It flattens all pose points before making one interpolation call per score
    channel; it neither copies nor rebuilds grid coefficients per pose. The
    returned total includes the unweighted Rosetta OOB penalty once per atom.
    """
    if ligand_coords.ndim != 3 or ligand_coords.shape[2] != 3:
        raise ValueError("ligand_coords must have shape [n_poses, n_atoms, 3]")
    n_poses, n_atoms, _ = ligand_coords.shape
    if n_poses == 0:
        raise ValueError("at least one ligand pose is required")
    if n_atoms == 0:
        raise ValueError("at least one ligand atom is required")
    if grid_atom_indices is None:
        grid_coords = ligand_coords
        n_grid_atoms = n_atoms
    else:
        if (
            grid_atom_indices.ndim != 1
            or grid_atom_indices.dtype != torch.int64
            or grid_atom_indices.device != ligand_coords.device
        ):
            raise ValueError(
                "grid_atom_indices must be a device-local one-dimensional int64 tensor"
            )
        if grid_atom_indices.numel() == 0:
            raise ValueError("at least one grid-scored atom is required")
        if bool(
            torch.any(grid_atom_indices < 0)
            or torch.any(grid_atom_indices >= n_atoms)
            or torch.unique(grid_atom_indices).numel() != grid_atom_indices.numel()
        ):
            raise ValueError("grid_atom_indices must contain unique in-range indices")
        grid_coords = ligand_coords.index_select(1, grid_atom_indices)
        n_grid_atoms = grid_atom_indices.shape[0]
    if ligand_charges.ndim != 1 or ligand_charges.shape[0] != n_grid_atoms:
        raise ValueError("ligand_charges must have shape [n_grid_atoms]")
    if len(ligand_atom_types) != n_grid_atoms:
        raise ValueError("ligand_atom_types must have one entry per grid-scored atom")
    if ligand_coords.dtype not in (torch.float32, torch.float64):
        raise TypeError("ligand_coords must be float32 or float64")
    if ligand_charges.dtype != ligand_coords.dtype:
        raise TypeError("ligand charges and coordinates must have the same dtype")
    if ligand_charges.device != ligand_coords.device:
        raise ValueError("ligand charges and coordinates must be on the same device")
    if (
        grids.geometry.origin.device != ligand_coords.device
        or grids.geometry.origin.dtype != ligand_coords.dtype
    ):
        raise ValueError("grid geometry must match ligand coordinate dtype and device")

    type_to_grid = grids.atom_type_to_grid
    unknown_types = sorted(set(ligand_atom_types) - set(type_to_grid))
    if unknown_types:
        raise ValueError(f"ligand atom types are absent from the grid: {unknown_types}")
    atom_grid_indices = torch.tensor(
        [type_to_grid[name] for name in ligand_atom_types],
        dtype=torch.int64,
        device=ligand_coords.device,
    )
    grid_indices = atom_grid_indices.repeat(n_poses)
    zero_grid_indices = torch.zeros_like(grid_indices)

    coefficients = (
        grids.fa_atr_coeffs,
        grids.fa_rep_coeffs,
        grids.fa_sol_coeffs,
        grids.fa_elec_coeffs,
    )
    if any(value is None for value in coefficients):
        raise ValueError("GridSet does not contain spline coefficients")
    for value in coefficients:
        if value.device != ligand_coords.device or value.dtype != ligand_coords.dtype:
            raise ValueError(
                "grid coefficients must match ligand coordinate dtype and device"
            )

    flat_coords = grid_coords.reshape(n_poses * n_grid_atoms, 3)
    fractional_indices = grids.geometry.zero_based_index_from_cartesian(flat_coords)
    bounded_indices, penalties = _bound_grid_indices(
        coefficients[0], fractional_indices
    )
    atr = _interpolate_bounded_grid(coefficients[0], bounded_indices, grid_indices)
    rep = ixform_rep(
        _interpolate_bounded_grid(coefficients[1], bounded_indices, grid_indices)
    )
    sol = _interpolate_bounded_grid(coefficients[2], bounded_indices, grid_indices)
    elec = ligand_charges.repeat(n_poses) * _interpolate_bounded_grid(
        coefficients[3].unsqueeze(0), bounded_indices, zero_grid_indices
    )
    unweighted_terms = torch.stack((atr, rep, sol, elec), dim=1).reshape(
        n_poses, n_grid_atoms, len(GRID_SCORE_TYPES)
    )

    if score_function is None:
        from tmol.score import beta2016_score_function

        score_function = beta2016_score_function(ligand_coords.device)
    weights = torch.stack(
        [score_function.get_weight(score_type) for score_type in GRID_SCORE_TYPES]
    ).to(device=ligand_coords.device, dtype=ligand_coords.dtype)
    weighted_per_pose_per_channel = torch.sum(unweighted_terms * weights, dim=1)
    totals = torch.sum(weighted_per_pose_per_channel, dim=1) + torch.sum(
        penalties.reshape(n_poses, n_grid_atoms), dim=1
    )
    if live_hbond is not None:
        from .live_hbond import score_ligand_hbond

        hbond_weight = score_function.get_weight(ScoreType.hbond).to(
            device=ligand_coords.device, dtype=ligand_coords.dtype
        )
        totals = totals + hbond_weight * score_ligand_hbond(live_hbond, ligand_coords)
    if lk_ball_grid:
        if live_lk_ball is None:
            raise ValueError("gridded LK-ball requires ligand water topology")
        totals = totals + _score_gridded_lk_ball(
            grids,
            live_lk_ball,
            ligand_coords,
            atom_grid_indices,
            grid_atom_indices,
        )
    elif live_lk_ball is not None:
        from .live_lk_ball import score_ligand_lk_ball

        lk_ball_weights = torch.stack(
            [
                score_function.get_weight(score_type)
                for score_type in LK_BALL_SCORE_TYPES
            ]
        ).to(device=ligand_coords.device, dtype=ligand_coords.dtype)
        totals = totals + torch.sum(
            _apply_live_weights(
                score_ligand_lk_ball(live_lk_ball, ligand_coords),
                lk_ball_weights,
            ),
            dim=1,
        )
    if live_ligand_intra is not None:
        from .live_ligand_intra import score_ligand_intra

        intra_energies = score_ligand_intra(live_ligand_intra, ligand_coords).to(
            device=ligand_coords.device, dtype=ligand_coords.dtype
        )
        totals = totals + intra_energies
    return totals


def score_ligand_grid_batched_with_gradients(
    grids: GridSet,
    ligand_coords: torch.Tensor,
    ligand_atom_types: Sequence[str],
    ligand_charges: torch.Tensor,
    *,
    score_function: "ScoreFunction | None" = None,
    live_hbond: "LiveHBondData | None" = None,
    live_ligand_intra: "LiveLigandIntraData | None" = None,
    live_lk_ball: "LiveLKBallData | None" = None,
    lk_ball_grid: bool = False,
    grid_atom_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-pose grid energies and analytic Cartesian gradients.

    This is the graph-free counterpart of :func:`score_ligand_grid_batched`.
    It calls the compiled M2e value-and-spatial-derivative kernel directly and
    applies the four all-atom grid-term weights, the inverse fa_rep transform
    derivative, the electrostatic charge multiplier, the out-of-bounds
    derivative, the grid-index-to-Cartesian scale, and the optional live hbond
    and selected LK-ball derivatives explicitly. Ligand-intra derivatives use
    a temporary local autograd graph over compiled term backward kernels. No
    input or returned tensor remains attached to an autograd graph.
    """
    if ligand_coords.ndim != 3 or ligand_coords.shape[2] != 3:
        raise ValueError("ligand_coords must have shape [n_poses, n_atoms, 3]")
    n_poses, n_atoms, _ = ligand_coords.shape
    if n_poses == 0:
        raise ValueError("at least one ligand pose is required")
    if n_atoms == 0:
        raise ValueError("at least one ligand atom is required")
    if grid_atom_indices is None:
        grid_coords = ligand_coords
        n_grid_atoms = n_atoms
    else:
        if (
            grid_atom_indices.ndim != 1
            or grid_atom_indices.dtype != torch.int64
            or grid_atom_indices.device != ligand_coords.device
        ):
            raise ValueError(
                "grid_atom_indices must be a device-local one-dimensional int64 tensor"
            )
        if grid_atom_indices.numel() == 0:
            raise ValueError("at least one grid-scored atom is required")
        if bool(
            torch.any(grid_atom_indices < 0)
            or torch.any(grid_atom_indices >= n_atoms)
            or torch.unique(grid_atom_indices).numel() != grid_atom_indices.numel()
        ):
            raise ValueError("grid_atom_indices must contain unique in-range indices")
        grid_coords = ligand_coords.index_select(1, grid_atom_indices)
        n_grid_atoms = grid_atom_indices.shape[0]
    if ligand_charges.ndim != 1 or ligand_charges.shape[0] != n_grid_atoms:
        raise ValueError("ligand_charges must have shape [n_grid_atoms]")
    if len(ligand_atom_types) != n_grid_atoms:
        raise ValueError("ligand_atom_types must have one entry per grid-scored atom")
    if ligand_coords.dtype not in (torch.float32, torch.float64):
        raise TypeError("ligand_coords must be float32 or float64")
    if ligand_charges.dtype != ligand_coords.dtype:
        raise TypeError("ligand charges and coordinates must have the same dtype")
    if ligand_charges.device != ligand_coords.device:
        raise ValueError("ligand charges and coordinates must be on the same device")
    if (
        grids.geometry.origin.device != ligand_coords.device
        or grids.geometry.origin.dtype != ligand_coords.dtype
    ):
        raise ValueError("grid geometry must match ligand coordinate dtype and device")

    type_to_grid = grids.atom_type_to_grid
    unknown_types = sorted(set(ligand_atom_types) - set(type_to_grid))
    if unknown_types:
        raise ValueError(f"ligand atom types are absent from the grid: {unknown_types}")
    atom_grid_indices = torch.tensor(
        [type_to_grid[name] for name in ligand_atom_types],
        dtype=torch.int64,
        device=ligand_coords.device,
    )
    grid_indices = atom_grid_indices.repeat(n_poses).contiguous()
    zero_grid_indices = torch.zeros_like(grid_indices)

    coefficients = (
        grids.fa_atr_coeffs,
        grids.fa_rep_coeffs,
        grids.fa_sol_coeffs,
        grids.fa_elec_coeffs,
    )
    if any(value is None for value in coefficients):
        raise ValueError("GridSet does not contain spline coefficients")
    for value in coefficients:
        if value.device != ligand_coords.device or value.dtype != ligand_coords.dtype:
            raise ValueError(
                "grid coefficients must match ligand coordinate dtype and device"
            )

    if score_function is None:
        from tmol.score import beta2016_score_function

        score_function = beta2016_score_function(ligand_coords.device)
    weights = torch.stack(
        [score_function.get_weight(score_type) for score_type in GRID_SCORE_TYPES]
    ).to(device=ligand_coords.device, dtype=ligand_coords.dtype)

    from .potentials.compiled import interpolate_with_derivatives

    with torch.no_grad():
        flat_coords = grid_coords.detach().reshape(n_poses * n_grid_atoms, 3)
        fractional_indices = grids.geometry.zero_based_index_from_cartesian(
            flat_coords
        ).contiguous()
        atr, datr, penalties, dpenalty = interpolate_with_derivatives(
            coefficients[0].contiguous(),
            fractional_indices,
            grid_indices,
            OUT_OF_BOUNDS_PENALTY,
        )
        rep_xformed, drep_xformed, _, _ = interpolate_with_derivatives(
            coefficients[1].contiguous(),
            fractional_indices,
            grid_indices,
            0.0,
        )
        sol, dsol, _, _ = interpolate_with_derivatives(
            coefficients[2].contiguous(),
            fractional_indices,
            grid_indices,
            0.0,
        )
        elec_grid, delec_grid, _, _ = interpolate_with_derivatives(
            coefficients[3].unsqueeze(0).contiguous(),
            fractional_indices,
            zero_grid_indices,
            0.0,
        )

        repeated_charges = ligand_charges.repeat(n_poses)
        rep = ixform_rep(rep_xformed)
        drep = drep_xformed * (-4.0 * torch.pow(rep_xformed, -5))[:, None]
        elec = repeated_charges * elec_grid
        delec = repeated_charges[:, None] * delec_grid

        unweighted = torch.stack((atr, rep, sol, elec), dim=1)
        energies = torch.sum(
            torch.sum(unweighted * weights, dim=1).reshape(
                n_poses, n_grid_atoms
            ),
            dim=1,
        ) + torch.sum(penalties.reshape(n_poses, n_grid_atoms), dim=1)

        derivative_by_channel = torch.stack((datr, drep, dsol, delec), dim=1)
        index_gradients = (
            torch.sum(derivative_by_channel * weights[None, :, None], dim=1) + dpenalty
        )
        grid_gradients = (index_gradients / grids.geometry.voxel_spacing).reshape(
            n_poses, n_grid_atoms, 3
        )
        if grid_atom_indices is None:
            coordinate_gradients = grid_gradients
        else:
            coordinate_gradients = torch.zeros_like(ligand_coords)
            coordinate_gradients[:, grid_atom_indices] = grid_gradients

        if live_hbond is not None:
            from .live_hbond import score_ligand_hbond_with_gradients

            hbond_energies, hbond_gradients = score_ligand_hbond_with_gradients(
                live_hbond, ligand_coords
            )
            hbond_weight = score_function.get_weight(ScoreType.hbond).to(
                device=ligand_coords.device, dtype=ligand_coords.dtype
            )
            energies = energies + hbond_weight * hbond_energies
            coordinate_gradients = coordinate_gradients + hbond_weight * hbond_gradients

        if lk_ball_grid:
            if live_lk_ball is None:
                raise ValueError("gridded LK-ball requires ligand water topology")
            lk_ball_energies, lk_ball_gradients = _score_gridded_lk_ball_with_gradients(
                grids,
                live_lk_ball,
                ligand_coords,
                atom_grid_indices,
                grid_atom_indices,
            )
            energies = energies + lk_ball_energies
            coordinate_gradients = coordinate_gradients + lk_ball_gradients
        elif live_lk_ball is not None:
            from .live_lk_ball import score_ligand_lk_ball_with_gradients

            lk_ball_energies, lk_ball_gradients = score_ligand_lk_ball_with_gradients(
                live_lk_ball, ligand_coords
            )
            lk_ball_weights = torch.stack(
                [
                    score_function.get_weight(score_type)
                    for score_type in LK_BALL_SCORE_TYPES
                ]
            ).to(device=ligand_coords.device, dtype=ligand_coords.dtype)
            energies = energies + torch.sum(
                _apply_live_weights(lk_ball_energies, lk_ball_weights), dim=1
            )
            coordinate_gradients = coordinate_gradients + torch.sum(
                _apply_live_weights(
                    lk_ball_gradients,
                    lk_ball_weights[None, :, None, None],
                ),
                dim=1,
            )

        if live_ligand_intra is not None:
            from .live_ligand_intra import score_ligand_intra_with_gradients

            intra_energies, intra_gradients = score_ligand_intra_with_gradients(
                live_ligand_intra, ligand_coords
            )
            energies = energies + intra_energies.to(
                device=ligand_coords.device, dtype=ligand_coords.dtype
            )
            coordinate_gradients = coordinate_gradients + intra_gradients.to(
                device=ligand_coords.device, dtype=ligand_coords.dtype
            )

    return energies, coordinate_gradients


def compute_grid_geometry(
    ligand_coords: torch.Tensor,
    voxel_spacing: float = DEFAULT_VOXEL_SPACING,
    bbox_padding: float = DEFAULT_BBOX_PADDING,
) -> GridGeometry:
    """Compute the exact M2 ligand-centered cubic grid geometry."""
    if ligand_coords.device.type != "cpu":
        raise ValueError("grid geometry must be computed on CPU")
    if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3:
        raise ValueError("ligand_coords must have shape [n_atoms, 3]")
    if ligand_coords.shape[0] == 0:
        raise ValueError("at least one ligand atom is required")
    if not ligand_coords.dtype.is_floating_point:
        raise TypeError("ligand_coords must have a floating-point dtype")
    if (
        not math.isfinite(voxel_spacing)
        or not math.isfinite(bbox_padding)
        or voxel_spacing <= 0
        or bbox_padding <= 0
    ):
        raise ValueError("voxel_spacing and bbox_padding must be finite and positive")

    ligand_coords = ligand_coords.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(ligand_coords).all()):
        raise ValueError("ligand coordinates must be finite")
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


def build_grid(
    pose_stack: PoseStack,
    probe_atom_types: Sequence[str],
    param_db: ParameterDatabase,
    *,
    ligand_name3: str = "LG1",
    voxel_spacing: float = DEFAULT_VOXEL_SPACING,
    bbox_padding: float = DEFAULT_BBOX_PADDING,
    score_function: "ScoreFunction | None" = None,
    build_lk_ball: bool = False,
    excluded_receptor_blocks: tuple[int, ...] = (),
    fa_rep_smoothing: float = 0.0,
) -> GridSet:
    """Build beta16 all-atom grids and optional Rosetta-style LK-ball grids.

    Probe atom-type names are deduplicated and sorted, making the grid-to-type
    mapping deterministic even when the caller passes a set.
    ``fa_rep_smoothing`` is an opt-in radius in Angstroms (0.0 disables it);
    Rosetta's reference radius is 0.375 A.
    """
    if pose_stack.n_poses != 1:
        raise ValueError("M2b grid construction supports exactly one pose")
    if pose_stack.device.type != "cpu" or pose_stack.coords.device.type != "cpu":
        raise ValueError("M2b grid construction is CPU-only")
    if pose_stack.coords.dtype not in (torch.float32, torch.float64):
        raise TypeError("pose coordinates must be float32 or float64")
    if not isinstance(build_lk_ball, bool):
        raise TypeError("build_lk_ball must be bool")
    if not math.isfinite(fa_rep_smoothing) or fa_rep_smoothing < 0:
        raise ValueError("fa_rep_smoothing must be finite and nonnegative")
    if any(
        not isinstance(block, int) or isinstance(block, bool)
        for block in excluded_receptor_blocks
    ):
        raise TypeError("excluded receptor block indices must be integers")
    if len(set(excluded_receptor_blocks)) != len(excluded_receptor_blocks):
        raise ValueError("excluded_receptor_blocks must not contain duplicates")
    if any(
        block < 0 or block >= pose_stack.max_n_blocks
        for block in excluded_receptor_blocks
    ):
        raise ValueError("excluded receptor block is out of range")

    if isinstance(probe_atom_types, str):
        raise TypeError("probe_atom_types must be a sequence of atom-type names")
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
        from .live_lk_ball import _generate_reference_waters

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
    if not bool(torch.isfinite(ligand_coords).all()):
        raise ValueError("ligand coordinates must be finite")
    if not bool(torch.isfinite(receptor_coords).all()):
        raise ValueError("receptor coordinates must be finite")
    if not bool(torch.isfinite(receptor_charges).all()):
        raise ValueError("receptor partial charges must be finite")
    geometry = compute_grid_geometry(ligand_coords, voxel_spacing, bbox_padding)
    ljlk_type_params, ljlk_global_params, elec_global_params = _pack_parameter_tensors(
        ljlk_term, elec_term
    )

    from .potentials.compiled import build_grid as build_grid_op

    dims = torch.tensor(geometry.dims, dtype=torch.int64)
    grid_args = (
        receptor_coords,
        receptor_types,
        receptor_charges,
        probe_atom_type_indices,
        geometry.origin.contiguous(),
        dims,
        geometry.voxel_spacing,
        ljlk_type_params,
        ljlk_global_params,
        elec_global_params,
    )
    if build_lk_ball:
        if score_function is None:
            from tmol.score import beta2016_score_function

            score_function = beta2016_score_function(
                torch.device("cpu"), param_db=param_db
            )
        lk_ball_weights = (
            torch.stack(
                [
                    score_function.get_weight(score_type)
                    for score_type in LK_BALL_SCORE_TYPES
                ]
            )
            .to(dtype=torch.float64, device="cpu")
            .contiguous()
        )
        lk_ball_enabled = bool(torch.any(lk_ball_weights != 0))
        lk_bridge_enabled = bool(torch.any(lk_ball_weights[2:] != 0))
        from .potentials.compiled import build_grid_lk_ball

        (
            fa_atr,
            fa_rep,
            fa_sol,
            fa_elec,
            lk_ball,
            lk_ball_water,
            lk_bridge,
        ) = build_grid_lk_ball(*grid_args, receptor_waters, lk_ball_weights)
    else:
        fa_atr, fa_rep, fa_sol, fa_elec = build_grid_op(*grid_args)
        lk_ball = lk_ball_water = lk_bridge = None
        lk_ball_enabled = lk_bridge_enabled = False
    fa_rep = _smooth_fa_rep_grid(fa_rep, fa_rep_smoothing, geometry.voxel_spacing)
    return GridSet(
        geometry=geometry,
        atom_type_names=atom_type_names,
        atom_type_indices=probe_atom_type_indices,
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

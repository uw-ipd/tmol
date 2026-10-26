"""Scoring of ligand pose batches against a rigid receptor.

The receptor enters through precomputed B-spline grids (``fa_atr``, ``fa_rep``,
``fa_sol``, ``fa_elec`` and, optionally, Rosetta's gridded LK-ball) plus live
hbond / LK-ball / ligand-intramolecular terms. :class:`GridScorer` builds every
per-ligand tensor once, so an evaluation allocates only its outputs and has no
host synchronisation or host-to-device copy: it can be captured in a CUDA graph.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tmol.score import ScoreFunction, ScoreType, beta2016_score_function

from .grid import (
    GRID_LK_BALL_FADE,
    GRID_LK_BALL_WATER_DISTANCE,
    GRID_SCORE_TYPES,
    LK_BALL_SCORE_TYPES,
    OUT_OF_BOUNDS_PENALTY,
    GridSet,
    ixform_rep,
)
from .live_hbond import LiveHBondData, score_ligand_hbond_with_gradients
from .live_ligand_intra import LiveLigandIntraData, score_ligand_intra_with_gradients
from .live_lk_ball import LiveLKBallData, score_ligand_lk_ball_with_gradients
from .potentials.compiled import (
    build_lk_ball_waters,
    interpolate_with_derivatives,
    propagate_lk_ball_water_gradients,
)


class _Interpolation(torch.autograd.Function):
    """Autograd bridge for the compiled value-and-spatial-derivative kernel."""

    @staticmethod
    def forward(ctx, coefficients, indices, grid_indices):
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
        grad = torch.zeros_like(derivatives)
        if grad_values is not None:
            grad.add_(derivatives * grad_values[:, None])
        if grad_penalties is not None:
            grad.add_(penalty_derivatives * grad_penalties[:, None])
        return None, grad, None


def interpolate_grid(
    coefficients: torch.Tensor,
    indices: torch.Tensor,
    grid_indices: torch.Tensor,
    penalty: bool = False,
):
    """Boundary-clamped tricubic B-spline values at zero-based ``indices``.

    Differentiable with respect to ``indices``. With ``penalty=True`` returns
    ``(values, out_of_bounds_penalties)``.
    """
    values, penalties = _Interpolation.apply(coefficients, indices, grid_indices)
    return (values, penalties) if penalty else values


class _Score(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, scorer):
        energies, gradients = scorer.value_and_grad(coords)
        ctx.save_for_backward(gradients)
        return energies

    @staticmethod
    def backward(ctx, grad_energies):
        (gradients,) = ctx.saved_tensors
        return gradients * grad_energies[:, None, None], None


class GridScorer:
    """Energy and analytic Cartesian gradient of ligand pose batches.

    Args:
        grids: Receptor grids (any device/dtype; copied to ``device``/``dtype``).
        ligand_atom_types: Grid atom type of each grid-scored ligand atom.
        ligand_charges: Partial charge of each grid-scored ligand atom.
        device, dtype: Where and in which precision poses are scored.
        score_function: Source of the term weights (default ``beta2016``).
        live_hbond, live_lk_ball, live_ligand_intra: Optional live terms.
        lk_ball_grid: Use the gridded Rosetta LK-ball total (needs
            ``live_lk_ball`` for the water topology) instead of the exact one.
        grid_atoms: Indices of the coordinate atoms scored on the grids, when
            the movable set is larger than the grid-scored set (flexible
            sidechains); default all atoms.
    """

    def __init__(
        self,
        grids: GridSet,
        ligand_atom_types: Sequence[str],
        ligand_charges: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        score_function: ScoreFunction | None = None,
        live_hbond: LiveHBondData | None = None,
        live_lk_ball: LiveLKBallData | None = None,
        live_ligand_intra: LiveLigandIntraData | None = None,
        lk_ball_grid: bool = False,
        grid_atoms: torch.Tensor | None = None,
    ) -> None:
        device = torch.empty(0, device=device).device
        self.grids = grids.on(device, dtype)
        self.live_hbond = live_hbond and live_hbond.to(device, dtype)
        self.live_lk_ball = live_lk_ball and live_lk_ball.to(device, dtype)
        self.live_ligand_intra = live_ligand_intra and live_ligand_intra.to(
            device, dtype
        )
        self.lk_ball_grid = lk_ball_grid
        if lk_ball_grid and self.live_lk_ball is None:
            raise ValueError("gridded LK-ball requires the ligand water topology")
        self._grid_atoms = None if grid_atoms is None else grid_atoms.to(device)

        to_grid = grids.atom_type_to_grid
        if any(name not in to_grid for name in ligand_atom_types):
            raise ValueError("ligand atom types are absent from the grid")
        self._grid_index = torch.tensor(
            [to_grid[name] for name in ligand_atom_types],
            dtype=torch.int64,
            device=device,
        )
        self._charges = ligand_charges.to(device=device, dtype=dtype)

        sfxn = score_function or beta2016_score_function(device)

        def weights(score_types):
            return torch.stack([sfxn.get_weight(t) for t in score_types]).to(
                device=device, dtype=dtype
            )

        self._weights = weights(GRID_SCORE_TYPES)
        self._hbond_weight = weights((ScoreType.hbond,))[0]
        self._lk_ball_weights = weights(LK_BALL_SCORE_TYPES)

        if lk_ball_grid:
            polar = self.live_lk_ball.ligand_polar_atoms
            if self._grid_atoms is None:
                polar_grid_index = self._grid_index[polar]
            else:
                position = torch.full(
                    (self.live_lk_ball.n_ligand_atoms,),
                    -1,
                    dtype=torch.int64,
                    device=device,
                )
                position[self._grid_atoms] = torch.arange(
                    self._grid_atoms.numel(), dtype=torch.int64, device=device
                )
                if bool(torch.any(position[polar] < 0)):
                    raise ValueError("an LK-ball polar atom is not grid-scored")
                polar_grid_index = self._grid_index[position[polar]]
            self._water_grid_index = polar_grid_index[:, None].expand(-1, 4).reshape(-1)

    @property
    def device(self) -> torch.device:
        return self._grid_index.device

    def _grid_channels(self, grid_coords: torch.Tensor):
        """Per-atom ``fa_atr, fa_rep, fa_sol, fa_elec`` values and gradients.

        Returns ``(values [pose, atom, 4], gradients [pose, atom, 4, 3],
        penalty [pose, atom], penalty_gradient [pose, atom, 3])`` in Cartesian
        units, before term weights.
        """
        n_poses, n_atoms, _ = grid_coords.shape
        grids = self.grids
        indices = grids.geometry.zero_based_index_from_cartesian(
            grid_coords.reshape(-1, 3)
        ).contiguous()
        atom_type = self._grid_index.repeat(n_poses)
        charge = self._charges.repeat(n_poses)

        atr, d_atr, penalty, d_penalty = interpolate_with_derivatives(
            grids.fa_atr_coeffs, indices, atom_type, OUT_OF_BOUNDS_PENALTY
        )
        rep, d_rep, _, _ = interpolate_with_derivatives(
            grids.fa_rep_coeffs, indices, atom_type, 0.0
        )
        sol, d_sol, _, _ = interpolate_with_derivatives(
            grids.fa_sol_coeffs, indices, atom_type, 0.0
        )
        elec, d_elec, _, _ = interpolate_with_derivatives(
            grids.fa_elec_coeffs[None], indices, torch.zeros_like(atom_type), 0.0
        )
        scale = 1.0 / grids.geometry.voxel_spacing
        values = torch.stack((atr, ixform_rep(rep), sol, charge * elec), dim=1)
        # fa_rep is stored transformed; chain rule through its inverse.
        d_rep = d_rep * (-4.0 * rep.pow(-5))[:, None]
        gradients = torch.stack((d_atr, d_rep, d_sol, charge[:, None] * d_elec), dim=1)
        return (
            values.reshape(n_poses, n_atoms, 4),
            (gradients * scale).reshape(n_poses, n_atoms, 4, 3),
            penalty.reshape(n_poses, n_atoms),
            (d_penalty * scale).reshape(n_poses, n_atoms, 3),
        )

    def _grid_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if self._grid_atoms is None:
            return coords
        return coords.index_select(1, self._grid_atoms)

    def grid_channels(self, coords: torch.Tensor) -> torch.Tensor:
        """Weighted ``fa_atr, fa_rep, fa_sol, fa_elec`` per pose, without the
        out-of-bounds penalty: ``[pose, 4]``."""
        with torch.no_grad():
            values = self._grid_channels(self._grid_coords(coords.detach()))[0]
            return (values * self._weights).sum(dim=1)

    def _gridded_lk_ball(self, coords: torch.Tensor):
        """Rosetta's gridded LK-ball total per pose and its Cartesian gradient."""
        grids, data = self.grids, self.live_lk_ball
        n_poses = coords.shape[0]
        grid_coords = self._grid_coords(coords)
        n_grid_atoms = grid_coords.shape[1]
        indices = grids.geometry.zero_based_index_from_cartesian(
            grid_coords.reshape(-1, 3)
        ).contiguous()
        heavy, d_heavy, _, _ = interpolate_with_derivatives(
            grids.lk_ball_coeffs, indices, self._grid_index.repeat(n_poses), 0.0
        )
        energies = heavy.reshape(n_poses, n_grid_atoms).sum(dim=1)
        grad = (d_heavy / grids.geometry.voxel_spacing).reshape(
            n_poses, n_grid_atoms, 3
        )
        if self._grid_atoms is not None:
            grad = torch.zeros_like(coords).index_copy(1, self._grid_atoms, grad)

        n_polar = data.ligand_polar_atoms.shape[0]
        if n_polar == 0:
            return energies, grad
        water_args = (
            data.ligand_polar_atoms,
            data.ligand_water_kinds,
            data.ligand_water_atoms,
            data.ligand_water_params,
            data.ligand_water_is_ring,
        )
        waters = build_lk_ball_waters(
            coords.contiguous(), *water_args, GRID_LK_BALL_WATER_DISTANCE
        )
        active = data.ligand_water_kinds != 0
        has_water = active.any(dim=1)
        safe = torch.where(
            active[None, :, :, None], waters, grids.geometry.origin.reshape(1, 1, 1, 3)
        )
        water_indices = grids.geometry.zero_based_index_from_cartesian(
            safe.reshape(-1, 3)
        ).contiguous()
        water_type = self._water_grid_index.repeat(n_poses)
        water, d_water, _, _ = interpolate_with_derivatives(
            grids.lk_ball_water_coeffs, water_indices, water_type, 0.0
        )
        bridge, d_bridge, _, _ = interpolate_with_derivatives(
            grids.lk_bridge_coeffs, water_indices, water_type, 0.0
        )
        if not grids.lk_bridge_enabled:
            bridge, d_bridge = torch.zeros_like(bridge), torch.zeros_like(d_bridge)
        water = water.reshape(n_poses, n_polar, 4)
        bridge = bridge.reshape(n_poses, n_polar, 4)
        d_water = d_water.reshape(n_poses, n_polar, 4, 3)
        d_bridge = d_bridge.reshape(n_poses, n_polar, 4, 3)

        # Softmax over the waters of a polar atom, softmin for bridges; atoms
        # without waters contribute nothing.
        fade = GRID_LK_BALL_FADE
        keep = has_water[None, :, None]
        water_m = torch.where(
            keep, torch.where(active[None], water, -torch.inf), torch.zeros_like(water)
        )
        bridge_m = torch.where(
            keep, torch.where(active[None], bridge, torch.inf), torch.zeros_like(bridge)
        )
        water_energy = fade * torch.logsumexp(water_m / fade, dim=2)
        weight_water = torch.softmax(water_m / fade, dim=2) * active[None]
        energy = water_energy
        weight_bridge = torch.zeros_like(weight_water)
        if grids.lk_bridge_enabled:
            energy = energy - fade * torch.logsumexp(-bridge_m / fade, dim=2)
            weight_bridge = torch.softmax(-bridge_m / fade, dim=2) * active[None]
        energies = energies + torch.where(
            has_water[None], energy, torch.zeros_like(energy)
        ).sum(dim=1)
        water_gradient = (
            weight_water[..., None] * d_water + weight_bridge[..., None] * d_bridge
        ) / grids.geometry.voxel_spacing
        grad = grad + propagate_lk_ball_water_gradients(
            coords.contiguous(),
            *water_args,
            water_gradient.contiguous(),
            GRID_LK_BALL_WATER_DISTANCE,
        )
        return energies, grad

    def value_and_grad(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pose energy ``[pose]`` and gradient ``[pose, atom, 3]``, graph-free.

        ``coords`` are the movable ligand (and sidechain) atoms, ``[pose, atom, 3]``.
        """
        if coords.dtype != self._charges.dtype or coords.device != self.device:
            raise TypeError(
                f"scorer works on {self.device} {self._charges.dtype} coordinates"
            )
        with torch.no_grad():
            coords = coords.detach().contiguous()
            values, gradients, penalty, d_penalty = self._grid_channels(
                self._grid_coords(coords)
            )
            weights = self._weights
            energies = (values * weights).sum(dim=(1, 2)) + penalty.sum(dim=1)
            grad = (gradients * weights[:, None]).sum(dim=2) + d_penalty
            if self._grid_atoms is not None:
                grad = torch.zeros_like(coords).index_copy(1, self._grid_atoms, grad)

            if self.live_hbond is not None:
                e, g = score_ligand_hbond_with_gradients(self.live_hbond, coords)
                energies = energies + self._hbond_weight * e
                grad = grad + self._hbond_weight * g
            if self.lk_ball_grid:
                if self.grids.lk_ball_enabled:
                    e, g = self._gridded_lk_ball(coords)
                    energies, grad = energies + e, grad + g
            elif self.live_lk_ball is not None:
                e, g = score_ligand_lk_ball_with_gradients(self.live_lk_ball, coords)
                weights = self._lk_ball_weights
                energies = energies + (e * weights).sum(dim=1)
                grad = grad + (g * weights[None, :, None, None]).sum(dim=1)
            if self.live_ligand_intra is not None:
                e, g = score_ligand_intra_with_gradients(self.live_ligand_intra, coords)
                energies, grad = energies + e, grad + g
        return energies, grad

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Per-pose energy, differentiable with respect to ``coords``."""
        if torch.is_grad_enabled() and coords.requires_grad:
            return _Score.apply(coords, self)
        return self.value_and_grad(coords)[0]

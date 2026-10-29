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
    grid_score_atoms,
    grid_score_waters,
    interpolate_with_derivatives,
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
        self._grid_atom_index = (
            torch.arange(len(ligand_atom_types), device=device)
            if self._grid_atoms is None
            else self._grid_atoms
        )
        # Heavy-atom part of the gridded LK-ball total (empty: not gridded).
        gridded_lk = lk_ball_grid and self.grids.lk_ball_enabled
        self._lk_heavy_coeffs = (
            self.grids.lk_ball_coeffs if gridded_lk else self.grids.fa_atr_coeffs[:0]
        )

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
            self._polar_grid_index = polar_grid_index

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
            grids = self.grids
            atom_energies, grad = grid_score_atoms(
                coords,
                self._grid_atom_index,
                self._grid_index,
                self._charges,
                grids.fa_atr_coeffs,
                grids.fa_rep_coeffs,
                grids.fa_sol_coeffs,
                grids.fa_elec_coeffs[None],
                self._lk_heavy_coeffs,
                grids.geometry.origin,
                grids.geometry.voxel_spacing,
                self._weights,
                OUT_OF_BOUNDS_PENALTY,
            )
            energies = atom_energies.sum(dim=1)
            if (
                self._lk_heavy_coeffs.shape[0]
                and self.live_lk_ball.ligand_polar_atoms.numel()
            ):
                data = self.live_lk_ball
                grid_score_waters(
                    coords,
                    data.ligand_polar_atoms,
                    data.ligand_water_kinds,
                    data.ligand_water_atoms,
                    data.ligand_water_params,
                    data.ligand_water_is_ring,
                    self._polar_grid_index,
                    grids.lk_ball_water_coeffs,
                    grids.lk_bridge_coeffs,
                    grids.geometry.origin,
                    grids.geometry.voxel_spacing,
                    GRID_LK_BALL_WATER_DISTANCE,
                    GRID_LK_BALL_FADE,
                    grids.lk_bridge_enabled,
                    energies,
                    grad,
                )

            if self.live_hbond is not None:
                e, g = score_ligand_hbond_with_gradients(self.live_hbond, coords)
                energies = energies + self._hbond_weight * e
                grad = grad + self._hbond_weight * g
            if not self.lk_ball_grid and self.live_lk_ball is not None:
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

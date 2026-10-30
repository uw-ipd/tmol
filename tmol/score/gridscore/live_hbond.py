"""Live ligand--receptor hydrogen bonds for rigid-receptor grid scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence

import torch

from tmol.database import ParameterDatabase
from tmol.pose._pose_stack import PoseStack
from tmol.score.hbond._hbond_energy_term import HBondEnergyTerm

from ._common import (
    LiveData,
    build_cell_list,
    cell_grid,
    derived_atom_indices,
    validate_ligand_coords,
)
from ._movable import movable_atoms as movable_atoms_of


@dataclass(frozen=True)
class LiveHBondData(LiveData):
    """One shared rigid-receptor hbond structure and one ligand topology.

    Receptor coordinates and cell lists occur once, independently of the
    number of ligand poses. Ligand tensors contain only block-local atom
    indices, so every pose supplies its own current base geometry.
    """

    _float64_fields: ClassVar[frozenset] = frozenset({"pair_polynomials"})

    n_ligand_atoms: int
    receptor_donors: torch.Tensor
    receptor_donor_types: torch.Tensor
    receptor_acceptors: torch.Tensor
    receptor_acceptor_types: torch.Tensor
    ligand_donor_atoms: torch.Tensor
    ligand_donor_types: torch.Tensor
    ligand_acceptor_atoms: torch.Tensor
    ligand_acceptor_types: torch.Tensor
    cell_origin: torch.Tensor
    cell_dims: torch.Tensor
    receptor_donor_cell_offsets: torch.Tensor
    receptor_donor_cell_indices: torch.Tensor
    receptor_acceptor_cell_offsets: torch.Tensor
    receptor_acceptor_cell_indices: torch.Tensor
    pair_params: torch.Tensor
    pair_polynomials: torch.Tensor
    global_params: torch.Tensor


def _hbond_sites(pose_stack: PoseStack, derived: torch.Tensor):
    """Every donor ``(heavy, H, type)`` and acceptor ``(A, base, base0, type)``.

    Rows hold global atom indices in block, tile, site order.
    """

    params = pose_stack.packed_block_types.hbpbt_params
    block_types = pose_stack.block_type_ind64[0]
    blocks = torch.nonzero(block_types >= 0).squeeze(1)
    types = block_types[blocks]
    offsets = pose_stack.block_coord_offset64[0, blocks]

    def sites(n_per_tile, atom_inds, site_types):
        n_tiles, width = atom_inds.shape[1:]
        tiles = torch.arange(n_tiles) * HBondEnergyTerm.tile_size
        chosen = torch.arange(width) < n_per_tile[types].unsqueeze(-1)
        atoms = offsets[:, None, None] + tiles[None, :, None] + atom_inds[types]
        return atoms[chosen].to(torch.int64), site_types[types][chosen].to(torch.int64)

    hydrogen, donor_type = sites(
        params.tile_n_donH, params.tile_donH_inds, params.tile_donorH_type
    )
    acceptor, acceptor_type = sites(
        params.tile_n_acc, params.tile_acc_inds, params.tile_acceptor_type
    )
    donor = derived[hydrogen, 0]
    base, base0 = derived[acceptor, 1], derived[acceptor, 2]
    if bool((donor < 0).any() | (base < 0).any() | (base0 < 0).any()):
        raise RuntimeError("hbond base generation returned no base atom")
    return (
        (donor, hydrogen, donor_type),
        (acceptor, base, base0, acceptor_type),
    )


def build_live_hbond_data(
    pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    *,
    sidechain_blocks: tuple[int, ...] = (),
    movable_atoms: Sequence[int] | None = None,
) -> LiveHBondData:
    """Build movable-atom topology and shared fixed-receptor hbond cells.

    ``movable_atoms`` are the compact-pose atoms that move (all ligand atoms by
    default; required with ``sidechain_blocks``).
    """

    if pose_stack.n_poses != 1 or pose_stack.device.type != "cpu":
        raise ValueError("live hbond construction requires one CPU pose")
    if sidechain_blocks and movable_atoms is None:
        raise ValueError("sidechain_blocks require the compact movable_atoms")
    moving = movable_atoms_of(pose_stack, ligand_block, sidechain_blocks)
    atom_map = moving.atom_map
    is_query = torch.ones(moving.n_atoms, dtype=torch.bool)
    if movable_atoms is not None:
        is_query = torch.zeros(moving.n_atoms, dtype=torch.bool)
        is_query[torch.as_tensor(list(movable_atoms), dtype=torch.int64)] = True

    term = HBondEnergyTerm(param_db, torch.device("cpu"))
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    donors, acceptors = _hbond_sites(pose_stack, derived_atom_indices(pose_stack))
    donor, hydrogen, donor_type = donors
    acceptor, base, base0, acceptor_type = acceptors

    def compact(atoms):
        return atom_map[atoms]

    # A site of a moving block is a ligand site when its heavy atom moves;
    # every other site belongs to the fixed receptor.
    donor_fixed = compact(hydrogen) < 0
    donor_mine = (compact(donor) >= 0) & is_query[compact(donor).clamp(min=0)]
    acceptor_fixed = compact(acceptor) < 0
    acceptor_mine = (compact(acceptor) >= 0) & is_query[compact(acceptor).clamp(min=0)]
    if bool((compact(donor)[donor_fixed] >= 0).any()) or bool(
        (
            (compact(base)[acceptor_fixed] >= 0) | (compact(base0)[acceptor_fixed] >= 0)
        ).any()
    ):
        raise ValueError("receptor hbond base points into the moving blocks")
    if bool((compact(hydrogen)[donor_mine] < 0).any()) or bool(
        ((compact(base)[acceptor_mine] < 0) | (compact(base0)[acceptor_mine] < 0)).any()
    ):
        raise ValueError("movable hbond geometry crosses a block boundary")

    coords = pose_stack.coords.reshape(-1, 3).detach().to(torch.float64)
    receptor_donors = coords[torch.stack((donor, hydrogen), 1)[donor_fixed]]
    receptor_acceptors = coords[torch.stack((acceptor, base, base0), 1)[acceptor_fixed]]

    pair_params, pair_polynomials, global_params = term._param_tables(torch.float64)
    max_ha_dis = float(global_params[0, 5])
    if not (max_ha_dis > 0):
        raise RuntimeError("hbond max H--A distance must be positive")

    anchors = torch.cat((receptor_donors[:, 1], receptor_acceptors[:, 0]))
    cell_origin, cell_dims = cell_grid(anchors, max_ha_dis)
    donor_offsets, donor_cell_indices = build_cell_list(
        receptor_donors[:, 1], cell_origin, cell_dims, max_ha_dis
    )
    acceptor_offsets, acceptor_cell_indices = build_cell_list(
        receptor_acceptors[:, 0], cell_origin, cell_dims, max_ha_dis
    )
    return LiveHBondData(
        n_ligand_atoms=moving.n_atoms,
        receptor_donors=receptor_donors,
        receptor_donor_types=donor_type[donor_fixed],
        receptor_acceptors=receptor_acceptors,
        receptor_acceptor_types=acceptor_type[acceptor_fixed],
        ligand_donor_atoms=torch.stack((compact(donor), compact(hydrogen)), 1)[
            donor_mine
        ],
        ligand_donor_types=donor_type[donor_mine],
        ligand_acceptor_atoms=torch.stack(
            (compact(acceptor), compact(base), compact(base0)), 1
        )[acceptor_mine],
        ligand_acceptor_types=acceptor_type[acceptor_mine],
        cell_origin=cell_origin.to(torch.float64).contiguous(),
        cell_dims=cell_dims.contiguous(),
        receptor_donor_cell_offsets=donor_offsets,
        receptor_donor_cell_indices=donor_cell_indices,
        receptor_acceptor_cell_offsets=acceptor_offsets,
        receptor_acceptor_cell_indices=acceptor_cell_indices,
        pair_params=pair_params.detach().clone().contiguous(),
        pair_polynomials=pair_polynomials.detach().clone().contiguous(),
        global_params=global_params.detach().clone().contiguous(),
    )


def _validate_ligand_coords(data: LiveHBondData, ligand_coords: torch.Tensor) -> None:
    validate_ligand_coords(
        ligand_coords, data.n_ligand_atoms, data.receptor_donors, "hbond"
    )


def _live_hbond_op(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from .potentials.compiled import live_hbond

    return live_hbond(
        ligand_coords.contiguous(),
        data.ligand_donor_atoms,
        data.ligand_donor_types,
        data.ligand_acceptor_atoms,
        data.ligand_acceptor_types,
        data.receptor_donors,
        data.receptor_donor_types,
        data.receptor_acceptors,
        data.receptor_acceptor_types,
        data.cell_origin,
        data.cell_dims,
        data.receptor_donor_cell_offsets,
        data.receptor_donor_cell_indices,
        data.receptor_acceptor_cell_offsets,
        data.receptor_acceptor_cell_indices,
        data.pair_params,
        data.pair_polynomials,
        data.global_params,
    )


class _LiveHBondAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ligand_coords: torch.Tensor, data: LiveHBondData):
        energies, gradients = _live_hbond_op(data, ligand_coords)
        ctx.save_for_backward(gradients)
        return energies

    @staticmethod
    def backward(ctx, grad_energies):
        (gradients,) = ctx.saved_tensors
        return gradients * grad_energies[:, None, None], None


def score_ligand_hbond(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return unweighted live interface hbond energy for every ligand pose."""

    _validate_ligand_coords(data, ligand_coords)
    if ligand_coords.requires_grad:
        return _LiveHBondAutograd.apply(ligand_coords, data)
    return _live_hbond_op(data, ligand_coords)[0]


def score_ligand_hbond_with_gradients(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted live hbond energies and analytic ligand gradients."""

    _validate_ligand_coords(data, ligand_coords)
    with torch.no_grad():
        return _live_hbond_op(data, ligand_coords.detach())

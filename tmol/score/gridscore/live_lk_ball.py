"""Live ligand--receptor LK-ball terms for rigid-receptor grid scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from tmol.database import ParameterDatabase
from tmol.pose._pose_stack import PoseStack
from tmol.score.common._convert_float64 import _convert_float64
from tmol.score.lk_ball._lk_ball_energy_term import LKBallEnergyTerm

from ._common import (
    LiveData,
    build_cell_list,
    cell_grid,
    derived_atom_indices,
    validate_ligand_coords,
)
from ._movable import movable_atoms as movable_atoms_of, pose_atom_table

MAX_N_WATER = 4


@dataclass(frozen=True)
class LiveLKBallData(LiveData):
    """One shared rigid-receptor LK-ball structure and ligand topology."""

    n_ligand_atoms: int
    ligand_atom_params: torch.Tensor
    ligand_heavy_atoms: torch.Tensor
    ligand_polar_atoms: torch.Tensor
    ligand_atom_to_polar: torch.Tensor
    ligand_water_kinds: torch.Tensor
    ligand_water_atoms: torch.Tensor
    ligand_water_params: torch.Tensor
    ligand_water_is_ring: torch.Tensor
    receptor_coords: torch.Tensor
    receptor_params: torch.Tensor
    receptor_waters: torch.Tensor
    receptor_polar_atoms: torch.Tensor
    cell_origin: torch.Tensor
    cell_dims: torch.Tensor
    receptor_cell_offsets: torch.Tensor
    receptor_cell_indices: torch.Tensor
    receptor_polar_cell_offsets: torch.Tensor
    receptor_polar_cell_indices: torch.Tensor
    global_params: torch.Tensor


def _pack_type_params(term: LKBallEnergyTerm) -> torch.Tensor:
    params = term.ljlk_param_resolver.type_params
    # Match LKBallTypeParams: these columns hold precomputed factors, not the
    # raw dgfree/lambda values. Compute in the live scorer's float64 precision.
    lk_dgfree = params.lk_dgfree.to(dtype=torch.float64)
    lk_lambda = params.lk_lambda.to(dtype=torch.float64)
    return torch.stack(
        tuple(
            value.to(dtype=torch.float64)
            for value in (
                params.lj_radius,
                -lk_dgfree / (2 * 5.56832799683 * lk_lambda),
                1 / (lk_lambda * lk_lambda),
                params.lk_volume,
                params.is_donor,
                params.is_hydroxyl,
                params.is_polarh,
                params.is_acceptor,
                params.is_carbon_lk,
            )
        ),
        dim=1,
    ).contiguous()


def _generate_reference_waters(
    pose_stack: PoseStack,
    term: LKBallEnergyTerm,
    *,
    water_distance: float | None = None,
) -> torch.Tensor:
    from tmol.score.lk_ball.potentials._compiled import _gen_pose_waters

    pbt = pose_stack.packed_block_types
    water_gen_global_params = term._lk_ball_water_gen_global_params
    if water_distance is not None:
        # Assign the Rosetta grid distance after promotion so fp64 grid builds
        # do not inherit a rounded float32 representation of 2.65 A.
        water_gen_global_params = water_gen_global_params.to(torch.float64).clone()
        water_gen_global_params[:, 0] = water_distance
    args = [
        pose_stack.coords.reshape(-1, 3).to(torch.float64).contiguous(),
        pose_stack.rot_coord_offset,
        pose_stack.pose_ind_for_atom,
        pose_stack.first_rot_for_block,
        pose_stack.first_rot_block_type,
        pose_stack.block_ind_for_rot,
        pose_stack.pose_ind_for_rot,
        pose_stack.block_type_ind_for_rot,
        pose_stack.n_rots_for_pose,
        pose_stack.rot_offset_for_pose,
        pose_stack.n_rots_for_block,
        pose_stack.rot_offset_for_block,
        torch.full_like(pose_stack.block_type_ind, -1),
        pose_stack.max_n_rots_per_pose,
        pose_stack.inter_residue_connections,
        pbt.n_atoms,
        pbt.n_conn,
        pbt.conn_atom,
        pbt.n_all_bonds,
        pbt.all_bonds,
        pbt.atom_all_bond_ranges,
        pbt.hbpbt_params.tile_n_donH,
        pbt.hbpbt_params.tile_n_acc,
        pbt.hbpbt_params.tile_donH_inds,
        pbt.hbpbt_params.tile_donH_hvy_inds,
        pbt.hbpbt_params.tile_which_donH_of_donH_hvy,
        pbt.hbpbt_params.tile_acc_inds,
        pbt.hbpbt_params.tile_acceptor_hybridization,
        pbt.hbpbt_params.tile_acceptor_n_attached_H,
        pbt.hbpbt_params.is_hydrogen,
        water_gen_global_params,
        term.ljlk_param_resolver.global_params.lkb_water_tors_sp2,
        term.ljlk_param_resolver.global_params.lkb_water_tors_sp3,
        term.ljlk_param_resolver.global_params.lkb_water_tors_ring,
    ]
    _convert_float64(args)
    with torch.no_grad():
        return _gen_pose_waters(*args).detach().to(torch.float64).contiguous()


def _ligand_water_topology(
    pose_stack: PoseStack,
    ligand_block: int,
    term: LKBallEnergyTerm,
    derived_atom_inds: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    pbt = pose_stack.packed_block_types
    type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pbt.active_block_types[type_index]
    hb = block_type.hbbt_params
    n_atoms = block_type.n_atoms
    offset = int(pose_stack.block_coord_offset64[0, ligand_block])
    end = offset + n_atoms
    polar_atoms = sorted(set(hb.don_hvy_inds.tolist()) | set(hb.acc_inds.tolist()))
    polar = torch.tensor(polar_atoms, dtype=torch.int64)
    atom_to_polar = torch.full((n_atoms,), -1, dtype=torch.int64)
    if polar.numel():
        atom_to_polar[polar] = torch.arange(polar.shape[0], dtype=torch.int64)
    kinds = torch.zeros((polar.shape[0], MAX_N_WATER), dtype=torch.int64)
    atoms = torch.full((polar.shape[0], MAX_N_WATER, 3), -1, dtype=torch.int64)
    params = torch.zeros((polar.shape[0], MAX_N_WATER, 3), dtype=torch.float64)
    rings = torch.zeros((polar.shape[0], MAX_N_WATER), dtype=torch.int64)

    water_globals = term._lk_ball_water_gen_global_params[0].to(torch.float64)
    water_dist = float(water_globals[0])
    for tile in range(hb.tile_n_donH.shape[0]):
        for site in range(int(hb.tile_n_donH[tile])):
            hydrogen = tile * term.tile_size + int(hb.tile_donH_inds[tile, site])
            donor = int(hb.tile_donH_hvy_inds[tile, site])
            slot = int(hb.tile_which_donH_of_donH_hvy[tile, site])
            if not (0 <= donor < n_atoms and 0 <= hydrogen < n_atoms):
                raise RuntimeError("ligand donor water geometry is not block-local")
            if not 0 <= slot < MAX_N_WATER:
                raise RuntimeError("ligand donor water slot exceeds MAX_N_WATER")
            polar_site = int(atom_to_polar[donor])
            if polar_site < 0 or int(kinds[polar_site, slot]) != 0:
                raise RuntimeError("ligand donor water topology is inconsistent")
            kinds[polar_site, slot] = 1
            atoms[polar_site, slot, :2] = torch.tensor((donor, hydrogen))
            params[polar_site, slot, 0] = water_dist

    torsions = {
        1: term.ljlk_param_resolver.global_params.lkb_water_tors_sp2,
        2: term.ljlk_param_resolver.global_params.lkb_water_tors_sp3,
        3: term.ljlk_param_resolver.global_params.lkb_water_tors_ring,
    }
    acceptors = {}
    for tile in range(hb.tile_n_acc.shape[0]):
        for site in range(int(hb.tile_n_acc[tile])):
            acceptor = tile * term.tile_size + int(hb.tile_acc_inds[tile, site])
            acceptors[acceptor] = (
                int(hb.tile_acceptor_hybridization[tile, site]),
                int(hb.tile_acceptor_n_attached_H[tile, site]),
            )
    for acceptor in hb.acc_inds.tolist():
        hybridization, water_offset = acceptors[int(acceptor)]
        global_acceptor = offset + int(acceptor)
        base = int(derived_atom_inds[global_acceptor, 1])
        base0 = int(derived_atom_inds[global_acceptor, 2])
        if base < 0 or base0 < 0 or base == base0:
            continue
        if not (offset <= base < end and offset <= base0 < end):
            raise ValueError(
                "ligand acceptor water geometry crosses a block boundary; "
                "live grid LK-ball cannot represent it"
            )
        if hybridization not in torsions:
            raise RuntimeError("ligand acceptor has unsupported hybridization")
        polar_site = int(atom_to_polar[int(acceptor)])
        for water_index, torsion in enumerate(torsions[hybridization]):
            slot = water_offset + water_index
            if not 0 <= slot < MAX_N_WATER:
                raise RuntimeError("ligand acceptor water slot exceeds MAX_N_WATER")
            if int(kinds[polar_site, slot]) != 0:
                raise RuntimeError("ligand acceptor water topology is inconsistent")
            kinds[polar_site, slot] = 2
            atoms[polar_site, slot] = torch.tensor(
                (int(acceptor), base - offset, base0 - offset), dtype=torch.int64
            )
            params[polar_site, slot] = torch.tensor(
                (water_dist, float(water_globals[hybridization]), float(torsion)),
                dtype=torch.float64,
            )
            rings[polar_site, slot] = int(hybridization == 3)
    return polar, atom_to_polar, kinds, atoms, params, rings


def _polar_atom_mask(pbt, block_types: torch.Tensor) -> torch.Tensor:
    """``[block type, atom]`` mask of hbond donor heavy atoms and acceptors."""
    mask = torch.zeros(pbt.atom_types.shape, dtype=torch.bool)
    for index in block_types.unique().tolist():
        hb = pbt.active_block_types[index].hbbt_params
        mask[index, torch.as_tensor(hb.don_hvy_inds, dtype=torch.int64)] = True
        mask[index, torch.as_tensor(hb.acc_inds, dtype=torch.int64)] = True
    return mask


def build_live_lk_ball_data(
    pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    *,
    sidechain_blocks: tuple[int, ...] = (),
    movable_atoms: Sequence[int] | None = None,
) -> LiveLKBallData:
    """Build movable water topology and one fixed-receptor cell structure.

    ``movable_atoms`` are the compact-pose atoms that move (all ligand atoms by
    default; required with ``sidechain_blocks``).
    """

    if pose_stack.n_poses != 1 or pose_stack.device.type != "cpu":
        raise ValueError("live LK-ball construction requires one CPU pose")
    if sidechain_blocks and movable_atoms is None:
        raise ValueError("sidechain_blocks require the compact movable_atoms")
    moving = movable_atoms_of(pose_stack, ligand_block, sidechain_blocks)
    is_query = torch.ones(moving.n_atoms, dtype=torch.bool)
    if movable_atoms is not None:
        is_query = torch.zeros(moving.n_atoms, dtype=torch.bool)
        is_query[torch.as_tensor(list(movable_atoms), dtype=torch.int64)] = True

    term = LKBallEnergyTerm(param_db, torch.device("cpu"))
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    type_params = _pack_type_params(term)
    reference_waters = _generate_reference_waters(pose_stack, term)
    derived = derived_atom_indices(pose_stack)
    pbt = pose_stack.packed_block_types
    is_hydrogen = pbt.hbpbt_params.is_hydrogen.to(torch.bool)

    # Moving atoms and their water topology, block by block of the compact pose.
    atom_types, hydrogens = [], []
    polar, kinds, water_atoms, water_params, rings = [], [], [], [], []
    for source, offset in zip(moving.source_blocks, moving.compact_offsets):
        type_index = int(pose_stack.block_type_ind64[0, source])
        n_atoms = pbt.active_block_types[type_index].n_atoms
        atom_types.append(pbt.atom_types[type_index, :n_atoms].to(torch.int64))
        hydrogens.append(is_hydrogen[type_index, :n_atoms])
        block_polar, _, *topology = _ligand_water_topology(
            pose_stack, source, term, derived
        )
        keep = is_query[block_polar + offset]
        block_kinds, block_atoms, block_params, block_rings = (
            t[keep] for t in topology
        )
        polar.append(block_polar[keep] + offset)
        kinds.append(block_kinds)
        water_atoms.append(torch.where(block_atoms >= 0, block_atoms + offset, -1))
        water_params.append(block_params)
        rings.append(block_rings)
    ligand_atom_types = torch.cat(atom_types)
    ligand_polar_atoms = torch.cat(polar).contiguous()
    ligand_atom_to_polar = torch.full((moving.n_atoms,), -1, dtype=torch.int64)
    ligand_atom_to_polar[ligand_polar_atoms] = torch.arange(ligand_polar_atoms.shape[0])
    ligand_heavy_atoms = torch.nonzero(is_query & ~torch.cat(hydrogens)).squeeze(1)

    # Fixed receptor heavy atoms, with their reference waters and polar flag.
    atom, _, block_type, local = pose_atom_table(pose_stack)
    heavy = (moving.atom_map[atom] < 0) & ~is_hydrogen[block_type, local]
    atom, block_type, local = atom[heavy], block_type[heavy], local[heavy]
    if not atom.numel():
        raise ValueError("pose contains no receptor heavy atoms")
    receptor_coords = pose_stack.coords.reshape(-1, 3)[atom].to(torch.float64)
    receptor_polar_atoms = torch.nonzero(
        _polar_atom_mask(pbt, block_type)[block_type, local]
    ).squeeze(1)

    global_params = term._lk_ball_global_params.to(torch.float64).contiguous()
    cutoff = float(global_params[0, 4])
    cell_origin, cell_dims = cell_grid(receptor_coords, cutoff)
    atom_offsets, atom_indices = build_cell_list(
        receptor_coords,
        cell_origin,
        cell_dims,
        cutoff,
        label="receptor LK-ball atom",
    )
    polar_offsets, polar_indices = build_cell_list(
        receptor_coords[receptor_polar_atoms],
        cell_origin,
        cell_dims,
        cutoff,
        receptor_polar_atoms,
        "receptor LK-ball atom",
    )
    return LiveLKBallData(
        n_ligand_atoms=moving.n_atoms,
        ligand_atom_params=type_params[ligand_atom_types].contiguous(),
        ligand_heavy_atoms=ligand_heavy_atoms.contiguous(),
        ligand_polar_atoms=ligand_polar_atoms,
        ligand_atom_to_polar=ligand_atom_to_polar,
        ligand_water_kinds=torch.cat(kinds).contiguous(),
        ligand_water_atoms=torch.cat(water_atoms).contiguous(),
        ligand_water_params=torch.cat(water_params).contiguous(),
        ligand_water_is_ring=torch.cat(rings).contiguous(),
        receptor_coords=receptor_coords.contiguous(),
        receptor_params=type_params[pbt.atom_types[block_type, local].to(torch.int64)],
        receptor_waters=reference_waters[atom],
        receptor_polar_atoms=receptor_polar_atoms.contiguous(),
        cell_origin=cell_origin.contiguous(),
        cell_dims=cell_dims,
        receptor_cell_offsets=atom_offsets,
        receptor_cell_indices=atom_indices,
        receptor_polar_cell_offsets=polar_offsets,
        receptor_polar_cell_indices=polar_indices,
        global_params=global_params,
    )


def _validate_ligand_coords(data: LiveLKBallData, ligand_coords: torch.Tensor) -> None:
    validate_ligand_coords(
        ligand_coords, data.n_ligand_atoms, data.receptor_coords, "LK-ball"
    )


def _live_lk_ball_op(
    data: LiveLKBallData,
    ligand_coords: torch.Tensor,
    *,
    use_cell_list: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .potentials.compiled import live_lk_ball

    return live_lk_ball(
        ligand_coords.contiguous(),
        data.ligand_atom_params,
        data.ligand_heavy_atoms,
        data.ligand_polar_atoms,
        data.ligand_atom_to_polar,
        data.ligand_water_kinds,
        data.ligand_water_atoms,
        data.ligand_water_params,
        data.ligand_water_is_ring,
        data.receptor_coords,
        data.receptor_params,
        data.receptor_waters,
        data.receptor_polar_atoms,
        data.cell_origin,
        data.cell_dims,
        data.receptor_cell_offsets,
        data.receptor_cell_indices,
        data.receptor_polar_cell_offsets,
        data.receptor_polar_cell_indices,
        data.global_params,
        use_cell_list,
    )


class _LiveLKBallAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ligand_coords: torch.Tensor, data: LiveLKBallData):
        energies, gradients = _live_lk_ball_op(data, ligand_coords)
        ctx.save_for_backward(gradients)
        return energies

    @staticmethod
    def backward(ctx, grad_energies: torch.Tensor):
        (gradients,) = ctx.saved_tensors
        expanded_grad_energies = grad_energies[:, :, None, None]
        coordinate_gradients = torch.sum(
            torch.where(
                expanded_grad_energies != 0,
                gradients * expanded_grad_energies,
                torch.zeros_like(gradients),
            ),
            dim=1,
        )
        return coordinate_gradients, None


def score_ligand_lk_ball(
    data: LiveLKBallData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return four unweighted live LK-ball channels for each ligand pose."""

    _validate_ligand_coords(data, ligand_coords)
    if ligand_coords.requires_grad:
        return _LiveLKBallAutograd.apply(ligand_coords, data)
    return _live_lk_ball_op(data, ligand_coords)[0]


def score_ligand_lk_ball_with_gradients(
    data: LiveLKBallData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted channels and analytic per-channel Cartesian gradients."""

    _validate_ligand_coords(data, ligand_coords)
    with torch.no_grad():
        return _live_lk_ball_op(data, ligand_coords.detach())

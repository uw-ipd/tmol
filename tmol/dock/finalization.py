"""RosettaVS-style exact refinement and binding-score finalization."""

from __future__ import annotations

from dataclasses import dataclass

import attr
import numpy
import torch

from tmol.dock.dock_loop import _repeat_pose_stack
from tmol.dock.ga import GAResult, _ligand_only_pose_stack
from tmol.optimization.minimizers import run_cart_min
from tmol.pose.constraint_set import ConstraintSet
from tmol.pose.pdb_info import PDBInfo
from tmol.pose.pose_stack import PoseStack
from tmol.score import beta2016_score_function
from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm
from tmol.score.score_types import ScoreType


@dataclass(frozen=True)
class DockingFinalization:
    """Exact-refined candidates and their RosettaVS ranking components.

    Candidate tensors have shape ``[1, top_k]`` and are sorted by ``dG`` for
    the single input ligand. ``pose_stack`` uses the same candidate order.
    """

    pose_stack: PoseStack
    genes: torch.Tensor
    pre_refine_energies: torch.Tensor
    complex_energies: torch.Tensor
    free_ligand_energies: torch.Tensor
    free_receptor_energies: torch.Tensor
    nchi: torch.Tensor
    dH: torch.Tensor
    TdS: torch.Tensor
    dG: torch.Tensor


def _score_pose_stack(pose_stack: PoseStack, sfxn) -> torch.Tensor:
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    with torch.no_grad():
        return scorer(pose_stack.coords)


def _ligand_atom_indices(pose_stack: PoseStack, ligand_block: int) -> torch.Tensor:
    block_type_indices = pose_stack.block_type_ind64[:, ligand_block]
    if bool(torch.any(block_type_indices < 0)):
        raise ValueError("ligand block is absent from at least one pose")
    if not bool(torch.all(block_type_indices == block_type_indices[0])):
        raise ValueError("finalization requires one homogeneous ligand type")
    block_type = pose_stack.packed_block_types.active_block_types[
        int(block_type_indices[0])
    ]
    local_atoms = torch.arange(
        block_type.n_atoms, dtype=torch.int64, device=pose_stack.device
    )
    return pose_stack.block_coord_offset64[:, ligand_block, None] + local_atoms


def _promote_top_candidates(
    reference_pose_stack: PoseStack,
    ligand_block: int,
    ga_result: GAResult,
    top_k: int,
) -> tuple[PoseStack, torch.Tensor]:
    """Copy only the top grid/GA ligand coordinates into full complexes."""

    if reference_pose_stack.n_poses != 1:
        raise ValueError("finalization requires one reference pose")
    flat = torch.argsort(ga_result.pool.energies)[:top_k]

    pool_ligand_block = (
        0 if ga_result.pool.pose_stack.max_n_blocks == 1 else ligand_block
    )
    source_indices = _ligand_atom_indices(ga_result.pool.pose_stack, pool_ligand_block)[
        flat
    ]
    local_atoms = torch.arange(
        source_indices.shape[1], dtype=torch.int64, device=reference_pose_stack.device
    )
    promoted = _repeat_pose_stack(reference_pose_stack, top_k)
    target_indices = promoted.block_coord_offset64[:, ligand_block, None] + local_atoms
    rows = torch.arange(promoted.n_poses, device=promoted.device)[:, None]
    coords = promoted.coords.clone()
    coords[rows, target_indices] = ga_result.pool.pose_stack.coords[
        flat[:, None], source_indices
    ]
    genes = ga_result.pool.genes[flat].unsqueeze(0)
    return attr.evolve(promoted, coords=coords), genes


def _receptor_only_pose_stack(pose_stack: PoseStack, ligand_block: int) -> PoseStack:
    """Delete one disconnected ligand block while preserving receptor topology."""

    if pose_stack.fragmented_ligand_mapping is not None:
        raise ValueError("finalization does not support fragmented ligand metadata")
    if ligand_block < 0 or ligand_block >= pose_stack.max_n_blocks:
        raise ValueError("ligand_block is out of range")
    if bool(torch.any(pose_stack.block_type_ind64[:, ligand_block] < 0)):
        raise ValueError("ligand block is absent from at least one pose")
    if not bool(
        torch.all(pose_stack.block_type_ind64 == pose_stack.block_type_ind64[0:1])
    ):
        raise ValueError("rigid-receptor finalization requires homogeneous topology")
    if bool(
        torch.any(pose_stack.inter_residue_connections64[:, ligand_block, :, 0] >= 0)
    ):
        raise ValueError("the finalized ligand must be disconnected from the receptor")

    real_blocks = torch.nonzero(
        pose_stack.block_type_ind64[0] >= 0, as_tuple=False
    ).flatten()
    keep_blocks = real_blocks[real_blocks != ligand_block]
    if keep_blocks.numel() == 0:
        raise ValueError("cannot construct an empty receptor")
    old_to_new = torch.full(
        (pose_stack.max_n_blocks,),
        -1,
        dtype=torch.int64,
        device=pose_stack.device,
    )
    old_to_new[keep_blocks] = torch.arange(
        keep_blocks.numel(), dtype=torch.int64, device=pose_stack.device
    )

    connections64 = pose_stack.inter_residue_connections64[:, keep_blocks].clone()
    partner_blocks = connections64[..., 0]
    real_partners = partner_blocks >= 0
    if bool(torch.any(partner_blocks[real_partners] == ligand_block)):
        raise ValueError("the finalized ligand must be disconnected from the receptor")
    partner_blocks[real_partners] = old_to_new[partner_blocks[real_partners]]
    if bool(torch.any(partner_blocks[real_partners] < 0)):
        raise ValueError("receptor connection points reference an absent block")

    block_type_ind64 = pose_stack.block_type_ind64[:, keep_blocks].clone()
    pbt = pose_stack.packed_block_types
    atom_counts = pbt.n_atoms[block_type_ind64[0]].to(torch.int64)
    offsets64 = torch.zeros_like(block_type_ind64)
    if keep_blocks.numel() > 1:
        offsets64[:, 1:] = torch.cumsum(atom_counts[:-1], dim=0)[None, :]
    n_source_atoms = int(torch.sum(pose_stack.n_ats_per_block[0]))
    ligand_start = pose_stack.block_coord_offset64[0, ligand_block]
    ligand_n_atoms = pbt.n_atoms[pose_stack.block_type_ind64[0, ligand_block]].to(
        torch.int64
    )
    all_source_atoms = torch.arange(
        n_source_atoms, dtype=torch.int64, device=pose_stack.device
    )
    source_atoms = all_source_atoms[
        (all_source_atoms < ligand_start)
        | (all_source_atoms >= ligand_start + ligand_n_atoms)
    ]
    coords = pose_stack.coords[:, source_atoms].clone()

    keep_host = keep_blocks.detach().cpu().numpy()
    source_host = source_atoms.detach().cpu().numpy()
    pdb_info = PDBInfo(
        residue_labels=pose_stack.pdb_info.residue_labels[:, keep_host].copy(),
        residue_insertion_codes=(
            pose_stack.pdb_info.residue_insertion_codes[:, keep_host].copy()
        ),
        chain_labels=pose_stack.pdb_info.chain_labels[:, keep_host].copy(),
        atom_occupancy=pose_stack.pdb_info.atom_occupancy[:, source_host].copy(),
        atom_b_factor=pose_stack.pdb_info.atom_b_factor[:, source_host].copy(),
    )
    inter_block_bondsep64 = pose_stack.inter_block_bondsep64[
        :, keep_blocks[:, None], keep_blocks[None, :]
    ].clone()
    chain_id64 = pose_stack.chain_id64[:, keep_blocks].clone()

    return PoseStack(
        packed_block_types=pbt,
        coords=coords,
        block_coord_offset=offsets64.to(torch.int32),
        block_coord_offset64=offsets64,
        inter_residue_connections=connections64.to(torch.int32),
        inter_residue_connections64=connections64,
        inter_block_bondsep=inter_block_bondsep64.to(torch.int32),
        inter_block_bondsep64=inter_block_bondsep64,
        block_type_ind=block_type_ind64.to(torch.int32),
        block_type_ind64=block_type_ind64,
        chain_id=chain_id64.to(torch.int32),
        chain_id64=chain_id64,
        pdb_info=pdb_info,
        constraint_set=None,
        device=pose_stack.device,
    )


def _with_coordinate_constraints(pose_stack: PoseStack) -> PoseStack:
    """Constrain every ligand atom to its current coordinate with sigma 1 A."""

    n_poses, n_atoms = pose_stack.coords.shape[:2]
    pose = torch.arange(n_poses, device=pose_stack.device)[:, None].expand(-1, n_atoms)
    atom = torch.arange(n_atoms, device=pose_stack.device)[None, :].expand(n_poses, -1)
    block = torch.zeros_like(atom)
    constraint_atoms = (
        torch.stack((pose, block, atom), dim=2).reshape(-1, 1, 3).to(torch.int32)
    )
    params = torch.zeros(
        (n_poses * n_atoms, 5), dtype=torch.float32, device=pose_stack.device
    )
    params[:, 1:4] = pose_stack.coords.reshape(-1, 3)
    params[:, 4] = 1.0
    constraints = ConstraintSet.create_empty(pose_stack.device, n_poses)
    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic_coordinate, constraint_atoms, params
    )
    return attr.evolve(pose_stack, constraint_set=constraints)


def count_non_proton_ligand_chis(pose_stack: PoseStack, ligand_block: int) -> int:
    """Count named ligand chis whose outer atoms are not polar hydrogens.

    Rosetta params encode proton chis as a named chi ending at the rotating
    OH/SH/NH hydrogen. Checking both outer atoms also handles an equivalent
    reversed torsion declaration.
    """

    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    atom_is_hydrogen = (
        pose_stack.packed_block_types.atom_is_hydrogen[block_type_index]
        .to(torch.bool)
        .detach()
        .cpu()
        .numpy()
    )
    nchi = 0
    for torsion_index in range(block_type.n_torsions):
        atoms = block_type.ordered_torsions[torsion_index, :, 0]
        if numpy.any(atoms < 0):
            raise ValueError("ligand chi atoms must be local to the ligand block")
        if bool(atom_is_hydrogen[int(atoms[0])]) or bool(
            atom_is_hydrogen[int(atoms[3])]
        ):
            continue
        nchi += 1
    return nchi


def finalize_ga_docking(
    reference_pose_stack: PoseStack,
    ligand_block: int,
    backend,
    ga_result: GAResult,
    *,
    top_k: int = 1,
) -> DockingFinalization:
    """Refine top GA candidates and rank them by ``dG = dH + 0.4*nchi``."""

    if reference_pose_stack.n_poses != 1:
        raise ValueError("finalization requires one reference pose")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    npool = ga_result.ranked_energies.shape[1]
    if top_k > npool:
        raise ValueError(f"top_k ({top_k}) cannot exceed GA pool size ({npool})")
    if not hasattr(backend, "param_db"):
        raise TypeError("finalization requires a backend with a parameter database")

    promoted, genes = _promote_top_candidates(
        reference_pose_stack, ligand_block, ga_result, top_k
    )
    exact_sfxn = beta2016_score_function(promoted.device, param_db=backend.param_db)
    pre_refine = _score_pose_stack(promoted, exact_sfxn)
    coord_mask = torch.zeros(
        promoted.coords.shape[:2], dtype=torch.bool, device=promoted.device
    )
    coord_mask.scatter_(1, _ligand_atom_indices(promoted, ligand_block), True)
    refined = run_cart_min(
        promoted,
        exact_sfxn,
        coord_mask=coord_mask,
        optimizer_kwargs={"max_iter": 50},
    )
    complex_energy = _score_pose_stack(refined, exact_sfxn)

    ligand_only = _ligand_only_pose_stack(refined, ligand_block)
    constrained_ligand = _with_coordinate_constraints(ligand_only)
    constrained_sfxn = beta2016_score_function(
        promoted.device, param_db=backend.param_db
    )
    constrained_sfxn.set_weight(ScoreType.constraint, 1.0)
    relaxed_ligand = run_cart_min(
        constrained_ligand,
        constrained_sfxn,
        optimizer_kwargs={"max_iter": 30},
    )
    free_ligand_energy = _score_pose_stack(relaxed_ligand, exact_sfxn)

    receptor = _receptor_only_pose_stack(reference_pose_stack, ligand_block)
    free_receptor_energy = _score_pose_stack(receptor, exact_sfxn)[0]
    free_receptor_energies = free_receptor_energy.expand_as(complex_energy)

    nchi_value = count_non_proton_ligand_chis(reference_pose_stack, ligand_block)
    nchi = torch.full(
        complex_energy.shape,
        nchi_value,
        dtype=torch.int64,
        device=promoted.device,
    )
    TdS = 0.4 * nchi.to(complex_energy.dtype)
    dH = complex_energy - free_ligand_energy - free_receptor_energies
    dG = dH + TdS

    dG_matrix = dG.reshape(1, top_k)
    dG_order = torch.argsort(dG_matrix, dim=1)

    def rank(values: torch.Tensor) -> torch.Tensor:
        matrix = values.reshape(1, top_k)
        return matrix.gather(1, dG_order)

    ranked_genes = genes.gather(1, dG_order[:, :, None].expand(-1, -1, genes.shape[2]))
    ranked_flat = dG_order.reshape(-1)
    ranked_pose_stack = attr.evolve(
        refined, coords=refined.coords[ranked_flat].detach().clone()
    )
    return DockingFinalization(
        pose_stack=ranked_pose_stack,
        genes=ranked_genes,
        pre_refine_energies=rank(pre_refine),
        complex_energies=rank(complex_energy),
        free_ligand_energies=rank(free_ligand_energy),
        free_receptor_energies=rank(free_receptor_energies),
        nchi=rank(nchi),
        dH=rank(dH),
        TdS=rank(TdS),
        dG=rank(dG),
    )


__all__ = [
    "DockingFinalization",
    "count_non_proton_ligand_chis",
    "finalize_ga_docking",
]

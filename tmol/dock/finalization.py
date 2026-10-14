"""RosettaVS-style exact refinement and binding-score finalization."""

from __future__ import annotations

from dataclasses import dataclass

import attr
import numpy
import torch

from tmol.dock.dock_loop import _repeat_pose_stack
from tmol.dock.ga import (
    GAResult,
    LigandGeneLayout,
    _ligand_only_pose_stack,
    _movable_block_mapping,
    _place_ligand_in_reference,
    _select_inter_block_bondsep,
    _sidechain_context_blocks,
)
from tmol.dock.ligand_dofs import build_ligand_docking_dofs
from tmol.optimization import run_cart_min, run_kin_min
from tmol.pose.constraint_set import ConstraintSet
from tmol.pose.pdb_info import PDBInfo
from tmol.pose.pose_stack import PoseStack
from tmol.database import ParameterDatabase
from tmol.score import beta2016_score_function
from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm
from tmol.score.score_types import ScoreType


# Rosetta dockflex "bbsc1" final refinement: one hard-rep Cartesian minimization
# (fa_rep multiplier 1.0, MinMover tolerance 1e-5, up to 200 iterations). Used for
# the flexible branch; the rigid branch uses the torsion-space soft->hard schedule.
_DOCKFLEX_SCHEDULE = ((1.0, 1e-5, 200),)


@dataclass(frozen=True)
class DockingFinalization:
    """Exact-refined candidates and their RosettaVS ranking components.

    Candidate tensors have shape ``[1, top_k]`` and are sorted by the selected
    metric (exact complex energy by default, Rosetta's ``top_pose_metric="score"``)
    for the single input ligand. ``pose_stack`` uses
    the same candidate order.
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


def _refine_fa_rep_scaled(base_fa_ljrep, param_db, device, rep_mult):
    """Fresh beta2016 with scaled fa_ljrep and all other weights identical."""
    sfxn = beta2016_score_function(device, param_db=param_db)
    sfxn.set_weight(ScoreType.fa_ljrep, float(base_fa_ljrep) * float(rep_mult))
    return sfxn


def _soft_hard_refine(
    promoted, coord_mask, *, param_db, device, exact_sfxn, schedule, cycles
):
    """Run soft-to-hard cycles, accepting each pose's best exact-energy state."""
    # beta2016_score_function MEMOIZES when param_db is None, so a None db would
    # make the soft/hard/exact score functions ALIAS one object (set_weight on
    # "hard" would clobber "soft"). Resolve an explicit database so every stage
    # is an independent, freshly built score function.
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    base_fa_ljrep = float(exact_sfxn.get_weight(ScoreType.fa_ljrep))
    step_sfxns = [
        (
            _refine_fa_rep_scaled(base_fa_ljrep, param_db, device, rep_mult),
            tol,
            max_iter,
        )
        for rep_mult, tol, max_iter in schedule
    ]
    best_coords = promoted.coords.clone()
    best_energy = _score_pose_stack(promoted, exact_sfxn)
    # A non-finite pre-refine score (e.g. a severe clash scoring to +/-inf/NaN)
    # must not block accept-to-best: seed it to +inf so any finite refined-cycle
    # energy is accepted. A candidate that stays non-finite ranks last (dH/dG=inf)
    # rather than poisoning the sort with NaN.
    best_energy = torch.where(
        torch.isfinite(best_energy),
        best_energy,
        torch.full_like(best_energy, float("inf")),
    )
    current = promoted
    # Cartesian L-BFGS, not Rosetta torsion-space linmin; preserve the rep ramp
    # and accept-to-best. Omit Rosetta's preliminary cycle and global optH.
    for _ in range(cycles):
        for step_sfxn, tol, max_iter in step_sfxns:
            current = run_cart_min(
                current,
                step_sfxn,
                coord_mask=coord_mask,
                # LBFGS_Armijo accepts relative energy tolerance via rtol.
                optimizer_kwargs={"max_iter": max_iter, "rtol": tol},
            )
        energy = _score_pose_stack(current, exact_sfxn)
        improved = torch.isfinite(energy) & (energy < best_energy)
        best_coords = torch.where(
            improved[:, None, None], current.coords, best_coords
        )
        best_energy = torch.where(improved, energy, best_energy)
        current = attr.evolve(current, coords=best_coords.clone())
    return attr.evolve(promoted, coords=best_coords), best_energy


def _ligand_cartmin(
    pose_stack, exact_sfxn, ligand_coord_mask, *, max_iter=50, rtol=1e-4
):
    """Ligand-only Cartesian minimization at exact hard weights."""
    return run_cart_min(
        pose_stack,
        exact_sfxn,
        coord_mask=ligand_coord_mask,
        optimizer_kwargs={"max_iter": max_iter, "rtol": rtol},
    )


def _torsion_soft_hard_refine(
    promoted, ligand_block, *, param_db, device, exact_sfxn, schedule, cycles
):
    """Refine ligand jump and named torsions, accepting best exact energies."""
    # An explicit database keeps the soft/hard score functions independent.
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    base_fa_ljrep = float(exact_sfxn.get_weight(ScoreType.fa_ljrep))
    step_sfxns = [
        (
            _refine_fa_rep_scaled(base_fa_ljrep, param_db, device, rep_mult),
            tol,
            max_iter,
        )
        for rep_mult, tol, max_iter in schedule
    ]
    ldofs = build_ligand_docking_dofs(promoted, ligand_block, sidechain_blocks=())
    ff, mm = ldofs.fold_forest, ldofs.move_map
    best_coords = promoted.coords.clone()
    best_energy = _score_pose_stack(promoted, exact_sfxn)
    # Non-finite starting energies must not block accept-to-best.
    best_energy = torch.where(
        torch.isfinite(best_energy),
        best_energy,
        torch.full_like(best_energy, float("inf")),
    )
    current = promoted
    # Fixed-hydrogen harness: omit whole-pose optH (including His/Asn/Gln
    # flips) and its extra pre-optH torsional cycle. Ramp steps minimize only;
    # ligand rotamer/proton-chi packing is not reproduced.
    # Ignore ramp_repack_min's third (coordinate-constraint ramp) argument:
    # Rosetta dockrigid defaults ramp_down_constraints to false, so it is inert.
    for _ in range(cycles):
        for step_sfxn, tol, max_iter in step_sfxns:
            current = run_kin_min(
                current,
                step_sfxn,
                ff,
                mm,
                # rtol maps Rosetta's fractional-energy tolerance. In float32,
                # 1e-5 is below the optimizer's dtype floor (harmless warning);
                # the hard step then stops by gradient/atol/max_iter.
                optimizer_kwargs={"max_iter": max_iter, "rtol": tol},
                kin_dtype=promoted.coords.dtype,
            )
        energy = _score_pose_stack(current, exact_sfxn)
        improved = torch.isfinite(energy) & (energy < best_energy)
        best_coords = torch.where(
            improved[:, None, None], current.coords, best_coords
        )
        best_energy = torch.where(improved, energy, best_energy)
        current = attr.evolve(current, coords=best_coords.clone())
    return attr.evolve(promoted, coords=best_coords), best_energy


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
    *,
    sidechain_blocks: tuple[int, ...] = (),
    compact_sidechain_blocks: tuple[int, ...] = (),
    sidechain_atom_indices: torch.Tensor | None = None,
) -> tuple[PoseStack, torch.Tensor]:
    """Copy the top grid/GA movable coordinates into full complexes."""

    if reference_pose_stack.n_poses != 1:
        raise ValueError("finalization requires one reference pose")
    flat = torch.argsort(ga_result.pool.energies)[:top_k]

    if sidechain_blocks:
        promoted = _repeat_pose_stack(reference_pose_stack, top_k)
        selected_compact = attr.evolve(
            _repeat_pose_stack(ga_result.pool.pose_stack.split(0), promoted.n_poses),
            coords=ga_result.pool.pose_stack.coords[flat].detach().clone(),
        )
        promoted = _place_ligand_in_reference(
            promoted,
            ligand_block,
            selected_compact,
            sidechain_blocks=sidechain_blocks,
            compact_sidechain_blocks=compact_sidechain_blocks,
            sidechain_atom_indices=sidechain_atom_indices,
        )
        genes = ga_result.pool.genes[flat].unsqueeze(0)
        return promoted, genes

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


def _flexible_sidechain_mapping(
    reference_pose_stack: PoseStack,
    ligand_block: int,
    backend,
    compact_pose_stack: PoseStack,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    torch.Tensor | None,
    tuple[tuple[int, torch.Tensor], ...],
]:
    """Rebuild compact and full moving-sidechain coordinate mappings."""

    sidechain_blocks = tuple(getattr(backend, "sidechain_blocks", ()))
    if not sidechain_blocks:
        return (), (), None, ()
    # The compact topology (block types, offsets, sidechain atom indices) is
    # identical across every pool pose, so rebuild the mapping metadata from a
    # single-pose slice instead of running inverse kinematics + fp64 torsion
    # response checks over all npool sampled conformations.
    compact_pose_stack = compact_pose_stack.split(0)
    if getattr(backend, "ligand_block", ligand_block) != ligand_block:
        raise ValueError("finalization backend ligand block does not match reference")
    context_blocks = _sidechain_context_blocks(
        reference_pose_stack, ligand_block, sidechain_blocks
    )
    if tuple(getattr(backend, "context_blocks", ())) != context_blocks:
        raise ValueError("finalization peptide context does not match backend")
    source_blocks, compact_sidechain_blocks, _ = _movable_block_mapping(
        reference_pose_stack, ligand_block, sidechain_blocks
    )
    if compact_pose_stack.max_n_blocks != len(source_blocks):
        raise ValueError("GA pool compact topology does not match flexible backend")

    docking_dofs = build_ligand_docking_dofs(
        compact_pose_stack,
        0,
        sidechain_blocks=compact_sidechain_blocks,
    )
    layout = LigandGeneLayout.from_pose_stack(compact_pose_stack, docking_dofs)
    compact_atoms = layout.sidechain_atom_indices[0]
    if not bool(torch.all(layout.sidechain_atom_indices == compact_atoms[None, :])):
        raise ValueError("finalization requires identical sidechain atom mappings")

    block_local_atoms = []
    for compact_block, source_block in zip(
        compact_sidechain_blocks, sidechain_blocks
    ):
        compact_start = compact_pose_stack.block_coord_offset64[0, compact_block]
        compact_end = compact_start + compact_pose_stack.block_type(
            0, compact_block
        ).n_atoms
        selected = compact_atoms[
            (compact_atoms >= compact_start) & (compact_atoms < compact_end)
        ]
        if selected.numel() != 0:
            block_local_atoms.append((source_block, selected - compact_start))
    if sum(local_atoms.numel() for _, local_atoms in block_local_atoms) != int(
        compact_atoms.numel()
    ):
        raise ValueError("sidechain atom mapping is incomplete")
    return (
        sidechain_blocks,
        compact_sidechain_blocks,
        compact_atoms,
        tuple(block_local_atoms),
    )


def _mapped_atom_indices(
    pose_stack: PoseStack,
    block_local_atoms: tuple[tuple[int, torch.Tensor], ...],
) -> torch.Tensor:
    """Return per-pose absolute atom indices for block-local atom selections."""

    return torch.cat(
        [
            pose_stack.block_coord_offset64[:, block, None]
            + local_atoms.to(device=pose_stack.device)[None, :]
            for block, local_atoms in block_local_atoms
        ],
        dim=1,
    )


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
    inter_block_bondsep = _select_inter_block_bondsep(pose_stack, keep_blocks)
    chain_id64 = pose_stack.chain_id64[:, keep_blocks].clone()

    return PoseStack(
        packed_block_types=pbt,
        coords=coords,
        block_coord_offset=offsets64.to(torch.int32),
        block_coord_offset64=offsets64,
        inter_residue_connections=connections64.to(torch.int32),
        inter_residue_connections64=connections64,
        inter_block_bondsep=inter_block_bondsep,
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
    refine_schedule: tuple[tuple[float, float, int], ...] = (
        (0.02, 0.01, 50),
        (1.0, 1e-5, 50),
    ),
    refine_cycles: int = 3,
    top_pose_metric: str = "score",
) -> DockingFinalization:
    """Soft-to-hard refine candidates and rank by dG or exact complex score.

    ``refine_schedule`` contains (rep multiplier, relative energy tolerance,
    max iterations) steps repeated ``refine_cycles`` times. ``top_pose_metric``
    selects ``"dG"`` (``dH + 0.4*nchi``) or ``"score"`` (exact complex energy).
    """

    if reference_pose_stack.n_poses != 1:
        raise ValueError("finalization requires one reference pose")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if top_pose_metric not in ("dG", "score"):
        raise ValueError('top_pose_metric must be "dG" or "score"')
    npool = ga_result.ranked_energies.shape[1]
    if top_k > npool:
        raise ValueError(f"top_k ({top_k}) cannot exceed GA pool size ({npool})")
    if not hasattr(backend, "param_db"):
        raise TypeError("finalization requires a backend with a parameter database")

    (
        sidechain_blocks,
        compact_sidechain_blocks,
        compact_sidechain_atoms,
        block_local_atoms,
    ) = _flexible_sidechain_mapping(
        reference_pose_stack, ligand_block, backend, ga_result.pool.pose_stack
    )
    if sidechain_blocks:
        promoted, genes = _promote_top_candidates(
            reference_pose_stack,
            ligand_block,
            ga_result,
            top_k,
            sidechain_blocks=sidechain_blocks,
            compact_sidechain_blocks=compact_sidechain_blocks,
            sidechain_atom_indices=compact_sidechain_atoms,
        )
    else:
        promoted, genes = _promote_top_candidates(
            reference_pose_stack, ligand_block, ga_result, top_k
        )
    exact_sfxn = beta2016_score_function(promoted.device, param_db=backend.param_db)
    pre_refine = _score_pose_stack(promoted, exact_sfxn)
    coord_mask = torch.zeros(
        promoted.coords.shape[:2], dtype=torch.bool, device=promoted.device
    )
    coord_mask.scatter_(1, _ligand_atom_indices(promoted, ligand_block), True)
    if sidechain_blocks:
        coord_mask.scatter_(
            1, _mapped_atom_indices(promoted, block_local_atoms), True
        )
    if sidechain_blocks:
        # FLEX: Rosetta dockflex uses a Cartesian "bbsc1" refinement = a single
        # hard-rep minimization (repeat 1, fa_rep 1.0, tol 1e-5, <=200 iter), NOT
        # a soft->hard ramp. Reproduce that here (ligand + flexible-sidechain atoms
        # move in Cartesian space; accept-to-best as in Rosetta).
        refined, complex_energy = _soft_hard_refine(
            promoted,
            coord_mask,
            param_db=backend.param_db,
            device=promoted.device,
            exact_sfxn=exact_sfxn,
            schedule=_DOCKFLEX_SCHEDULE,
            cycles=1,
        )
    else:
        # RIGID: bracket torsion-space soft-to-hard with ligand Cartesian min.
        ligand_mask = coord_mask  # No flexible sidechains: ligand atoms only.
        promoted = _ligand_cartmin(promoted, exact_sfxn, ligand_mask)
        promoted, _ = _torsion_soft_hard_refine(
            promoted,
            ligand_block,
            param_db=backend.param_db,
            device=promoted.device,
            exact_sfxn=exact_sfxn,
            schedule=refine_schedule,
            cycles=refine_cycles,
        )
        refined = _ligand_cartmin(promoted, exact_sfxn, ligand_mask)
        # Rigid docking fixes the receptor by construction, but run_kin_min
        # refolds the whole pose from internal DOFs and float32 forward
        # kinematics near the ligand root jump leaves ~1e-3 A noise on the
        # nominally fixed receptor atoms. Restore every non-ligand atom to the
        # reference so only the ligand is refined (ligand_mask marks the ligand).
        keep_ligand = ligand_mask.unsqueeze(-1)
        refined = attr.evolve(
            refined,
            coords=torch.where(
                keep_ligand,
                refined.coords,
                reference_pose_stack.coords.expand(refined.n_poses, -1, -1),
            ),
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
    if sidechain_blocks:
        real_blocks = torch.nonzero(
            reference_pose_stack.block_type_ind64[0] >= 0, as_tuple=False
        ).flatten()
        keep_blocks = real_blocks[real_blocks != ligand_block].tolist()
        source_to_receptor = {
            source_block: receptor_block
            for receptor_block, source_block in enumerate(keep_blocks)
        }
        receptor_block_local_atoms = tuple(
            (source_to_receptor[source_block], local_atoms)
            for source_block, local_atoms in block_local_atoms
        )
        receptor_coord_mask = torch.zeros(
            receptor.coords.shape[:2], dtype=torch.bool, device=receptor.device
        )
        receptor_coord_mask.scatter_(
            1,
            _mapped_atom_indices(receptor, receptor_block_local_atoms),
            True,
        )
        receptor = run_cart_min(
            receptor,
            exact_sfxn,
            coord_mask=receptor_coord_mask,
            optimizer_kwargs={"max_iter": 50},
        )
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

    sort_key = dG if top_pose_metric == "dG" else complex_energy
    sort_order = torch.argsort(sort_key.reshape(1, top_k), dim=1)

    def rank(values: torch.Tensor) -> torch.Tensor:
        matrix = values.reshape(1, top_k)
        return matrix.gather(1, sort_order)

    ranked_genes = genes.gather(1, sort_order[:, :, None].expand(-1, -1, genes.shape[2]))
    ranked_flat = sort_order.reshape(-1)
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

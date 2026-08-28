import torch
import time

from tmol.pose import PoseStack
from tmol.score import ScoreFunction

from tmol.pack import (
    PackerTask,
    SetPackerTask,
    PackerEnergyTables,
    run_simulated_annealing,
    impose_top_rotamer_assignments,
)
from tmol.pack.rotamer import build_rotamers


def pack_rotamers(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    task: PackerTask,
    verbose: bool = False,
) -> PoseStack:
    """Optimize side-chain conformers for a pose stack.

    Args:
        pose_stack: Poses whose task-enabled blocks will be packed.
        sfxn: Score function used to rank rotamer assignments.
        task: Allowed block types, conformers, and packing positions.
        verbose: Print synchronized stage timings when true.

    Returns:
        A new pose stack containing the lowest-ranked assignment per pose.
    """

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    task = SetPackerTask.from_packer_task(task)
    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()

    (
        packer_energy_tables,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        end_time2,
        end_time3,
    ) = _calculate_packer_energies(pose_stack, sfxn, rotamer_set, task, verbose=verbose)

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time4 = time.perf_counter()

    _, rotamer_assignments = run_simulated_annealing(packer_energy_tables)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time5 = time.perf_counter()

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack,
        rotamer_set,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        rotamer_assignments,
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time6 = time.perf_counter()

    if verbose:
        print(
            f"pack_rotamers {end_time6 - start_time: .2f}"
            + f" build rots: {end_time1 - start_time: .2f} calcRPEs: {end_time2 - end_time1: .2f}"
            + f" build IG: {end_time3 - end_time2: .2f} build IG part2: {end_time4 - end_time3: .2f}"
            + f" run SA: {end_time5 - end_time4: .2f} pose ctor: {end_time6 - end_time5: .2f}"
        )

    return new_pose_stack


def _calculate_packer_energies(pose_stack, sfxn, rotamer_set, task, verbose=False):
    from tmol.pack.compiled import build_interaction_graph

    pbt = pose_stack.packed_block_types
    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()

    chunk_size = 16

    (
        max_n_bump_checked_rotamers_per_pose_tensor,
        n_molten_blocks_per_pose,
        n_bc_rots_per_pose,
        bc_rot_offset_for_pose,
        n_bc_rots_for_molten_block,
        bc_rot_offset_for_molten_block,
        molten_block_ind_for_bc_rot,
        rotamer_for_nonmolten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        energy1b,
        chunk_pair_offset_for_block_pair,
        chunk_pair_offset,
        energy2b,
    ) = build_interaction_graph(
        task.bump_check,
        chunk_size,
        pbt.n_types,
        rotamer_set.n_rots_for_pose,
        rotamer_set.rot_offset_for_pose,
        rotamer_set.n_rots_for_block,
        rotamer_set.rot_offset_for_block,
        rotamer_set.pose_for_rot,
        rotamer_set.block_type_ind_for_rot,
        rotamer_set.block_ind_for_rot,
        energies.indices().to(torch.int32),
        energies.values(),
        verbose,
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    packer_energy_tables = PackerEnergyTables(
        max_n_rotamers_per_pose=max_n_bump_checked_rotamers_per_pose_tensor.item(),
        pose_n_res=n_molten_blocks_per_pose,
        pose_n_rotamers=n_bc_rots_per_pose,
        pose_rotamer_offset=bc_rot_offset_for_pose,
        nrotamers_for_res=n_bc_rots_for_molten_block,
        oneb_offsets=bc_rot_offset_for_molten_block,
        res_for_rot=molten_block_ind_for_bc_rot,
        chunk_size=chunk_size,
        chunk_offset_offsets=chunk_pair_offset_for_block_pair,
        chunk_offsets=chunk_pair_offset,
        energy1b=energy1b,
        energy2b=energy2b,
    )

    return (
        packer_energy_tables,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        end_time2,
        end_time3,
    )

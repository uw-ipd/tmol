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
from tmol.pack._impose_rotamers import chosen_rotamer_for_block
from tmol.pack.rotamer import build_rotamers
from tmol.pack.rotamer._conjugated_groups import write_group_members


def pack_rotamers(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    task: PackerTask,
    verbose=False,
    **sa_params,
):

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
        collapse,
        _bg_bg_energies,
        end_time2,
        end_time3,
    ) = _calculate_packer_energies(pose_stack, sfxn, rotamer_set, task, verbose=verbose)

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time4 = time.perf_counter()

    scores, rotamer_assignments = run_simulated_annealing(
        packer_energy_tables, **sa_params
    )
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
    if collapse is not None:
        # the packer chose for the representative; its group moves with it
        assignment = chosen_rotamer_for_block(
            pose_stack,
            rotamer_for_nonmolten_block,
            n_molten_blocks_per_pose,
            bc_rot_offset_for_molten_block,
            bc_rot_to_orig_rot,
            rotamer_assignments[:, 0, :],
        )
        new_pose_stack = write_group_members(
            new_pose_stack, rotamer_set, collapse, assignment
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
    from tmol.pack.rotamer._conjugated_groups import (
        collapse_group_energies,
        collapse_group_rotamers,
    )

    pbt = pose_stack.packed_block_types
    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    # a group of covalently joined blocks is one choice, not several: fold its
    #    members onto a representative so the packer cannot pick a conformer
    #    for one that disagrees with its neighbour
    groups = rotamer_set.correlated_groups
    collapsed = collapse_group_rotamers(pose_stack, rotamer_set, groups)
    if collapsed is not None:
        collapse, group_of_rot, conformer_of_rot = collapsed
        energies = collapse_group_energies(
            energies, collapse, group_of_rot, conformer_of_rot
        )
        n_rots_for_block = collapse.compact_n_rots_for_block
        rot_offset_for_block = collapse.compact_rot_offset_for_block
        block_ind_for_rot = collapse.compact_block_ind_for_rot
        pose_for_rot = collapse.compact_pose_for_rot
        block_type_ind_for_rot = collapse.compact_block_type_ind_for_rot
        n_rots_for_pose = collapse.compact_n_rots_for_pose
        rot_offset_for_pose = collapse.compact_rot_offset_for_pose
    else:
        collapse = None
        n_rots_for_block = rotamer_set.n_rots_for_block
        rot_offset_for_block = rotamer_set.rot_offset_for_block
        block_ind_for_rot = rotamer_set.block_ind_for_rot
        pose_for_rot = rotamer_set.pose_for_rot
        block_type_ind_for_rot = rotamer_set.block_type_ind_for_rot
        n_rots_for_pose = rotamer_set.n_rots_for_pose
        rot_offset_for_pose = rotamer_set.rot_offset_for_pose

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
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_type_ind_for_rot,
        block_ind_for_rot,
        energies.indices().to(torch.int32),
        energies.values(),
        verbose,
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if collapse is not None:
        # the graph was built on the compacted numbering; everything downstream
        #    addresses the rotamer set itself, so hand back original indices
        bc_rot_to_orig_rot = collapse.compact_to_orig[bc_rot_to_orig_rot]
        is_bg = rotamer_for_nonmolten_block != -1
        rotamer_for_nonmolten_block = torch.where(
            is_bg,
            collapse.compact_to_orig[rotamer_for_nonmolten_block.clamp(min=0)],
            rotamer_for_nonmolten_block,
        )

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
        collapse,
        bg_bg_energies,
        end_time2,
        end_time3,
    )
